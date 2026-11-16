"""Common inference interface shared by the four restoration tasks.

    result = restore(image, task, mask=None, seed=42)            # agreed team signature
    result = restore(image, task, mask=None, seed=42, method="sd15_finetuned")  # optional extra

Each task owner implements a `TaskHandler` subclass (see docs/inference_interface.md).
The `Registry` keeps at most `max_gpu_models` GPU models loaded, serialises GPU work with a
lock, times every call and converts failures into a result object instead of an exception.
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TASKS = ("denoising", "super_resolution", "colorization", "inpainting")
TASK_LABELS = {
    "denoising": "Denoising",
    "super_resolution": "Super-resolution (4x)",
    "colorization": "Colorization",
    "inpainting": "Inpainting",
}


@dataclass
class RestoreResult:
    image: Image.Image | None
    task: str
    method: str
    model_version: str
    input_size: tuple[int, int]          # (width, height) received
    processed_size: tuple[int, int]      # (width, height) the model actually ran at
    output_size: tuple[int, int]         # (width, height) of the returned image
    elapsed_seconds: float
    success: bool
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        """JSON-safe summary (everything except the image)."""
        return {
            "task": self.task, "method": self.method, "model_version": self.model_version,
            "input_size": list(self.input_size), "processed_size": list(self.processed_size),
            "output_size": list(self.output_size), "elapsed_seconds": round(self.elapsed_seconds, 3),
            "success": self.success, "error": self.error, **self.extra,
        }


class TaskHandler:
    """Base class for one restoration method of one task.

    Subclasses set the class attributes and implement `run`; heavy models also implement
    `load`/`unload` so the registry can move them on and off the GPU.
    """

    task: str = ""
    method: str = ""
    version: str = ""
    label: str = ""
    description: str = ""
    uses_gpu: bool = False
    needs_mask: bool = False
    generative: bool = False  # True when the seed changes the output

    def __init__(self):
        self._loaded = not self.uses_gpu

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = not self.uses_gpu

    def run(self, image: Image.Image, mask: np.ndarray | None, seed: int):
        """Return (output PIL image, processed (width, height))."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.method, "label": self.label or self.method, "version": self.version,
                "description": self.description, "uses_gpu": self.uses_gpu,
                "generative": self.generative, "loaded": self.loaded}


def normalize_image(image) -> Image.Image:
    """Accept a path, PIL image or array and return an RGB PIL image."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as src:
            image = ImageOps.exif_transpose(src)
            image.load()
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        image = Image.fromarray(arr)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image).__name__}")
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def normalize_mask(mask, size: tuple[int, int]) -> np.ndarray:
    """Return a uint8 {0,1} mask of shape (height, width); 1 = missing/reconstruct."""
    if isinstance(mask, (str, Path)):
        with Image.open(mask) as src:
            mask = src.convert("L")
            mask.load()
    if isinstance(mask, Image.Image):
        if mask.mode in ("RGBA", "LA"):
            alpha = np.asarray(mask.getchannel("A"))
            gray = np.asarray(mask.convert("L"))
            mask = Image.fromarray(np.where(alpha > 0, gray, 0).astype(np.uint8))
        if mask.size != size:
            mask = mask.convert("L").resize(size, Image.NEAREST)
        arr = np.asarray(mask.convert("L")) > 127
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        threshold = 127 if arr.dtype == np.uint8 and arr.max() > 1 else 0.5
        arr = arr > threshold
        if arr.shape != (size[1], size[0]):
            arr = np.asarray(Image.fromarray(arr.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return arr.astype(np.uint8)


class Registry:
    def __init__(self, max_gpu_models: int = 1):
        self.max_gpu_models = max_gpu_models
        self._handlers: dict[str, dict[str, TaskHandler]] = {task: {} for task in TASKS}
        self._defaults: dict[str, str] = {}
        self._lock = threading.RLock()
        self._last_used: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------ registration
    def register(self, handler: TaskHandler, default: bool = False) -> TaskHandler:
        if handler.task not in TASKS:
            raise ValueError(f"Unknown task {handler.task!r}")
        self._handlers[handler.task][handler.method] = handler
        if default or handler.task not in self._defaults:
            self._defaults[handler.task] = handler.method
        return handler

    def set_default(self, task: str, method: str) -> None:
        self.handler(task, method)  # validates the pair
        self._defaults[task] = method

    def handler(self, task: str, method: str | None = None) -> TaskHandler:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; choose one of {TASKS}")
        methods = self._handlers[task]
        if not methods:
            raise ValueError(f"No method is registered for {task!r}")
        method = method or self._defaults[task]
        if method not in methods:
            raise ValueError(f"Unknown method {method!r} for {task!r}; available: {sorted(methods)}")
        return methods[method]

    def describe(self) -> list[dict]:
        return [{
            "task": task, "label": TASK_LABELS[task],
            "needs_mask": any(h.needs_mask for h in self._handlers[task].values()),
            "default_method": self._defaults.get(task),
            "methods": [h.describe() for h in self._handlers[task].values()],
        } for task in TASKS]

    # ------------------------------------------------------------------ GPU memory
    def _ensure_loaded(self, target: TaskHandler) -> None:
        if target.loaded:
            return
        if target.uses_gpu:
            loaded = [h for methods in self._handlers.values() for h in methods.values()
                      if h.uses_gpu and h.loaded and h is not target]
            loaded.sort(key=lambda h: self._last_used.get((h.task, h.method), 0.0))
            while len(loaded) >= self.max_gpu_models:
                loaded.pop(0).unload()
                free_gpu_memory()
        target.load()

    def unload_all(self) -> None:
        with self._lock:
            for methods in self._handlers.values():
                for h in methods.values():
                    if h.uses_gpu and h.loaded:
                        h.unload()
            free_gpu_memory()

    # ------------------------------------------------------------------ inference
    def restore(self, image, task: str, mask=None, seed: int = 42, method: str | None = None) -> RestoreResult:
        start = time.perf_counter()
        input_size = (0, 0)
        handler = None
        try:
            handler = self.handler(task, method)
            pil = normalize_image(image)
            input_size = pil.size
            mask_arr = None
            if handler.needs_mask:
                if mask is None:
                    raise ValueError("This task needs a mask (white = area to fill).")
                mask_arr = normalize_mask(mask, pil.size)
                if not mask_arr.any():
                    raise ValueError("The mask is empty. Paint or upload the area to fill (white = missing).")
                if mask_arr.all():
                    raise ValueError("The mask covers the whole image; leave some context visible.")
            with self._lock:
                self._ensure_loaded(handler)
                self._last_used[(handler.task, handler.method)] = time.time()
                output, processed = handler.run(pil, mask_arr, int(seed))
            output = normalize_image(output)
            extra = {"mask_coverage": round(float(mask_arr.mean()), 5)} if mask_arr is not None else {}
            return RestoreResult(output, task, handler.method, handler.version, input_size,
                                 tuple(processed), output.size, time.perf_counter() - start, True, None, extra)
        except Exception as exc:  # noqa: BLE001 - report every failure to the caller
            message = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
            if "out of memory" in str(exc).lower():
                message = "GPU out of memory. Try a smaller image or retry after other jobs finish. " + message
                if handler is not None and handler.uses_gpu:
                    handler.unload()
                free_gpu_memory()
            return RestoreResult(None, task, getattr(handler, "method", method or ""),
                                 getattr(handler, "version", ""), input_size, (0, 0), (0, 0),
                                 time.perf_counter() - start, False, message)


def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


_default_registry: Registry | None = None


def set_default_registry(registry: Registry) -> None:
    global _default_registry
    _default_registry = registry


def restore(image, task: str, mask=None, seed: int = 42, method: str | None = None) -> RestoreResult:
    """Team-wide entry point. Uses the registry configured with `set_default_registry`."""
    if _default_registry is None:
        raise RuntimeError("No registry configured; call set_default_registry(build_registry(...)) first.")
    return _default_registry.restore(image, task, mask=mask, seed=seed, method=method)
