"""Stable Diffusion 1.5 inpainting handler (pretrained or with a PlantSR LoRA adapter).

Model: stable-diffusion-v1-5/stable-diffusion-inpainting (CreativeML OpenRAIL-M). Its UNet
takes 9 input channels: 4 noisy latents + 1 mask + 4 latents of the masked image. White
(1) in the mask means "repaint", which matches the team convention.

The model was trained at 512 x 512, so 256 x 256 benchmark crops are upsampled to the
working resolution, inpainted, resized back and composited:
    final = (1 - mask) * input + mask * generated
Visible pixels are therefore copied exactly; only the missing region comes from the model.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from .interface import TaskHandler, free_gpu_memory

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_PROMPT = "a high-resolution photo of a plant, natural leaves, sharp focus"
DEFAULT_NEGATIVE = "blurry, text, watermark, frame, border, cartoon, painting"
FILL_VALUE = 128


def pick_device(prefer: str | None = None):
    import torch

    if prefer:
        device = torch.device(prefer)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device, dtype


def load_inpaint_pipeline(model_id: str = DEFAULT_MODEL_ID, device=None, dtype=None, scheduler: str = "dpm",
                          local_files_only: bool = False):
    """Load the inpainting pipeline (fp16 weights on GPU when available, no safety checker).

    The safety checker is disabled because it blacks out false positives, which would corrupt
    benchmark metrics on plant photos; inputs are user photos of plants, not text prompts.
    """
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

    if device is None or dtype is None:
        auto_device, auto_dtype = pick_device()
        device = device or auto_device
        dtype = dtype or auto_dtype
    kwargs = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
                  local_files_only=local_files_only)
    pipe = None
    warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")  # newer diffusers prefer `dtype`
    if dtype == torch.float16:
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, variant="fp16", **kwargs)
        except (OSError, ValueError):
            pipe = None  # no fp16 variant files: load the default weights and cast
    if pipe is None:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, **kwargs)
    if scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def snapshot_revision(pipe) -> str:
    """Hugging Face commit hash of the loaded base weights, when loaded from the hub cache."""
    path = str(getattr(pipe.unet.config, "_name_or_path", ""))
    match = re.search(r"snapshots[\\/]([0-9a-f]{40})", path)
    return match.group(1)[:12] if match else "local"


def model_size(width: int, height: int, work_res: int, multiple: int = 8) -> tuple[int, int]:
    """Scale so the long side equals `work_res`, rounding both sides to a multiple of 8."""
    scale = work_res / max(width, height)
    w = max(multiple, int(round(width * scale / multiple)) * multiple)
    h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return w, h


def lora_file(lora_path) -> Path:
    path = Path(lora_path)
    if path.is_dir():
        candidates = sorted(path.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors LoRA file in {path}")
        return candidates[0]
    return path


def short_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class StableDiffusionInpaint(TaskHandler):
    task = "inpainting"
    uses_gpu = True
    needs_mask = True
    generative = True

    def __init__(self, method: str = "sd15_pretrained", model_id: str = DEFAULT_MODEL_ID, lora_path=None,
                 work_res: int = 512, prompt: str = DEFAULT_PROMPT, negative_prompt: str = DEFAULT_NEGATIVE,
                 steps: int = 25, guidance: float = 5.0, label: str | None = None, pipe=None):
        self.method = method
        self.model_id = model_id
        self.lora_path = lora_path
        self.work_res = work_res
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.guidance = guidance
        self.pipe = pipe
        self.label = label or ("SD 1.5 inpainting + PlantSR LoRA" if lora_path else "SD 1.5 inpainting (pretrained)")
        self.description = (f"Latent diffusion inpainting at {work_res}px working resolution, "
                            f"{steps} steps, guidance {guidance}. Visible pixels are kept exactly.")
        super().__init__()
        self._refresh_version()
        self._loaded = pipe is not None

    def settings(self) -> dict:
        return {"model_id": self.model_id, "lora_path": str(self.lora_path or ""), "work_res": self.work_res,
                "prompt": self.prompt, "negative_prompt": self.negative_prompt, "steps": self.steps,
                "guidance": self.guidance, "scheduler": "DPMSolverMultistep"}

    def load(self) -> None:
        if self.pipe is None:
            self.pipe = load_inpaint_pipeline(self.model_id)
            if self.lora_path:
                self.pipe.load_lora_weights(str(Path(self.lora_path)), adapter_name="plantsr")
        self._refresh_version()
        self._loaded = True

    def _refresh_version(self) -> None:
        base = f"{self.model_id}@{snapshot_revision(self.pipe)}" if self.pipe is not None else self.model_id
        self.version = base + (f"+lora:{short_hash(lora_file(self.lora_path))}" if self.lora_path else "")

    def unload(self) -> None:
        self.pipe = None
        self._loaded = False
        free_gpu_memory()

    def generate(self, image: Image.Image, mask: np.ndarray, seed: int) -> Image.Image:
        """Raw model output at the input size (before compositing)."""
        import torch

        width, height = image.size
        mw, mh = model_size(width, height, self.work_res)
        rgb = np.asarray(image).copy()
        rgb[mask > 0] = FILL_VALUE  # never let pixels under the mask reach the model
        model_image = Image.fromarray(rgb).resize((mw, mh), Image.BICUBIC)
        model_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((mw, mh), Image.NEAREST)
        generator = torch.Generator("cpu").manual_seed(int(seed))
        kwargs = dict(prompt=self.prompt, image=model_image, mask_image=model_mask, height=mh, width=mw,
                      num_inference_steps=self.steps, guidance_scale=self.guidance, generator=generator,
                      strength=1.0)
        if self.guidance > 1.0 and self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt
        with torch.inference_mode():
            generated = self.pipe(**kwargs).images[0]
        return generated.resize((width, height), Image.BICUBIC)

    def run(self, image, mask, seed):
        generated = self.generate(image, mask, seed)
        out = np.where(mask[..., None] > 0, np.asarray(generated), np.asarray(image)).astype(np.uint8)
        return Image.fromarray(out), model_size(image.width, image.height, self.work_res)
