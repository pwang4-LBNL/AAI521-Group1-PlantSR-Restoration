"""Offline PlantSR ×4 image inference and the team's Flask registry adapter."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .models import forward_sr, load_bundle

METHODS = {"hat_l": "HAT-L", "real_esrgan": "Real-ESRGAN"}
SCALE = 4


def as_rgb(image):
    """Orient a PIL upload and composite any transparency over white."""
    if not isinstance(image, Image.Image):
        raise TypeError("Pass a PIL.Image.Image object.")
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "white")
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return image.convert("RGB")


def _load(method, package_dir, device):
    if method not in METHODS:
        raise ValueError(f"Choose a method from {list(METHODS)}.")
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    model, bundle = load_bundle(root / "weights" / f"{method}.pt")
    epoch = int(bundle["epoch"])
    if bundle["model_metadata"]["name"] != method or bundle["phase"] != "fine_tuned":
        raise ValueError(f"Unexpected checkpoint for {method}.")
    if method == "real_esrgan" and epoch != 10:
        raise ValueError("Expected the Real-ESRGAN epoch-10 checkpoint.")
    return model.eval().requires_grad_(False).to(device), epoch


def _predict(tile, model, method, device):
    pixels = np.asarray(tile, dtype=np.float32)
    height, width = pixels.shape[:2]
    if method == "hat_l":
        # HAT shifts 16-pixel windows. Edge padding also supports one-pixel inputs.
        pixels = np.pad(pixels, ((0, -height % 16), (0, -width % 16), (0, 0)), mode="edge")
    batch = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0).div_(255).to(device)
    output = forward_sr(model, method, batch)
    if not torch.isfinite(output).all():
        raise FloatingPointError(f"{method} produced non-finite pixels.")
    output = output[0, :, :height * SCALE, :width * SCALE].clamp(0, 1).mul(255).round().byte()
    image = Image.fromarray(output.permute(1, 2, 0).contiguous().cpu().numpy())
    return image, (pixels.shape[1], pixels.shape[0])


def _restore(image, predict):
    """Return a full-field RGB image and the largest padded tile size."""
    image = as_rgb(image)
    width, height = image.size
    core, halo = 128, 16
    if max(width, height) <= core:
        return predict(image)
    output = Image.new("RGB", (SCALE * width, SCALE * height))
    largest = (0, 0)
    for top in range(0, height, core):
        for left in range(0, width, core):
            right, bottom = min(left + core, width), min(top + core, height)
            x, y = max(0, left - halo), max(0, top - halo)
            patch = image.crop((x, y, min(width, right + halo), min(height, bottom + halo)))
            up, size = predict(patch)
            largest = (max(largest[0], size[0]), max(largest[1], size[1]))
            inner = (SCALE * (left-x), SCALE * (top-y), SCALE * (right-x), SCALE * (bottom-y))
            output.paste(up.crop(inner), (SCALE * left, SCALE * top))
    return output, largest


class Predictor:
    """Load one checkpoint once for repeated requests."""
    def __init__(self, method="hat_l", *, package_dir=None, device=None):
        self.method = method
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.epoch = _load(method, package_dir, self.device)

    @torch.inference_mode()
    def with_tile_size(self, image):
        return _restore(image, lambda tile: _predict(tile, self.model, self.method, self.device))

    def __call__(self, image):
        return self.with_tile_size(image)[0]


def restore_sr(image, method="hat_l", *, package_dir=None, device=None):
    """Upscale one PIL image. Use Predictor to reuse the model across images."""
    return Predictor(method, package_dir=package_dir, device=device)(image)


def make_handler(method="hat_l", *, package_dir=None, device=None):
    """Build a lazy TaskHandler for the shared Flask registry."""
    from src.inference.interface import TaskHandler

    if method not in METHODS:
        raise ValueError(f"Choose a method from {list(METHODS)}.")

    class PlantSRHandler(TaskHandler):
        task = "super_resolution"
        uses_gpu = True

        def __init__(self):
            super().__init__()
            self.method = method
            self.label = f"{METHODS[method]} ×4 (PlantSR)"
            self.description = "Upscales the full image. Large inputs use overlapping tiles."
            self.predictor = None
            self.version = method

        def load(self):
            self.predictor = Predictor(method, package_dir=package_dir, device=device)
            self.version = f"{method}-epoch-{self.predictor.epoch}"
            self._loaded = True

        def unload(self):
            self.predictor = None
            self._loaded = False

        def run(self, image, mask=None, seed=42):
            if self.predictor is None:
                self.load()
            return self.predictor.with_tile_size(image)

    return PlantSRHandler()
