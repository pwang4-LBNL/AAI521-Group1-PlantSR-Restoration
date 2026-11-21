"""Simple, CPU-only baselines for all four tasks.

They follow the team evaluation table (README section 7) and let the application run every
task end to end before the fine-tuned models are integrated. Each one is labelled as a
baseline in the UI so it is never mistaken for a learned model.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .interface import TaskHandler


def _fit(image: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


class OpenCVInpaint(TaskHandler):
    task = "inpainting"
    needs_mask = True

    def __init__(self, flag: str = "telea", radius: int = 5):
        self.flag = flag
        self.radius = radius
        self.method = f"opencv_{flag}"
        self.label = {"telea": "OpenCV Telea (baseline)", "ns": "OpenCV Navier-Stokes (baseline)"}[flag]
        self.version = f"opencv-{cv2.__version__}-{flag}-r{radius}"
        self.description = "Classical diffusion-based fill from the hole boundary. No learning."
        super().__init__()

    def run(self, image, mask, seed):
        rgb = np.asarray(image)
        flag = cv2.INPAINT_TELEA if self.flag == "telea" else cv2.INPAINT_NS
        filled = cv2.inpaint(rgb, (mask > 0).astype(np.uint8) * 255, self.radius, flag)
        out = np.where(mask[..., None] > 0, filled, rgb).astype(np.uint8)
        return Image.fromarray(out), image.size


class NLMeansDenoise(TaskHandler):
    task = "denoising"
    method = "opencv_nlmeans"
    label = "Non-local means (baseline)"
    description = "OpenCV fastNlMeansDenoisingColored (h=10). Large images are reduced to 1024 px."

    def __init__(self, strength: int = 10, max_side: int = 1024):
        self.strength, self.max_side = strength, max_side
        self.version = f"opencv-{cv2.__version__}-nlmeans-h{strength}"
        super().__init__()

    def run(self, image, mask, seed):
        work = _fit(image, self.max_side)
        out = cv2.fastNlMeansDenoisingColored(np.asarray(work), None, self.strength, self.strength, 7, 21)
        return Image.fromarray(out), work.size


class BicubicSR(TaskHandler):
    task = "super_resolution"
    method = "bicubic_x4"
    label = "Bicubic x4 (baseline)"
    description = "Bicubic 4x upscaling. Inputs larger than 512 px are reduced first."

    def __init__(self, scale: int = 4, max_input_side: int = 512):
        self.scale, self.max_input_side = scale, max_input_side
        self.version = f"pil-bicubic-x{scale}"
        super().__init__()

    def run(self, image, mask, seed):
        work = _fit(image, self.max_input_side)
        out = work.resize((work.width * self.scale, work.height * self.scale), Image.BICUBIC)
        return out, work.size


class GrayscaleColorization(TaskHandler):
    task = "colorization"
    method = "grayscale_rgb"
    label = "Grayscale as RGB (weak baseline)"
    version = "pil-grayscale"
    description = "Returns the luminance replicated to three channels. A deliberately weak reference."

    def run(self, image, mask, seed):
        return image.convert("L").convert("RGB"), image.size


def register_baselines(registry) -> None:
    registry.register(OpenCVInpaint("telea"))
    registry.register(OpenCVInpaint("ns"))
    registry.register(NLMeansDenoise())
    registry.register(BicubicSR())
    registry.register(GrayscaleColorization())
