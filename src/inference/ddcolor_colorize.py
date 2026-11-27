"""DDColor colorization for the shared interface (owner: Jinyuan).

Port of `to_gray_u8`, `ddcolor_forward` and `run_colorizer` from
notebooks/PlantSR_Denoise_Colorize_Finetune.ipynb:

1. Any upload is first reduced to neutral grey with the same CIE L* (a* = b* = 0), quantised to
   uint8 and averaged over channels, exactly like the notebook's fixed grayscale inputs. Inputs
   that are already neutral grey (R = G = B) are used as they are.
2. The grey image is resized to 512 × 512 (bilinear), DDColor predicts the Lab ab channels,
   and they are resized back to the input size.
3. The output combines the input's own L* with the predicted ab (Lab -> sRGB, clipped).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .interface import TaskHandler, free_gpu_memory
from .model_utils import (autocast_ctx, hf_download, load_spandrel, pick_device, resolve, to_tensor, to_uint8,
                          verify_sha256)

PRETRAINED = {"repo_id": "piddnad/ddcolor_paper_tiny", "filename": "pytorch_model.bin"}


def gray_from_rgb(x):
    """float Nx3xHxW sRGB -> neutral grey with the same CIE L* (still three channels)."""
    import kornia
    import torch

    lightness = kornia.color.rgb_to_lab(x)[:, :1]
    zeros = torch.zeros_like(lightness)
    return kornia.color.lab_to_rgb(torch.cat([lightness, zeros, zeros], 1), clip=True)


def to_gray_u8(rgb_u8: np.ndarray, device=None) -> np.ndarray:
    """HxWx3 uint8 -> HxWx3 uint8 neutral grey (the colorization benchmark input)."""
    import torch

    device = device or torch.device("cpu")
    with torch.no_grad():
        gray = to_uint8(gray_from_rgb(to_tensor(rgb_u8, device)))
    return np.repeat(gray.mean(axis=2, keepdims=True).round().astype(np.uint8), 3, axis=2)


def is_neutral_gray(rgb_u8: np.ndarray) -> bool:
    return bool(np.array_equal(rgb_u8[..., 0], rgb_u8[..., 1]) and np.array_equal(rgb_u8[..., 1], rgb_u8[..., 2]))


class DDColorColorize(TaskHandler):
    task = "colorization"
    uses_gpu = True
    generative = False

    def __init__(self, method: str = "ddcolor_finetuned", checkpoint=None, sha256: str | None = None,
                 pretrained: dict | None = None, input_size: int = 512, label: str | None = None,
                 description: str | None = None, device: str | None = None):
        super().__init__()
        self.method = method
        self.checkpoint = resolve(checkpoint) if checkpoint else None
        self.expected_sha256 = sha256
        self.pretrained = pretrained
        self.kind = "finetuned" if checkpoint else "pretrained"
        self.input_size = input_size
        self.device_name = device
        self.model = self.device = None
        self.label = label or ("DDColor-tiny · fine-tuned on PlantSR" if checkpoint
                               else "DDColor-tiny · original pretrained")
        self.description = description or (
            "Predicts colour for a grayscale photo (DDColor). Colour uploads are converted to grayscale first, "
            "so you can compare the result with the original colours.")
        self.version = (f"{PRETRAINED['repo_id']}+ft:{self.expected_sha256[:12]}" if checkpoint and self.expected_sha256
                        else f"{PRETRAINED['repo_id']}+ft" if checkpoint else PRETRAINED["repo_id"])

    def load(self) -> None:
        self.device = pick_device(self.device_name)
        if self.checkpoint is not None:
            digest = verify_sha256(self.checkpoint, self.expected_sha256)
            path, self.version = self.checkpoint, f"{PRETRAINED['repo_id']}+ft:{digest[:12]}"
        else:
            source = self.pretrained or PRETRAINED
            path, self.version = hf_download(source["repo_id"], source["filename"], source.get("revision"))
        self.model = load_spandrel(path, "DDColor", self.device)
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        free_gpu_memory()

    def run(self, image: Image.Image, mask, seed: int):
        import kornia
        import torch
        import torch.nn.functional as F

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if is_neutral_gray(rgb):
            gray_u8 = rgb
        else:
            gray_u8 = to_gray_u8(rgb, self.device)
            self.notes = ["The colour upload was converted to grayscale (CIE L*) before colorization."]
        with torch.inference_mode():
            gray = to_tensor(gray_u8, self.device)
            with torch.autocast(self.device.type, enabled=False):  # colour conversions in fp32
                lightness = kornia.color.rgb_to_lab(gray.float())[:, :1]
            x = F.interpolate(gray, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            with autocast_ctx(self.device, allow_fp16=False):  # DDColor is not fp16-safe
                ab = self.model(x)
            ab = F.interpolate(ab.float(), size=gray.shape[-2:], mode="bilinear", align_corners=False)
            rgb_out = kornia.color.lab_to_rgb(torch.cat([lightness, ab], 1), clip=True)
        return Image.fromarray(to_uint8(rgb_out)), (self.input_size, self.input_size)
