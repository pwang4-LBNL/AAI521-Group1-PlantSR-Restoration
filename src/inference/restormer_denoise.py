"""Restormer denoising for the shared interface (owner: Jinyuan).

Port of `run_denoiser` from notebooks/PlantSR_Denoise_Colorize_Finetune.ipynb:
float RGB in [0, 1], reflect-pad to a multiple of 8, mixed precision (bf16 where supported,
otherwise fp16), crop back and round to uint8. Images larger than one tile are processed in
overlapping tiles so GPU memory stays bounded.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .interface import TaskHandler, free_gpu_memory
from .model_utils import (autocast_ctx, hf_download, load_spandrel, pick_device, resolve, sha256_file, tiled,
                          to_tensor, to_uint8, verify_sha256)

PRETRAINED = {"repo_id": "deepinv/Restormer", "filename": "gaussian_color_denoising_blind.pth"}


class RestormerDenoise(TaskHandler):
    task = "denoising"
    uses_gpu = True
    generative = False

    def __init__(self, method: str = "restormer_finetuned", checkpoint=None, sha256: str | None = None,
                 pretrained: dict | None = None, tile: int = 512, halo: int = 32, label: str | None = None,
                 description: str | None = None, device: str | None = None):
        super().__init__()
        self.method = method
        self.checkpoint = resolve(checkpoint) if checkpoint else None
        self.expected_sha256 = sha256
        self.pretrained = pretrained  # {"repo_id", "filename"} when loading the original weights
        self.kind = "finetuned" if checkpoint else "pretrained"
        self.tile, self.halo = tile, halo
        self.device_name = device
        self.model = self.device = None
        self.label = label or ("Restormer · fine-tuned on PlantSR" if checkpoint else "Restormer · original pretrained")
        self.description = description or (
            "Blind Gaussian colour denoising (Restormer). Works at the uploaded resolution; "
            f"large images run in overlapping {tile}px tiles.")
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
        self.model = load_spandrel(path, "Restormer", self.device)
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        free_gpu_memory()

    def _denoise(self, rgb_u8: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        h, w = rgb_u8.shape[:2]
        x = to_tensor(rgb_u8, self.device)
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            mode = "reflect" if ph < h and pw < w else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        with torch.inference_mode(), autocast_ctx(self.device, allow_fp16=True):
            y = self.model(x)
        return to_uint8(y[..., :h, :w])

    def run(self, image: Image.Image, mask, seed: int):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        out, largest = tiled(rgb, self._denoise, core=self.tile, halo=self.halo)
        if largest != (rgb.shape[1], rgb.shape[0]):
            self.notes = [f"Processed in overlapping tiles of up to {largest[0]}×{largest[1]} px."]
        return Image.fromarray(out), largest


def checkpoint_digest(path) -> str:
    return sha256_file(resolve(path))
