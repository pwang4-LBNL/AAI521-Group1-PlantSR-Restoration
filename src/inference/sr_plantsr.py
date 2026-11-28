"""4x super-resolution for the shared interface, using Peng's `plantsr_sr` package unchanged.

`plantsr_sr/` (repository root) is exported by notebooks/PlantSR_x4_Super_Resolution_Colab_combined_final.ipynb.
Its fine-tuned weights live in `plantsr_sr/weights/{hat_l,real_esrgan}.pt` (not in Git; see
docs/deployment.md). This adapter adds two things the app needs:

* an input-size limit: the output is 4x larger in each direction, so uploads are reduced to
  `max_input_side` first (the note is shown to the user);
* the original pretrained HAT-L / Real-ESRGAN weights as optional comparison methods, loaded
  with Peng's checksum-verified `load_original`.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .interface import TaskHandler, free_gpu_memory
from .model_utils import ROOT, fit_max_side, pick_device, resolve, sha256_file

DEFAULT_PACKAGE_DIR = ROOT / "plantsr_sr"
NAMES = {"hat_l": "HAT-L", "real_esrgan": "Real-ESRGAN"}


def weights_path(method: str, package_dir=None) -> Path:
    return resolve(package_dir or DEFAULT_PACKAGE_DIR) / "weights" / f"{method}.pt"


def manifest_sha256(method: str, package_dir=None) -> str | None:
    manifest = resolve(package_dir or DEFAULT_PACKAGE_DIR) / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8")).get("models", {}).get(method, {}).get("sha256")
    return None


class PlantSRx4(TaskHandler):
    task = "super_resolution"
    uses_gpu = True
    generative = False

    def __init__(self, method: str = "hat_l", package_dir=None, pretrained: bool = False,
                 max_input_side: int = 512, label: str | None = None, device: str | None = None):
        super().__init__()
        if method not in NAMES:
            raise ValueError(f"Unknown SR model {method!r}; choose one of {list(NAMES)}.")
        self.model_name = method
        self.method = f"{method}_pretrained" if pretrained else method
        self.package_dir = package_dir
        self.pretrained = pretrained
        self.kind = "pretrained" if pretrained else "finetuned"
        self.max_input_side = max_input_side
        self.device_name = device
        self.predictor = self.device = self.model = None
        suffix = "original pretrained" if pretrained else "fine-tuned on PlantSR"
        self.label = label or f"{NAMES[method]} ×4 · {suffix}"
        self.description = (f"{NAMES[method]} 4× upscaling of the whole image in overlapping tiles. "
                            f"Inputs larger than {max_input_side} px are reduced first.")
        self.version = f"{method}-original" if pretrained else f"{method}-fine-tuned"

    def load(self) -> None:
        self.device = pick_device(self.device_name)
        if self.pretrained:
            from plantsr_sr import models

            self.model = models.load_original(self.model_name).eval().requires_grad_(False).to(self.device)
            spec = models.SPECS[self.model_name]
            self.version = f"{spec['repo_id']}@{spec['revision'][:12]}"
        else:
            from plantsr_sr.inference import Predictor

            path = weights_path(self.model_name, self.package_dir)
            expected = manifest_sha256(self.model_name, self.package_dir)
            digest = sha256_file(path)
            if expected and digest != expected:
                raise ValueError(f"{path} does not match plantsr_sr/manifest.json; re-copy Peng's bundle.")
            self.predictor = Predictor(self.model_name, package_dir=resolve(self.package_dir or DEFAULT_PACKAGE_DIR),
                                       device=str(self.device))
            self.version = f"{self.model_name}-epoch-{self.predictor.epoch}+sha:{digest[:12]}"
        self._loaded = True

    def unload(self) -> None:
        self.predictor = self.model = None
        self._loaded = False
        free_gpu_memory()

    def run(self, image: Image.Image, mask, seed: int):
        import torch

        image, note = fit_max_side(image.convert("RGB"), self.max_input_side)
        if self.predictor is not None:
            output, tile = self.predictor.with_tile_size(image)
        else:
            from plantsr_sr.inference import _predict, _restore

            with torch.inference_mode():
                output, tile = _restore(image, lambda t: _predict(t, self.model, self.model_name, self.device))
        notes = [note] if note else []
        if max(image.size) > 128:  # Peng's tiler uses 128 px tiles with 16 px of context
            notes.append(f"Upscaled in overlapping tiles (largest {tile[0]}×{tile[1]} px).")
        self.notes = notes
        return output, image.size


def available_methods(package_dir=None) -> dict[str, bool]:
    return {m: weights_path(m, package_dir).is_file() for m in NAMES}
