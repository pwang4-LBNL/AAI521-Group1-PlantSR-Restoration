"""Validation/test crops with each task owner's fixed degradation, for benchmark mode.

The app shows PSNR/SSIM only for these examples, where the clean reference is known. The inputs
are generated with the same code the owners used for their reported numbers:

| Task | Input | Source |
|---|---|---|
| denoising | Gaussian noise sigma 15, `clip(round(clean + N(0, sigma^2)))`, seed `sha256("noise|<sample_id>|15|42")` | Jinyuan, `add_noise_fixed` |
| super_resolution | 256 -> 64 Pillow bicubic, `reducing_gap=None` | Peng, `get_pairs` |
| colorization | neutral grey with the reference CIE L* (kornia Lab, a* = b* = 0) | Jinyuan, `to_gray_u8` |
| inpainting | fixed mask v1 from the inpainting manifest (or regenerated identically) | Ali, `fixed_eval_masks` |
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.inpaint_masks import apply_mask, fixed_eval_masks, load_mask_png
from src.evaluation.inpaint_metrics import full_metrics, inpaint_metrics

DENOISE_SIGMA = 15
SEED = 42


def jinyuan_seed(*parts) -> int:
    """`stable_seed` from the denoising/colorization notebook."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % 2**32


def add_noise_fixed(clean_u8: np.ndarray, sigma, key: str, seed: int = SEED) -> np.ndarray:
    """Jinyuan's fixed benchmark noise (identical inputs in every session)."""
    rng = np.random.default_rng(jinyuan_seed("noise", key, sigma, seed))
    noisy = clean_u8.astype(np.float32) + rng.normal(0.0, sigma, clean_u8.shape).astype(np.float32)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def bicubic_lr(hr: Image.Image, factor: int = 4) -> Image.Image:
    """Peng's fixed 4x degradation (Pillow bicubic, reducing_gap=None)."""
    return hr.resize((hr.width // factor, hr.height // factor), Image.Resampling.BICUBIC, reducing_gap=None)


def gray_lab_l(rgb_u8: np.ndarray) -> np.ndarray:
    """Jinyuan's grayscale input (needs torch + kornia, which the colorization model uses anyway)."""
    from src.inference.ddcolor_colorize import to_gray_u8

    return to_gray_u8(rgb_u8)


def delta_e00(output, reference) -> float:
    from skimage.color import deltaE_ciede2000, rgb2lab

    return float(deltaE_ciede2000(rgb2lab(np.asarray(reference)), rgb2lab(np.asarray(output))).mean())


class BenchmarkExamples:
    def __init__(self, prepared_dir, split: str = "val", inpaint_manifest=None, limit: int | None = 24,
                 seed: int = SEED):
        self.prepared_dir = Path(prepared_dir)
        self.split, self.seed = split, seed
        with open(self.prepared_dir / "manifests" / f"{split}.csv", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("eval_crop_path")]
        ordered = sorted(rows, key=lambda r: r["sample_id"])
        self.rows = {r["sample_id"]: r for r in (ordered[:limit] if limit else ordered)}
        self.masks, self.mask_info = {}, {}
        manifest_rows = []
        if inpaint_manifest and Path(inpaint_manifest).is_file():
            with open(inpaint_manifest, encoding="utf-8-sig", newline="") as f:
                manifest_rows = [r for r in csv.DictReader(f) if r["sample_id"] in self.rows]
        for r in manifest_rows:
            self.masks[r["sample_id"]] = load_mask_png(self.prepared_dir / r["mask_path"])
            self.mask_info[r["sample_id"]] = {"mask_type": r["mask_type"], "mask_coverage": float(r["mask_coverage"])}
        missing = [sid for sid in self.rows if sid not in self.masks]
        if missing:  # same deterministic generator as the saved manifests (it depends on all IDs of the split)
            generated = fixed_eval_masks([r["sample_id"] for r in rows], split, seed)
            for sid in missing:
                mask, params, _ = generated[sid]
                self.masks[sid] = mask
                self.mask_info[sid] = {"mask_type": params["mask_type"], "mask_coverage": params["coverage"]}

    def __len__(self):
        return len(self.rows)

    def ids(self) -> list[str]:
        return list(self.rows)

    def list(self) -> list[dict]:
        return [{"sample_id": sid, "source_image_id": f"{r['official_split']}/{r['relative_path']}",
                 **self.mask_info[sid]} for sid, r in self.rows.items()]

    def reference(self, sample_id: str) -> np.ndarray:
        with Image.open(self.prepared_dir / self.rows[sample_id]["eval_crop_path"]) as src:
            return np.asarray(src.convert("RGB"))

    def degraded(self, sample_id: str, task: str):
        """Return (input PIL image, mask or None) for one task."""
        if sample_id not in self.rows:
            raise KeyError(f"Unknown example {sample_id!r}")
        ref = self.reference(sample_id)
        if task == "denoising":
            return Image.fromarray(add_noise_fixed(ref, DENOISE_SIGMA, sample_id, self.seed)), None
        if task == "super_resolution":
            return bicubic_lr(Image.fromarray(ref)), None
        if task == "colorization":
            return Image.fromarray(gray_lab_l(ref)), None
        if task == "inpainting":
            mask = self.masks[sample_id]
            return Image.fromarray(apply_mask(ref, mask)), mask
        raise ValueError(f"Unknown task {task!r}")

    def metrics(self, sample_id: str, task: str, output, mask=None) -> dict:
        ref = self.reference(sample_id)
        out = np.asarray(output)
        if out.shape != ref.shape:
            return {"note": f"output {out.shape[1]}x{out.shape[0]} differs from reference {ref.shape[1]}x{ref.shape[0]}"}
        if task == "inpainting":
            m = inpaint_metrics(out, ref, mask if mask is not None else self.masks[sample_id])
            return {"masked_psnr": m["masked_psnr"], "masked_ssim": m["masked_ssim"], "boundary_l1": m["boundary_l1"]}
        scores = full_metrics(out, ref)
        if task == "colorization":
            scores["delta_e00"] = delta_e00(out, ref)
        return scores
