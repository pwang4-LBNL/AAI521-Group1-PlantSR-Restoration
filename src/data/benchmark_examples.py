"""Validation crops with fixed degradations for the application's benchmark mode.

The app shows PSNR/SSIM only for these examples, where the clean reference is known.
Degradations follow the README recipes (section 5):
  denoising        Gaussian noise, sigma 15 (0-255 units), seed fixed per sample
  super_resolution 256 -> 64 bicubic downsampling (x4)
  colorization     grayscale replicated to three channels
  inpainting       fixed mask from the inpainting manifest (or regenerated deterministically)
Replace these preview recipes with the shared generators once each task owner publishes them.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.inpaint_masks import apply_mask, fixed_eval_masks, load_mask_png, seed_for
from src.evaluation.inpaint_metrics import full_metrics, inpaint_metrics


class BenchmarkExamples:
    def __init__(self, prepared_dir, split: str = "val", inpaint_manifest=None, limit: int = 24, seed: int = 42):
        self.prepared_dir = Path(prepared_dir)
        self.split, self.seed = split, seed
        with open(self.prepared_dir / "manifests" / f"{split}.csv", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("eval_crop_path")]
        self.rows = {r["sample_id"]: r for r in sorted(rows, key=lambda r: r["sample_id"])[:limit]}
        self.masks, self.mask_info = {}, {}
        manifest_rows = []
        if inpaint_manifest and Path(inpaint_manifest).is_file():
            with open(inpaint_manifest, encoding="utf-8-sig", newline="") as f:
                manifest_rows = [r for r in csv.DictReader(f) if r["sample_id"] in self.rows]
        for r in manifest_rows:
            self.masks[r["sample_id"]] = load_mask_png(self.prepared_dir / r["mask_path"])
            self.mask_info[r["sample_id"]] = {"mask_type": r["mask_type"], "mask_coverage": float(r["mask_coverage"])}
        missing = [sid for sid in self.rows if sid not in self.masks]
        if missing:  # same deterministic generator as the saved manifests
            all_ids = [r["sample_id"] for r in rows]
            generated = fixed_eval_masks(all_ids, split, seed)
            for sid in missing:
                mask, params, _ = generated[sid]
                self.masks[sid] = mask
                self.mask_info[sid] = {"mask_type": params["mask_type"], "mask_coverage": params["coverage"]}

    def __len__(self):
        return len(self.rows)

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
            rng = np.random.default_rng(seed_for(self.seed, "denoise_sigma15", sample_id))
            noisy = ref.astype(np.float32) + rng.normal(0.0, 15.0, ref.shape)
            return Image.fromarray(np.clip(np.round(noisy), 0, 255).astype(np.uint8)), None
        if task == "super_resolution":
            h, w = ref.shape[:2]
            return Image.fromarray(ref).resize((w // 4, h // 4), Image.BICUBIC), None
        if task == "colorization":
            return Image.fromarray(ref).convert("L").convert("RGB"), None
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
        return full_metrics(out, ref)
