"""Inpainting metrics focused on the missing region and its boundary.

All functions take RGB uint8 arrays (H, W, 3) or float arrays in [0, 1] and a mask with
1 = missing/reconstructed, 0 = preserved. Full-image scores are supplementary only: when
visible pixels are copied back by compositing they inflate full-image PSNR/SSIM.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity

RESULT_FIELDS = ["sample_id", "task", "method", "checkpoint_revision", "seed", "metric_name",
                 "metric_value", "inference_seconds", "status", "error_note"]


def to_float(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def as_mask(mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > (127 if arr.dtype == np.uint8 and arr.max() > 1 else 0.5)


def composite(known, generated, mask) -> np.ndarray:
    """final = (1 - mask) * known + mask * generated, returned as uint8 RGB."""
    m = as_mask(mask)[..., None]
    out = np.where(m, np.asarray(generated, np.uint8), np.asarray(known, np.uint8))
    return out.astype(np.uint8)


def _psnr(mse: float) -> float:
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def boundary_band(mask, width: int = 3) -> np.ndarray:
    """Pixels within `width` px on either side of the mask edge (where seams appear)."""
    m = as_mask(mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(m, kernel) > 0) & ~(cv2.erode(m, kernel) > 0)


def _gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((rgb * 255).astype(np.float32), cv2.COLOR_RGB2GRAY) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _bbox(mask: np.ndarray, pad: int, min_size: int, shape) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = shape
    def span(lo, hi, limit):
        lo, hi = max(0, lo - pad), min(limit, hi + 1 + pad)
        if hi - lo < min_size:  # grow small boxes symmetrically, staying inside the image
            lo = max(0, lo - (min_size - (hi - lo)) // 2)
            hi = min(limit, lo + min_size)
            lo = max(0, hi - min_size)
        return int(lo), int(hi)

    x0, x1 = span(xs.min(), xs.max(), w)
    y0, y1 = span(ys.min(), ys.max(), h)
    return x0, y0, x1, y1


class LPIPSMetric:
    """LPIPS on the padded bounding box of the missing region (optional dependency)."""

    def __init__(self, net: str = "alex", device: str | None = None, pad: int = 8, min_size: int = 64):
        import lpips  # noqa: WPS433 - optional dependency
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
        self.pad, self.min_size = pad, min_size

    def __call__(self, output, reference, mask) -> float:
        out, ref, m = to_float(output), to_float(reference), as_mask(mask)
        x0, y0, x1, y1 = _bbox(m, self.pad, self.min_size, m.shape)
        tensors = [
            self.torch.from_numpy(a[y0:y1, x0:x1]).permute(2, 0, 1)[None].to(self.device) * 2 - 1
            for a in (out, ref)
        ]
        with self.torch.no_grad():
            return float(self.model(*tensors).item())


def inpaint_metrics(output, reference, mask, lpips_fn=None, band_width: int = 3) -> dict:
    """Missing-region, boundary and (supplementary) full-image metrics."""
    out, ref, m = to_float(output), to_float(reference), as_mask(mask)
    if out.shape != ref.shape or out.shape[:2] != m.shape:
        raise ValueError(f"Shape mismatch: output {out.shape}, reference {ref.shape}, mask {m.shape}")
    if not m.any():
        raise ValueError("Mask is empty; nothing to evaluate.")
    diff = out - ref
    hole = diff[m]
    band = boundary_band(m, band_width)
    ssim_full, ssim_map = structural_similarity(ref, out, channel_axis=2, data_range=1.0, full=True)
    ssim_map = ssim_map.mean(axis=2)
    grad_out, grad_ref = _gradient_magnitude(out), _gradient_magnitude(ref)
    metrics = {
        "mask_coverage": float(m.mean()),
        "masked_l1": float(np.abs(hole).mean()),
        "masked_psnr": _psnr(float((hole ** 2).mean())),
        "masked_ssim": float(ssim_map[m].mean()),
        "boundary_l1": float(np.abs(diff[band]).mean()),
        "boundary_grad_l1": float(np.abs(grad_out[band] - grad_ref[band]).mean()),
        "full_psnr_supplementary": _psnr(float((diff ** 2).mean())),
        "full_ssim_supplementary": float(ssim_full),
    }
    if lpips_fn is not None:
        metrics["masked_bbox_lpips"] = lpips_fn(out, ref, m)
    return metrics


def full_metrics(output, reference) -> dict:
    """PSNR/SSIM for tasks without a mask (denoising, SR, colorization)."""
    out, ref = to_float(output), to_float(reference)
    if out.shape != ref.shape:
        raise ValueError(f"Shape mismatch: output {out.shape}, reference {ref.shape}")
    return {
        "psnr": _psnr(float(((out - ref) ** 2).mean())),
        "ssim": float(structural_similarity(ref, out, channel_axis=2, data_range=1.0)),
    }


def result_rows(sample_id, method, metrics: dict | None, *, task="inpainting", checkpoint_revision="",
                seed="", inference_seconds=None, status="ok", error_note=""):
    """Long-format rows matching the team's results fields."""
    base = {"sample_id": sample_id, "task": task, "method": method,
            "checkpoint_revision": checkpoint_revision, "seed": seed,
            "inference_seconds": "" if inference_seconds is None else round(float(inference_seconds), 4),
            "status": status, "error_note": error_note}
    if not metrics:
        return [{**base, "metric_name": "", "metric_value": ""}]
    return [{**base, "metric_name": k, "metric_value": v} for k, v in metrics.items()]
