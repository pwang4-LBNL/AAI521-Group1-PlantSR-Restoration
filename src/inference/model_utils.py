"""Shared helpers for loading the team's PyTorch checkpoints and running them on large images.

Heavy libraries (torch, spandrel, huggingface_hub) are imported inside the functions so the
app and the CPU baselines start without them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def resolve(path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path, expected: str | None) -> str:
    """Return the file's SHA-256; raise if an expected value is given and differs."""
    actual = sha256_file(path)
    if expected and actual != expected.lower():
        raise ValueError(f"Checkpoint {path} has SHA-256 {actual[:12]}…, expected {expected[:12]}…. "
                         "Re-copy the file from its owner's handoff.")
    return actual


def pick_device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bf16_ok(device) -> bool:
    import torch

    return device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)


def autocast_ctx(device, allow_fp16: bool):
    """Mixed precision exactly as in the training notebooks: bf16 where supported, else fp16 if allowed."""
    import torch

    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    if bf16_ok(device):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16, enabled=allow_fp16)


def hf_download(repo_id: str, filename: str, revision: str | None = None) -> tuple[Path, str]:
    """Download a Hugging Face file; return (path, 'repo@commit12')."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    commit = path.parent.name if len(path.parent.name) == 40 else (revision or "main")
    return path, f"{repo_id}@{commit[:12]}"


_EXTRA_ARCHES = False


def _install_extra_arches() -> None:
    """Register Restormer and DDColor (spandrel_extra_arches) once, as the training notebook does."""
    global _EXTRA_ARCHES
    if not _EXTRA_ARCHES:
        import spandrel_extra_arches

        spandrel_extra_arches.install()
        _EXTRA_ARCHES = True


def load_spandrel(path, expected_arch: str, device):
    """Load a checkpoint into its matching architecture with spandrel (as the training notebooks do)."""
    import torch
    from spandrel import ModelLoader, UnsupportedModelError

    _install_extra_arches()
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    try:
        descriptor = ModelLoader().load_from_state_dict(state)
    except UnsupportedModelError as exc:
        raise ValueError(f"{path} is not a model spandrel recognises (expected {expected_arch}).") from exc
    if descriptor.architecture.name != expected_arch:
        raise ValueError(f"{path} contains a {descriptor.architecture.name} model, expected {expected_arch}.")
    return descriptor.model.to(device).eval().requires_grad_(False)


def to_tensor(rgb_u8: np.ndarray, device):
    """HxWx3 uint8 -> 1x3xHxW float in [0, 1]."""
    import torch

    arr = np.ascontiguousarray(rgb_u8)
    return torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1)[None].float().div(255).to(device)


def to_uint8(tensor) -> np.ndarray:
    """1x3xHxW float -> HxWx3 uint8, rounded like the notebooks' saved PNGs."""
    return tensor.detach().float().clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()


def tiled(rgb_u8: np.ndarray, run_tile, core: int, halo: int, scale: int = 1):
    """Run `run_tile` on overlapping tiles and stitch the cores.

    Each tile carries `halo` pixels of context on every side; only its centre is kept, so tile
    borders do not show. Returns (output array, largest tile (w, h) processed).
    """
    h, w = rgb_u8.shape[:2]
    if max(h, w) <= core + 2 * halo:
        out = run_tile(rgb_u8)
        return out, (w, h)
    out = np.zeros((h * scale, w * scale, 3), np.uint8)
    largest = (0, 0)
    for top in range(0, h, core):
        for left in range(0, w, core):
            bottom, right = min(top + core, h), min(left + core, w)
            y0, x0 = max(0, top - halo), max(0, left - halo)
            y1, x1 = min(h, bottom + halo), min(w, right + halo)
            tile_out = run_tile(rgb_u8[y0:y1, x0:x1])
            largest = (max(largest[0], x1 - x0), max(largest[1], y1 - y0))
            out[top * scale:bottom * scale, left * scale:right * scale] = tile_out[
                (top - y0) * scale:(bottom - y0) * scale, (left - x0) * scale:(right - x0) * scale]
    return out, largest


def fit_max_side(image: Image.Image, max_side: int | None):
    """Downscale so the longer side is at most `max_side`; return (image, note or None)."""
    if not max_side or max(image.size) <= max_side:
        return image, None
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    note = (f"Input reduced from {image.width}×{image.height} to {size[0]}×{size[1]} "
            f"(limit {max_side} px) to keep processing time reasonable.")
    return image.resize(size, Image.LANCZOS), note
