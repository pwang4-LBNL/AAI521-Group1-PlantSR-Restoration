"""Write randomly initialised HAT-L / Real-ESRGAN bundles in Peng's export format (tests only)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_fake_bundles(weights_dir: Path, methods=("hat_l", "real_esrgan")) -> dict:
    from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet
    from spandrel.architectures.HAT.__arch.HAT import HAT

    from plantsr_sr import models

    builders = {
        "hat_l": lambda: HAT(upscale=4, img_size=64, embed_dim=180, depths=[6] * 12, num_heads=[6] * 12,
                             window_size=16, compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
                             overlap_ratio=0.5, mlp_ratio=2.0, upsampler="pixelshuffle", resi_connection="1conv"),
        "real_esrgan": lambda: RRDBNet(in_nc=3, out_nc=3, num_filters=64, num_blocks=23, scale=4),
    }
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    torch.manual_seed(0)
    for method in methods:
        model = builders[method]()
        # Re-load through spandrel so the module types match what load_bundle produces.
        model = models._load_spandrel(method, model.state_dict())
        bundle = models.make_bundle(model, method, epoch=2 if method == "hat_l" else 10)
        path = weights_dir / f"{method}.pt"
        torch.save({k: bundle[k] for k in ("model_state", "model_metadata", "phase", "epoch", "format_version")}, path)
        paths[method] = path
    return paths


if __name__ == "__main__":
    print(write_fake_bundles(Path(sys.argv[1])))
