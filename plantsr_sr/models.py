"""PlantSR ×4 model loading and inference. Checkpoint reconstruction works offline."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from spandrel import ImageModelDescriptor, ModelLoader

SPECS = {
    "hat_l": {
        "repo_id": "jaideepsingh/upscale_models",
        "revision": "867e7c0ad519aba5d36b5bb4ef4a4a91781914c4",
        "filename": "HAT/HAT-L_SRx4_ImageNet-pretrain.pth",
        "sha256": "5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043",
        "architecture": "HAT",
        "scale": 4
    },
    "real_esrgan": {
        "repo_id": "kornia/real_esrgan",
        "revision": "e500e0f0d4f4b0ccb900d1b65442568ef382d6cf",
        "filename": "RealESRGAN_x4plus.safetensors",
        "sha256": "e82eee5fa7456f49025e70d906cd12aef0a90e5360d31d722488e9775156bce4",
        "architecture": "ESRGAN",
        "scale": 4
    }
}
_CHECKPOINT_TARGETS = {"hat_l": ("RHAG", 12), "real_esrgan": ("RRDB", 23)}


def _spec(name: str) -> dict:
    if name not in SPECS:
        raise ValueError(f"Unknown model {name!r}; choose one of {list(SPECS)}.")
    return SPECS[name]


def _verify_sha(path: str | Path, expected: str) -> None:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch for {path}: {actual} != {expected}")


def _unwrap_state(state: Mapping) -> dict:
    """Prefer EMA weights from standard upstream generator checkpoint bundles."""
    while isinstance(state, Mapping):
        nested = next(
            (state[key] for key in ("params_ema", "params", "state_dict", "model_state")
             if key in state and isinstance(state[key], Mapping)),
            None,
        )
        if nested is None:
            break
        state = nested
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The checkpoint does not contain a nonempty state dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("Model state must map string parameter names to tensors.")
    return dict(state)


def _load_spandrel(name: str, state: Mapping) -> nn.Module:
    descriptor = ModelLoader(device="cpu").load_from_state_dict(_unwrap_state(state))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(f"{name}: expected a single-image super-resolution model.")
    expected = _spec(name)
    if descriptor.architecture.id != expected["architecture"] or descriptor.scale != 4:
        raise ValueError(
            f"{name}: got {descriptor.architecture.id} x{descriptor.scale}, "
            f"expected {expected['architecture']} x4."
        )
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise ValueError(f"{name}: expected three RGB input and output channels.")
    # Bypass descriptor.__call__: it deliberately disables autograd for inference.
    return descriptor.model


def _validate_model(model: nn.Module, name: str) -> None:
    expected = {"hat_l": "HAT", "real_esrgan": "RRDBNet"}[name]
    if type(model).__name__ != expected:
        raise ValueError(f"Expected {expected}, got {type(model).__name__}.")
    if name == "hat_l" and (model.upscale != 4 or len(model.layers) != 12):
        raise ValueError("Expected HAT-L ×4 with 12 RHAG stages.")
    stage, required = _CHECKPOINT_TARGETS[name]
    if sum(type(block).__name__ == stage for block in model.modules()) != required:
        raise ValueError(f"Expected {required} {stage} blocks for {name}.")


def load_original(name: str) -> nn.Module:
    """Fetch a pinned original checkpoint and verify its bytes before loading."""
    from huggingface_hub import hf_hub_download

    spec = _spec(name)
    path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"], revision=spec["revision"])
    _verify_sha(path, spec["sha256"])
    if spec["filename"].endswith(".safetensors"):
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    model = _load_spandrel(name, state).float()
    _validate_model(model, name)
    return model


def make_bundle(model: nn.Module, name: str, epoch: int) -> dict:
    """Serialize the raw model with the same keys as the original PlantSR run."""
    _validate_model(model, name)
    return {
        "format_version": 1,
        "model_metadata": {"name": name, "spec": dict(_spec(name))},
        "model_state": model.state_dict(),
        "phase": "fine_tuned",
        "epoch": int(epoch),
    }


def load_bundle(path: str | Path) -> tuple[nn.Module, dict]:
    """Restore an existing PlantSR model on CPU without fetching weights."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    metadata = bundle["model_metadata"]
    name = metadata["name"]
    if metadata["spec"] != _spec(name):
        raise ValueError(f"Unexpected model specification in {path}")
    model = _load_spandrel(name, bundle["model_state"]).float()
    _validate_model(model, name)
    return model, bundle


def forward_sr(model: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
    """Run the differentiable model; keep predictions unclamped for L1 training."""
    _spec(name)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected an NCHW RGB tensor, received {tuple(x.shape)}.")
    sr = model(x)
    expected = (x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4)
    if tuple(sr.shape) != expected:
        raise ValueError(f"{name}: output shape {tuple(sr.shape)} differs from {expected}.")
    return sr


def enable_activation_checkpointing(model: nn.Module, name: str) -> int:
    """Checkpoint whole stages while preserving native state-dict parameter names."""
    _validate_model(model, name)
    class_name, expected_count = _CHECKPOINT_TARGETS[name]
    targets = [module for module in model.modules()
               if module.__class__.__name__ == class_name]
    if len(targets) != expected_count:
        raise ValueError(f"{name}: activation-checkpoint target count is incorrect.")

    def install(module):
        if getattr(module, "_plantsr_activation_checkpointing", False):
            return
        original_forward = module.forward

        def checkpointed_forward(*args, **kwargs):
            if module.training and torch.is_grad_enabled():
                return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
            return original_forward(*args, **kwargs)

        module.forward = checkpointed_forward
        module._plantsr_activation_checkpointing = True

    for module in targets:
        install(module)
    return len(targets)
