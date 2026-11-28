"""Build the application registry from configs/app.json.

Every task gets its simple baseline plus the team's models:

| Task | Fine-tuned (default) | Original pretrained (comparison) | Owner |
|---|---|---|---|
| denoising | Restormer, `checkpoints/denoise/best.pth` | deepinv/Restormer | Jinyuan |
| colorization | DDColor-tiny, `checkpoints/colorize/best.pth` | piddnad/ddcolor_paper_tiny | Jinyuan |
| super_resolution | HAT-L (+ Real-ESRGAN), `plantsr_sr/weights/` | HAT-L from Hugging Face | Peng |
| inpainting | SD 1.5 inpainting + PlantSR LoRA, `checkpoints/inpaint/lora/` | SD 1.5 inpainting | Ali |

A method whose checkpoint is missing is skipped with a reason (shown by /api/health and the
UI), so the app always starts; the task then falls back to its pretrained model or baseline.
Paths are relative to the repository root.
"""
from __future__ import annotations

import json
from pathlib import Path

from .baselines import register_baselines
from .interface import Registry
from .model_utils import ROOT, resolve

DEFAULT_CONFIG = {
    "max_gpu_models": 4,
    "include_pretrained": True,
    "device": None,
    "denoising": {
        "checkpoint": "checkpoints/denoise/best.pth",
        "sha256": "f01c68c05c056e73e0926b4f0c8ae6f497e0441970ab2552b112accd4562b040",
        "tile": 512,
        "halo": 32,
    },
    "colorization": {
        "checkpoint": "checkpoints/colorize/best.pth",
        "sha256": "d52fcfdb9752775e1550c3820229ec189b9ee0c7481a13f3b0f6f11e08d8af0e",
        "input_size": 512,
    },
    "super_resolution": {
        "package_dir": "plantsr_sr",
        "methods": ["hat_l", "real_esrgan"],
        "default": "hat_l",
        "pretrained_methods": ["hat_l"],
        "max_input_side": 512,
    },
    "inpainting": {
        "enable_pretrained": True,
        "model_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "work_res": 512,
        "prompt": "a high-resolution photo of a plant, natural leaves, sharp focus",
        "negative_prompt": "blurry, text, watermark, frame, border, cartoon, painting",
        "steps": 25,
        "guidance": 5.0,
        "lora_path": "checkpoints/inpaint/lora",
        "lora_label": "",
        "default_method": "sd15_finetuned",
    },
}


def load_config(path=None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and Path(path).is_file():
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def _missing(path) -> str:
    try:
        shown = resolve(path).relative_to(ROOT)
    except ValueError:
        shown = resolve(path)
    return f"checkpoint not installed: {shown} (see docs/deployment.md)"


def build_registry(config: dict | None = None, with_diffusion: bool = True, with_models: bool = True) -> Registry:
    """Create the registry. `with_models=False` gives baselines only (fast, CPU, for UI work and tests)."""
    config = config or load_config()
    registry = Registry(max_gpu_models=int(config.get("max_gpu_models", 4)))
    register_baselines(registry)
    if not with_models:
        return registry
    device = config.get("device")
    pretrained = bool(config.get("include_pretrained", True))

    # ---------------------------------------------------------------- denoising (Jinyuan)
    from .restormer_denoise import RestormerDenoise

    dn = config.get("denoising", {})
    dn_ok = bool(dn.get("checkpoint")) and resolve(dn["checkpoint"]).is_file()
    if dn_ok:
        registry.register(RestormerDenoise("restormer_finetuned", checkpoint=dn["checkpoint"], sha256=dn.get("sha256"),
                                           tile=dn.get("tile", 512), halo=dn.get("halo", 32), device=device),
                          default=True)
    else:
        registry.mark_unavailable("denoising", "restormer_finetuned", _missing(dn.get("checkpoint", "?")))
    if pretrained:
        registry.register(RestormerDenoise("restormer_pretrained", tile=dn.get("tile", 512), halo=dn.get("halo", 32),
                                           device=device),
                          default=not dn_ok)

    # ---------------------------------------------------------------- colorization (Jinyuan)
    from .ddcolor_colorize import DDColorColorize

    cz = config.get("colorization", {})
    cz_ok = bool(cz.get("checkpoint")) and resolve(cz["checkpoint"]).is_file()
    if cz_ok:
        registry.register(DDColorColorize("ddcolor_finetuned", checkpoint=cz["checkpoint"], sha256=cz.get("sha256"),
                                          input_size=cz.get("input_size", 512), device=device), default=True)
    else:
        registry.mark_unavailable("colorization", "ddcolor_finetuned", _missing(cz.get("checkpoint", "?")))
    if pretrained:
        registry.register(DDColorColorize("ddcolor_pretrained", input_size=cz.get("input_size", 512), device=device),
                          default=not cz_ok)

    # ---------------------------------------------------------------- super-resolution (Peng)
    from .sr_plantsr import PlantSRx4, weights_path

    sr = config.get("super_resolution", {})
    package_dir = sr.get("package_dir", "plantsr_sr")
    sr_ok = False
    for method in sr.get("methods", ["hat_l", "real_esrgan"]):
        if weights_path(method, package_dir).is_file():
            sr_ok = True
            registry.register(PlantSRx4(method, package_dir=package_dir, max_input_side=sr.get("max_input_side", 512),
                                        device=device), default=method == sr.get("default", "hat_l"))
        else:
            registry.mark_unavailable("super_resolution", method, _missing(weights_path(method, package_dir)))
    if pretrained:
        for method in sr.get("pretrained_methods", ["hat_l"]):
            registry.register(PlantSRx4(method, package_dir=package_dir, pretrained=True,
                                        max_input_side=sr.get("max_input_side", 512), device=device),
                              default=not sr_ok)

    # ---------------------------------------------------------------- inpainting (Ali)
    inp = config.get("inpainting", {})
    if with_diffusion:
        from .sd_inpaint import StableDiffusionInpaint, lora_file

        common = {k: inp[k] for k in ("model_id", "work_res", "prompt", "negative_prompt", "steps", "guidance") if k in inp}
        if inp.get("enable_pretrained", True):
            registry.register(StableDiffusionInpaint("sd15_pretrained", **common), default=True)
        lora = inp.get("lora_path")
        try:
            lora_ok = bool(lora) and lora_file(resolve(lora)).is_file()
        except FileNotFoundError:
            lora_ok = False
        if lora_ok:
            registry.register(StableDiffusionInpaint("sd15_finetuned", lora_path=resolve(lora),
                                                     label=inp.get("lora_label") or None, **common))
        elif lora:
            registry.mark_unavailable("inpainting", "sd15_finetuned", _missing(lora))
        default = inp.get("default_method")
        if default:
            try:
                registry.set_default("inpainting", default)
            except ValueError:
                pass  # e.g. the fine-tuned LoRA is not installed yet; keep the registered default
    return registry
