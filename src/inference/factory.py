"""Build the application registry from a small JSON config.

Teammates add their handlers here (one `register` call per method) once their wrappers
follow the TaskHandler contract in docs/inference_interface.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from .baselines import register_baselines
from .interface import Registry

DEFAULT_CONFIG = {
    "max_gpu_models": 1,
    "inpainting": {
        "enable_pretrained": True,
        "model_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "work_res": 512,
        "prompt": "a high-resolution photo of a plant, natural leaves, sharp focus",
        "negative_prompt": "blurry, text, watermark, frame, border, cartoon, painting",
        "steps": 25,
        "guidance": 5.0,
        "lora_path": "",
        "lora_label": "",
        "default_method": "",
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


def build_registry(config: dict | None = None, with_diffusion: bool = True) -> Registry:
    config = config or load_config()
    registry = Registry(max_gpu_models=int(config.get("max_gpu_models", 1)))
    register_baselines(registry)
    inp = config.get("inpainting", {})
    if with_diffusion:
        from .sd_inpaint import StableDiffusionInpaint

        common = {k: inp[k] for k in ("model_id", "work_res", "prompt", "negative_prompt", "steps", "guidance") if k in inp}
        if inp.get("enable_pretrained", True):
            registry.register(StableDiffusionInpaint("sd15_pretrained", **common), default=True)
        lora = inp.get("lora_path")
        if lora and Path(lora).exists():
            registry.register(StableDiffusionInpaint("sd15_finetuned", lora_path=lora, label=inp.get("lora_label") or None,
                                                     **common), default=True)
    if inp.get("default_method"):
        registry.set_default("inpainting", inp["default_method"])
    # --- Teammates: register Real-ESRGAN / denoising / colorization handlers here, e.g.
    # registry.register(RealESRGANHandler(checkpoint=...), default=True)
    return registry
