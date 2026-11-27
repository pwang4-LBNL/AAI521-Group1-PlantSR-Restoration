# Common inference interface

Shared by all four tasks (agreed in Week 1). The app, the evaluation notebook and the tests all call it. Which model each method runs is listed in `docs/model_provenance.md`.

```python
from src.inference.interface import restore

result = restore(image, task, mask=None, seed=42)                       # agreed signature
result = restore(image, task, mask=None, seed=42, method="sd15_finetuned")  # optional: pick a method
```

| Argument | Meaning |
| --- | --- |
| `image` | PIL image, RGB `uint8` array or file path. Converted to RGB; transparency is flattened onto white. |
| `task` | `"denoising"`, `"super_resolution"`, `"colorization"` or `"inpainting"`. |
| `mask` | Inpainting only. Same size as `image`; **1/white = missing (reconstruct), 0/black = preserve**. PIL, array or path, encoded as 0/1 or 0/255. |
| `seed` | Integer used by generative models; deterministic methods ignore it. |
| `method` | Optional. Defaults to the task's default method (baseline, pretrained or fine-tuned). |

`restore` never raises for bad input or model errors. It returns a `RestoreResult`:

| Field | Meaning |
| --- | --- |
| `image` | Output PIL RGB image (`None` on failure) |
| `task`, `method`, `model_version` | What ran, including the checkpoint revision, e.g. `stable-diffusion-v1-5/stable-diffusion-inpainting@<commit>+lora:<sha12>` |
| `input_size`, `processed_size`, `output_size` | `(width, height)` received, actually processed by the model, and returned |
| `elapsed_seconds` | Wall time for the call, including model loading when it happens |
| `success`, `error` | `False` with a readable message on failure (validation errors, GPU out-of-memory, etc.) |
| `extra` | Optional details: inpainting adds `mask_coverage`; `notes` explains resizing, tiling, grayscale conversion or a retry after GPU memory ran out |

`result.metadata()` gives the same information as a JSON-safe dict for the app and the results CSV.

## Methods in the app

| Task | Fine-tuned (default when installed) | Original pretrained | Baselines |
| --- | --- | --- | --- |
| `denoising` | `restormer_finetuned` | `restormer_pretrained` | `opencv_nlmeans` |
| `super_resolution` | `hat_l`, `real_esrgan` | `hat_l_pretrained` | `bicubic_x4` |
| `colorization` | `ddcolor_finetuned` | `ddcolor_pretrained` | `grayscale_rgb` |
| `inpainting` | `sd15_finetuned` (after the full LoRA fine-tune) | `sd15_pretrained` (current default) | `opencv_telea`, `opencv_ns` |

Wrappers: `src/inference/restormer_denoise.py`, `ddcolor_colorize.py` (ported from Jinyuan's notebook),
`sr_plantsr.py` (calls Peng's exported `plantsr_sr` package unchanged) and `sd_inpaint.py`. Each handler
reports `kind` (`finetuned`, `pretrained` or `baseline`), which the app uses to group methods.

## Adding a model

Write one small subclass of `TaskHandler` per method and register it in `src/inference/factory.py`:

```python
from PIL import Image
from src.inference.interface import TaskHandler

class RealESRGANx4(TaskHandler):
    task = "super_resolution"
    method = "realesrgan_finetuned"
    label = "Real-ESRGAN x4 (PlantSR fine-tuned)"
    description = "Real-ESRGAN generator fine-tuned with L1 on PlantSR 64->256 pairs."
    uses_gpu = True          # lets the registry unload it when another big model is needed
    generative = False       # True if the seed changes the output

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.version = f"kornia/real_esrgan+ft:{checkpoint}"
        self.model = None
        super().__init__()

    def load(self):          # called lazily on the first request
        self.model = ...     # build the network, load weights, .to("cuda").eval()
        self._loaded = True

    def unload(self):
        self.model = None
        self._loaded = False

    def run(self, image: Image.Image, mask, seed: int):
        # image: RGB PIL at the uploaded size. Resize/tile here if needed and report it.
        output = ...         # RGB PIL
        processed_size = image.size
        return output, processed_size
```

```python
# src/inference/factory.py
registry.register(RealESRGANx4(checkpoint="checkpoints/sr/realesrgan_plantsr_v1.pth"), default=True)
```

Conventions:

- **Images in and out are RGB PIL images, values 0–255.** Do your own normalisation inside `run`.
- **Do not load weights at import time or in `__init__`.** Load in `load()`; the app keeps at most one large GPU model loaded (`max_gpu_models`) and calls `unload()` on the least recently used one.
- **Report `processed_size` honestly.** If you downscale a large upload, tile, or crop, return the size the model actually saw; the app displays it.
- **Inpainting outputs are composited**: `final = (1 - mask) * input + mask * generated`, so visible pixels are copied exactly.
- **Determinism**: use `seed` for any sampling (`torch.Generator("cpu").manual_seed(seed)`), so benchmark runs are reproducible.
- Baselines (`opencv_telea`, `opencv_nlmeans`, `bicubic_x4`, `grayscale_rgb`) are already registered, so the app works end to end before every model is ready.

## Quick test for a new handler

```python
from src.inference.factory import build_registry
from src.inference.interface import set_default_registry, restore

registry = build_registry(with_diffusion=False)
registry.register(RealESRGANx4("..."), default=True)
set_default_registry(registry)
r = restore("prepared/crops/val/<sample_id>.png", "super_resolution")
print(r.success, r.error, r.metadata())
```
