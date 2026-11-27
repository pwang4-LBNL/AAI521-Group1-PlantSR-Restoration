# app

Flask application and user interface for the four restoration tasks.

## Features

- Upload a JPG/PNG (drag and drop) or choose a **benchmark example** (validation crop with a fixed degradation).
- All four tasks with real inference; each task defaults to its fine-tuned model when installed:
  Restormer denoising and DDColor colorization (Jinyuan), HAT-L ×4 super-resolution (Peng),
  Stable Diffusion 1.5 inpainting (Ali). Methods are grouped as fine-tuned, original pretrained and baselines;
  missing checkpoints are listed instead of breaking the app.
- Inpainting mask: paint with a brush or eraser, upload a mask PNG (white = fill; 0/255 or 0/1), live coverage percentage.
- Controls are locked while a job runs, and each result is labelled with the task it was submitted for.
- Processing status with queue position; one background worker runs GPU jobs one at a time, keeps at most
  `max_gpu_models` models loaded and retries once after freeing memory if the GPU runs out.
- Side-by-side input/output, received → processed → output dimensions, model version, timing and notes
  (e.g. "processed in tiles", "converted to grayscale first").
- PSNR/SSIM (and ΔE00 for colorization) shown **only** for benchmark examples, where the reference is known.
- PNG download, and a 1–5 rating with an optional comment for user testing (`logs/feedback.csv`; every job is
  also logged to `logs/jobs.csv`, without images).

## Run

Full setup (Windows GPU PC, weights, tunnel, user testing): `docs/deployment.md`. In short, from the repository root:

```bash
pip install -r app/requirements.txt                  # after PyTorch for your CUDA version
python scripts/install_checkpoints.py --status       # see docs/deployment.md to install weights
python app/app.py --host 0.0.0.0 --port 7860 --preload \
    --prepared-dir /path/to/prepared \
    --inpaint-manifest /path/to/prepared/derived/inpainting_v1/inpainting_val.csv
```

`--baselines-only` starts the app without any model (UI work on any machine); `--no-diffusion` skips Stable Diffusion.
On Colab, `notebooks/PlantSR_App_Integration.ipynb` section 9 starts the app and prints a link.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/tasks` | Tasks, methods and defaults |
| GET | `/api/examples` | Benchmark examples (validation crops) |
| GET | `/api/examples/<id>/input.png?task=…` · `/mask.png` | Degraded input / fixed mask |
| POST | `/api/jobs` | Form fields `task`, `method`, `seed`, `source` (`upload`/`example`), `image`, `mask`, `example_id`, optional `tester`, `session` |
| GET | `/api/jobs/<id>` | Status, timing, sizes, metrics |
| GET | `/api/jobs/<id>/output.png[?download=1]` | Result |
| POST | `/api/feedback` | JSON `job_id`, `rating` (1–5), optional `comment`, `tester`, `session` |
| GET | `/api/health` | Device, queue length, loaded models, defaults, methods not installed |
