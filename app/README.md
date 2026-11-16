# app

Flask application and user interface for the four restoration tasks.

## Features (Week 1 layout)

- Upload a JPG/PNG (drag and drop) or choose a **benchmark example** (validation crop with a fixed degradation).
- Choose a task and method. Baselines are always available; diffusion inpainting runs on the GPU.
- Inpainting mask: paint with a brush or eraser, upload a mask PNG (white = fill), live coverage percentage.
- Processing status with queue position; one background worker runs GPU jobs one at a time.
- Side-by-side input/output, received → processed → output dimensions, model version and timing.
- PSNR/SSIM shown **only** for benchmark examples, where the reference is known.
- PNG download.

## Run

Colab: run `notebooks/PlantSR_Inpainting_Week1.ipynb`; section 9 starts the app and prints a link.

Local GPU machine (from the repository root):

```bash
pip install -r app/requirements.txt          # plus PyTorch for your CUDA version
python app/app.py --config configs/app.json \
    --prepared-dir /path/to/prepared \
    --inpaint-manifest /path/to/prepared/derived/inpainting_v1/inpainting_val.csv \
    --host 0.0.0.0 --port 7860
```

`--no-diffusion` starts the app with baselines only, which is handy for UI work without a GPU.

To let teammates or the instructor open a development version from another device, put a tunnel in front of the local server, for example `cloudflared tunnel --url http://localhost:7860`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/tasks` | Tasks, methods and defaults |
| GET | `/api/examples` | Benchmark examples (validation crops) |
| GET | `/api/examples/<id>/input.png?task=…` · `/mask.png` | Degraded input / fixed mask |
| POST | `/api/jobs` | Form fields `task`, `method`, `seed`, `source` (`upload`/`example`), `image`, `mask`, `example_id` |
| GET | `/api/jobs/<id>` | Status, timing, sizes, metrics |
| GET | `/api/jobs/<id>/output.png[?download=1]` | Result |
| GET | `/api/health` | Device, queue length, loaded models |
