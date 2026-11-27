# Deployment and user testing

The app is a Flask application served by **waitress** with one background GPU worker. It runs all four
tasks with real inference: Restormer denoising and DDColor colorization (fine-tuned by Jinyuan),
HAT-L / Real-ESRGAN ×4 super-resolution (fine-tuned by Peng) and Stable Diffusion 1.5 inpainting (Ali).

| Where | Use it for | How |
| --- | --- | --- |
| Windows PC with an NVIDIA GPU | User testing, the demo, access from other devices | This page, sections 1–5 |
| Google Colab (T4) | Development, the validation/test evaluation, quick team previews | `notebooks/PlantSR_App_Integration.ipynb` |

Colab's proxy link only works while the notebook runs and is tied to the Colab account, so testers and
the demo use the PC deployment.

## 1. Environment (Windows, RTX 4080, Python 3.12)

From Git Bash or PowerShell in the repository root:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r app/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. If it prints `False`, update the NVIDIA driver and reinstall
PyTorch from the CUDA index above (a plain `pip install torch` gives the CPU build on Windows).

Requirements pin the versions the task owners trained with (`spandrel 0.4.2`, `spandrel_extra_arches 0.2.0`,
`kornia 0.9.1`, `Pillow 11.3.0`).

## 2. Model weights

Weights are never committed (see `.gitignore`). Download the files from the team Drive and install them:

| Task | File in Drive | Owner |
| --- | --- | --- |
| Denoising | `Denoise/best.pth` (or the ZIP of that folder) | Jinyuan |
| Colorization | `Colorize/best.pth` (or the ZIP of that folder) | Jinyuan |
| Super-resolution | `PlantSR_x4_app.zip` (`--download-sr` fetches it) | Peng |
| Inpainting | LoRA folder from the full fine-tune (not final yet) | Ali |

```bash
python scripts/install_checkpoints.py \
    --denoise ~/Downloads/Denoise.zip --colorize ~/Downloads/Colorize.zip \
    --sr ~/Downloads/PlantSR_x4_app.zip
python scripts/install_checkpoints.py --status        # what the app will use
```

The script checks each file's SHA-256 against `configs/app.json` (denoising, colorization) or Peng's
`manifest.json` (super-resolution), and confirms that `plantsr_sr/*.py` in the repository is identical to the
code in Peng's bundle. A missing checkpoint never stops the app: that method is listed as "not installed"
and the task falls back to the original pretrained model or a baseline.

The first inpainting or pretrained-model request downloads weights from Hugging Face (about 2.5 GB for
SD 1.5 inpainting) into the Hugging Face cache. Run one inpainting job before testers arrive.

## 3. Start the server

```bash
python app/app.py --host 0.0.0.0 --port 7860 --preload \
    --prepared-dir /path/to/prepared \
    --inpaint-manifest /path/to/prepared/derived/inpainting_v1/inpainting_val.csv
```

| Option | Meaning |
| --- | --- |
| `--preload` | Load every task's default model at start-up, so the first tester does not wait for loading |
| `--max-gpu-models N` | Keep at most N large models on the GPU (default 4; least recently used is unloaded) |
| `--prepared-dir`, `--inpaint-manifest` | Enable benchmark examples (validation crops with known references, the only place PSNR/SSIM is shown) |
| `--log-dir` | Where `jobs.csv` and `feedback.csv` are written (default `logs/`) |
| `--no-diffusion`, `--baselines-only` | Development without Stable Diffusion / without any model |

At start-up the server prints the model table (`*` = default, `!` = not installed). Check it, then open
<http://localhost:7860>. `GET /api/health` reports the device, loaded models, defaults and missing methods.

GPU work runs one job at a time in a queue; other requests wait and see their queue position. If a request
runs out of GPU memory, the other models are unloaded and the job is retried once. Large uploads are handled by
tiling (denoising, super-resolution) or by resizing to the model's input size (colorization, inpainting);
the result panel shows the received, processed and returned dimensions and explains any change.

## 4. Access from another device

Same Wi-Fi: open `http://<PC IP address>:7860` (`ipconfig` shows the IPv4 address). Allow Python through
Windows Defender Firewall for private networks when Windows asks.

Anywhere (testers at home, the instructor): a Cloudflare quick tunnel gives a temporary HTTPS address and needs no account.

```bash
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:7860
```

Share the `https://….trycloudflare.com` address it prints; it stops working when you close the tunnel. Anyone
with the address can use the app while it is up, so only run the tunnel during testing sessions.

**Second-device check (plan item C5):** on a phone using mobile data, open the tunnel address, run one task
with an uploaded photo and one inpainting job with a drawn mask, download both PNGs, and note the time in
`reports/user_testing/README.md`.

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| A method shows "not installed" | Run `scripts/install_checkpoints.py --status`; install the missing file |
| `SHA-256 … does not match` | The file differs from the one the owner reported. Ask the owner; `--force` installs it and you must update `configs/app.json` |
| `GPU out of memory` | Lower `--max-gpu-models`, or use a smaller image; the app already retries once after unloading other models |
| `torch.cuda.is_available()` is `False` | Reinstall PyTorch from the CUDA index (section 1) |
| Pillow version warning in the SR notebook | `pip install Pillow==11.3.0` |
| Page loads but jobs never finish | The worker is still loading a model (first request); watch the console |

## 6. User testing (3–5 volunteers)

**Before:** start the server with `--preload`, run one job per task, open the tunnel, and give each tester an ID
(T1, T2, …). Open the app for each tester as `<address>/?tester=T1`, so every job and rating is labelled
with the ID (testers can also type it into the feedback form; it is remembered in that browser).

**Script for each tester (about 10 minutes):**

1. Upload one of your own plant photos (JPG or PNG) and run **denoising**; compare and download the result.
2. Run **super-resolution** on a small photo or a benchmark example.
3. Run **colorization** on any photo (the app converts it to grayscale first and says so).
4. Run **inpainting**: paint over part of a leaf (or upload a mask), run it, download the result.
5. After each result, give a 1–5 rating and say what was confusing, slow or wrong.

**Recorded automatically** (no images are stored):

| File | Columns |
| --- | --- |
| `logs/jobs.csv` | time, session, tester, task, method, model version, input/processed/output size, queue wait, run time, status, error |
| `logs/feedback.csv` | time, session, tester, job, task, method, rating 1–5, comment |

**Recorded by the observer** in `reports/user_testing/observations.csv`: tester, step, completed (yes/no/with help),
confusing controls, perceived quality, perceived waiting time, notes.

**After:** summarise completion rate per step, median waiting time per task (from `jobs.csv`), mean rating per
task and the recurring comments; list each issue, the fix, and the retest result in
`reports/user_testing/README.md`.
