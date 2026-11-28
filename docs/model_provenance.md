# Model provenance

What each method in the app runs, where its weights come from, and who trained it. Weights are not in
Git; `scripts/install_checkpoints.py` installs and verifies them (`docs/deployment.md`, section 2).

| Task | Method (app id) | Base weights | Fine-tuning | Installed file · SHA-256 | Owner |
| --- | --- | --- | --- | --- | --- |
| Denoising | `restormer_finetuned` (default) | Restormer, `deepinv/Restormer` · `gaussian_color_denoising_blind.pth` | `notebooks/PlantSR_Denoise_Colorize_Finetune.ipynb`, σ ∈ [5, 25], best validation PSNR at σ = 15 | `checkpoints/denoise/best.pth` · `f01c68c05c05…` | Jinyuan |
| Denoising | `restormer_pretrained` | same, without fine-tuning (downloaded on first use) | — | Hugging Face cache | — |
| Colorization | `ddcolor_finetuned` (default) | DDColor tiny, `piddnad/ddcolor_paper_tiny` · `pytorch_model.bin` | same notebook, 512 px input, best validation ΔE00 | `checkpoints/colorize/best.pth` · `d52fcfdb9752…` | Jinyuan |
| Colorization | `ddcolor_pretrained` | same, without fine-tuning | — | Hugging Face cache | — |
| Super-resolution | `hat_l` (default), `real_esrgan` | HAT-L `jaideepsingh/upscale_models` · `HAT/HAT-L_SRx4_ImageNet-pretrain.pth`; Real-ESRGAN `kornia/real_esrgan` · `RealESRGAN_x4plus.safetensors` (revisions pinned in `plantsr_sr/models.py`) | `notebooks/PlantSR_x4_Super_Resolution_Colab_combined_final.ipynb` | `plantsr_sr/weights/{hat_l,real_esrgan}.pt` · SHA-256 in `plantsr_sr/manifest.json` | Peng |
| Super-resolution | `hat_l_pretrained` | HAT-L as above (downloaded and SHA-checked by `plantsr_sr`) | — | Hugging Face cache | — |
| Inpainting | `sd15_pretrained` | `stable-diffusion-v1-5/stable-diffusion-inpainting` (CreativeML OpenRAIL-M) | zero-shot; setting chosen on validation in `notebooks/PlantSR_Inpainting_Week1_Final.ipynb` | Hugging Face cache | Ali |
| Inpainting | `sd15_finetuned` (default when installed) | same + PlantSR LoRA (rank 16, UNet attention) | `notebooks/PlantSR_Inpainting_C2_Finetune.ipynb`, checkpoint chosen by validation missing-region LPIPS | `checkpoints/inpaint/lora/` (from Drive `inpainting_c2/lora_best/`) | Ali |
| all | `opencv_nlmeans`, `bicubic_x4`, `grayscale_rgb`, `opencv_telea`, `opencv_ns` | classical baselines, no weights | — | — | — |

Super-resolution bundle: Peng's `PlantSR_x4_app.zip`, linked from `docs/super_resolution.md` (Google Drive
file id `1j4plR5Zc-MUNYD42HYTR-lCmgLLjZ-VA`, used by `install_checkpoints.py --download-sr`). Peng's own
validation and test results are in `reports/super_resolution/`. Its
`plantsr_sr/models.py` and `inference.py` are committed in the repository; a test checks that they are
identical to the code Peng's notebook exports.

## How the app reproduces each owner's pipeline

| Task | Pre-processing | Model call | Post-processing |
| --- | --- | --- | --- |
| Denoising | RGB in [0, 1]; reflect-pad to a multiple of 8 | bf16 autocast on GPUs with compute capability ≥ 8, else fp16; images larger than 512 px run in overlapping 512 px tiles (32 px halo) | crop padding, round to `uint8` |
| Colorization | colour inputs are converted to Lab L\* gray first (the app says so); L\* kept in fp32 | gray RGB resized to 512 × 512 (bilinear), predicts ab; fp32 or bf16 only | ab resized back, combined with the original L\*, Lab → RGB, clipped |
| Super-resolution | inputs longer than 512 px are reduced first (the app says so) | Peng's `plantsr_sr` tiling (core 128, halo 16; HAT pads to a multiple of 16) | ×4 output |
| Inpainting | mask 1/white = fill; working resolution 512 | SD 1.5 inpainting, 25 DPM-Solver++ steps, guidance 5, fixed seed | composite: only masked pixels change |

Benchmark inputs follow the owners' definitions exactly: Gaussian noise σ = 15 from
`stable_seed("noise", sample_id, σ, 42)`, Pillow 11.3.0 bicubic ×4 with `reducing_gap=None`, Lab L\*
grayscale, and the fixed Week 1 masks. `notebooks/PlantSR_App_Integration.ipynb` checks that the app
reproduces each owner's validation numbers.
