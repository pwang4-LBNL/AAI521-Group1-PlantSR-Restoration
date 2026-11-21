# PlantSR 4× Super-Resolution

[Open the Colab notebook](../notebooks/PlantSR_x4_Super_Resolution_Colab_combined_final.ipynb)
  
[Download the verified application bundle from Google Drive](https://drive.google.com/file/d/1j4plR5Zc-MUNYD42HYTR-lCmgLLjZ-VA/view?usp=drive_link)

## Models

| Application option | Fine-tuned checkpoint | Role |
| --- | --- | --- |
| HAT-L | Epoch 2 | Default model |
| Real-ESRGAN | Epoch 10 | Alternative model |

Both models accept an RGB image and upscale its width and height by 4×.
The bundle includes both weights, inference code, dependencies, and a Flask demo.
Checkpoint filenames and SHA-256 hashes are recorded in `plantsr_sr/manifest.json`.

## Verification

Both models passed the Flask upload test in a fresh Python process after the ZIP
was extracted. Each converted a 64 × 64 input into a downloadable 256 × 256 RGB PNG.

## Team app integration

1. Extract `PlantSR_x4_app.zip`.
2. Copy `plantsr_sr/` beside `src/` in the team repository.
3. Use the dependency versions recorded in the bundle's `requirements.txt`.
4. Follow the bundle's `README.md` to register `hat_l` as the default and
   `real_esrgan` as an alternative in `src/inference/factory.py`.
5. In the team app, select super-resolution and test one upload with each model.