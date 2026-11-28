"""Build a folder of test images for checking every app feature by hand (and for user-testing sessions).

    python scripts/make_test_kit.py --images /content/plantsr/prepared/images/val --out test_kit

`--images` can be any folder of plant photos (JPG/PNG). The script picks the most detailed ones and writes
inputs that exercise each task, the pipeline, and the edge cases, plus CHECKLIST.md with what to expect.
Degradations match the benchmark: seeded Gaussian noise, Pillow bicubic x4, CIE Lab L* grayscale.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.benchmark_examples import add_noise_fixed, bicubic_lr  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def detail_score(image: Image.Image) -> float:
    """Edge energy plus colourfulness: favours sharp, leafy, colourful photos."""
    small = image.convert("RGB").resize((256, 256))
    gray = np.asarray(small.convert("L"), np.float32)
    edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    rgb = np.asarray(small, np.float32)
    colour = np.std(rgb[..., 0] - rgb[..., 1]) + np.std(0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2])
    return float(edges + 0.3 * colour)


def pick_photos(folder: Path, count: int) -> list[tuple[str, Image.Image]]:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No JPG/PNG images found in {folder}")
    scored = []
    for path in paths[:400]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            scored.append((detail_score(image), path.stem, image.copy()))
    scored.sort(key=lambda item: -item[0])
    picked = [(stem, image) for _, stem, image in scored[:count]]
    while len(picked) < count:  # tiny folders: reuse photos
        picked.append(picked[len(picked) % len(scored)])
    return picked


def center_crop(image: Image.Image, size: int) -> Image.Image:
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def lab_gray(image: Image.Image) -> Image.Image:
    """Neutral grayscale from CIE L* (what the app and the benchmark use)."""
    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(np.asarray(image.convert("RGB")))
    lab[..., 1:] = 0
    return Image.fromarray(np.clip(np.round(lab2rgb(lab) * 255), 0, 255).astype(np.uint8))


def blob_mask(size: tuple[int, int], seed: int, coverage: float = 0.06) -> Image.Image:
    """A soft-edged, leaf-damage-like mask: white = area to fill."""
    rng = np.random.default_rng(seed)
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    target = coverage * w * h
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    while (np.asarray(mask) > 0).sum() < target:
        r = rng.uniform(0.03, 0.08) * min(w, h)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        cx = float(np.clip(cx + rng.normal(0, r), 0.1 * w, 0.9 * w))
        cy = float(np.clip(cy + rng.normal(0, r), 0.1 * h, 0.9 * h))
    return mask.filter(ImageFilter.MaxFilter(3)).point(lambda v: 255 if v > 0 else 0)


def damage(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Show the masked area as a pale 'hole' so testers know where to paint."""
    out = np.asarray(image.convert("RGB")).copy()
    out[np.asarray(mask) > 0] = (235, 235, 230)
    return Image.fromarray(out)


def save(image: Image.Image, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **kwargs)


def build(images: Path, out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    photos = pick_photos(images, 6)
    a, b, c, d, e, f = (img for _, img in photos)

    # 1 denoising -------------------------------------------------------------
    clean = center_crop(a, 256)
    save(clean, out / "1_denoising" / "reference_clean.png")
    save(Image.fromarray(add_noise_fixed(np.asarray(clean), 15, "kit-a")), out / "1_denoising" / "noisy_sigma15.png")
    save(Image.fromarray(add_noise_fixed(np.asarray(clean), 25, "kit-a")), out / "1_denoising" / "noisy_sigma25.png")
    big = a.resize((a.width * 1600 // max(a.size), a.height * 1600 // max(a.size)), Image.Resampling.LANCZOS)
    save(Image.fromarray(add_noise_fixed(np.asarray(big), 15, "kit-big")), out / "1_denoising" / "noisy_large_1600px.jpg",
         quality=95)

    # 2 super-resolution -------------------------------------------------------
    hr = center_crop(b, 256)
    save(hr, out / "2_super_resolution" / "reference_256.png")
    save(bicubic_lr(hr), out / "2_super_resolution" / "low_res_64.png")
    save(bicubic_lr(center_crop(b, 512)), out / "2_super_resolution" / "low_res_128.png")
    save(b.resize((b.width * 800 // max(b.size), b.height * 800 // max(b.size)), Image.Resampling.LANCZOS),
         out / "2_super_resolution" / "too_big_800px.jpg", quality=92)

    # 3 colorization -----------------------------------------------------------
    colour = center_crop(c, 512)
    save(colour, out / "3_colorization" / "reference_colour.png")
    save(lab_gray(colour), out / "3_colorization" / "grayscale.png")
    save(lab_gray(colour).convert("L"), out / "3_colorization" / "grayscale_single_channel.png")
    save(colour, out / "3_colorization" / "colour_photo.jpg", quality=92)

    # 4 inpainting -------------------------------------------------------------
    leaf = center_crop(d, 512)
    mask = blob_mask(leaf.size, seed=4)
    save(leaf, out / "4_inpainting" / "photo.png")
    save(mask, out / "4_inpainting" / "mask.png")
    save(Image.fromarray((np.asarray(mask) > 0).astype(np.uint8)), out / "4_inpainting" / "mask_values_0_1.png")
    save(mask.resize((256, 256), Image.Resampling.NEAREST), out / "4_inpainting" / "mask_wrong_size_256.png")
    save(damage(leaf, mask), out / "4_inpainting" / "photo_with_hole.png")

    # 5 pipelines --------------------------------------------------------------
    base = center_crop(e, 512)
    pmask = blob_mask(base.size, seed=5)
    broken = damage(Image.fromarray(add_noise_fixed(np.asarray(lab_gray(base)), 15, "kit-p")), pmask)
    save(base, out / "5_pipeline" / "A_reference_colour.png")
    save(broken, out / "5_pipeline" / "A_old_photo_noisy_gray_with_hole.png")
    save(pmask, out / "5_pipeline" / "A_mask.png")
    small = bicubic_lr(center_crop(e, 384))
    save(Image.fromarray(add_noise_fixed(np.asarray(small), 15, "kit-q")), out / "5_pipeline" / "B_small_noisy_96.png")

    # 6 edge cases ---------------------------------------------------------------
    edge = out / "6_edge_cases"
    rgba = center_crop(f, 384).convert("RGBA")
    alpha = Image.new("L", rgba.size, 0)
    ImageDraw.Draw(alpha).ellipse((24, 24, 360, 360), fill=255)
    rgba.putalpha(alpha)
    save(rgba, edge / "transparent_background.png")
    rotated = center_crop(f, 384).resize((384, 256))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    save(rotated, edge / "phone_photo_rotated_exif.jpg", quality=92, exif=exif)
    save(center_crop(f, 24), edge / "tiny_24px.png")
    save(center_crop(f, 384).convert("CMYK"), edge / "cmyk.jpg", quality=92)
    huge = f.resize((6000, 6000 * f.height // f.width), Image.Resampling.BICUBIC)
    save(huge, edge / "huge_6000px.jpg", quality=85)
    (edge / "not_really_an_image.png").write_text("this is a text file with a .png name\n", encoding="utf-8")
    save(center_crop(f, 256), edge / "photo.webp", quality=90)

    (out / "CHECKLIST.md").write_text(CHECKLIST, encoding="utf-8")
    return sorted(p for p in out.rglob("*") if p.is_file())


CHECKLIST = """# PlantSR app: manual test checklist

Tick each row on the deployed app (Colab link or the PC + tunnel). "Details" = the table under the result.

## 1 · Single tasks (One task mode, default fine-tuned model)

| # | Do this | Files | Expect |
|---|---|---|---|
| 1.1 | Denoising, upload | `1_denoising/noisy_sigma15.png` | Grain gone, fine veins kept; compare with `reference_clean.png`. Method "Restormer · fine-tuned" |
| 1.2 | Denoising, stronger noise | `noisy_sigma25.png` | Still clearly cleaner (trained on σ 5–25) |
| 1.3 | Denoising, large photo | `noisy_large_1600px.jpg` | Output same size as input; Note "Processed in overlapping tiles" |
| 1.4 | Switch method to NL-means and Restormer pretrained on 1.1 | same | NL-means visibly blurrier; both run |
| 2.1 | Super-resolution | `2_super_resolution/low_res_64.png` | Output 256×256 (4×); sharper than bicubic; compare with `reference_256.png` |
| 2.2 | Same with Real-ESRGAN, bicubic, HAT-L pretrained | same | All return 256×256; bicubic softest |
| 2.3 | Super-resolution, larger input | `low_res_128.png` | Output 512×512 |
| 2.4 | Super-resolution, too big | `too_big_800px.jpg` | Note "Input reduced to … before upscaling"; output 4× the reduced size |
| 3.1 | Colorization of a gray photo | `3_colorization/grayscale.png` | Plausible greens; no "converted" note; compare with `reference_colour.png` |
| 3.2 | Single-channel gray PNG | `grayscale_single_channel.png` | Works the same as 3.1 |
| 3.3 | Colour photo | `colour_photo.jpg` | Note "converted to grayscale (CIE L*) first"; output recoloured |
| 4.1 | Inpainting, paint the mask | `4_inpainting/photo_with_hole.png` | Run button disabled until you paint; paint over the pale hole; only painted area changes |
| 4.2 | Inpainting, upload mask | `photo.png` + `mask.png` | Coverage ≈ 6%; filled region blends with the leaf |
| 4.3 | Mask stored as 0/1 | `photo.png` + `mask_values_0_1.png` | Same coverage as 4.2 (not 0%) |
| 4.4 | Mask of a different size | `photo.png` + `mask_wrong_size_256.png` | Accepted; mask stretched to the photo; works |
| 4.5 | Eraser / Clear / Show toggle | any | Coverage updates; Clear sets 0% and disables Run |
| 4.6 | Same seed twice, then a new seed | 4.2 | Same seed = identical output; new seed = different fill |

## 2 · Benchmark examples (metrics only here)

| # | Do this | Expect |
|---|---|---|
| 5.1 | Benchmark example tab, each task, run | PSNR/SSIM shown (ΔE00 for colorization); inpainting shows masked metrics |
| 5.2 | Upload a photo afterwards | No PSNR/SSIM for uploads |

## 3 · Pipelines

| # | Do this | Files | Expect |
|---|---|---|---|
| 6.1 | Default pipeline Denoising → Colorization → Inpainting; paint/upload mask | `5_pipeline/A_old_photo_noisy_gray_with_hole.png` + `A_mask.png` | Status "Step 1 of 3 …"; strip with 3 thumbnails; final is clean, coloured, hole filled. Compare with `A_reference_colour.png` |
| 6.2 | Click each thumbnail; download one step's PNG | 6.1 | Output switches; file named `…step2_colorization…png` |
| 6.3 | Reorder by dragging ⠿ and with ↑ ↓; remove ✕; Add step | — | Order and numbers update; removed task appears in "Add step" |
| 6.4 | Denoising → Super-resolution | `B_small_noisy_96.png` | 384×384 result, clean and sharp |
| 6.5 | Super-resolution → Denoising (reversed) | same | Warning about the 4× larger image; slower than 6.4. Compare the two results: order matters |
| 6.6 | Super-resolution → Inpainting with a mask | `4_inpainting/photo.png` + `mask.png` | Note "the mask was resized to …"; only masked area filled |
| 6.7 | Pipeline without inpainting | any | Mask tools hidden; no mask needed |
| 6.8 | Benchmark tab while in Pipeline | — | Disabled with a tooltip |

## 4 · Edge cases

| # | File | Expect |
|---|---|---|
| 7.1 | `6_edge_cases/transparent_background.png` | Accepted; runs (transparent area becomes a solid colour) |
| 7.2 | `phone_photo_rotated_exif.jpg` | Shown upright (portrait 256×384), output upright too |
| 7.3 | `tiny_24px.png` | Runs for every task (SR gives 96×96) |
| 7.4 | `cmyk.jpg` | Accepted; colours look normal |
| 7.5 | `huge_6000px.jpg` | Accepted (under 40 MP / 25 MB); denoising tiles it (takes longer); colorization fast |
| 7.6 | `not_really_an_image.png` | Clear error "could not be read as a JPG or PNG"; app still usable |
| 7.7 | `photo.webp` | Rejected: "upload a JPG or PNG" |

## 5 · App behaviour

| # | Do this | Expect |
|---|---|---|
| 8.1 | While a job runs | All controls locked; status shows time; Run button disabled |
| 8.2 | Start a job in two browser tabs at once | Second shows "Waiting in queue (position 1)…", then runs |
| 8.3 | Change task while a result is shown, then look at the result | Result still labelled with the task it ran |
| 8.4 | Rate 1–5 + comment, Send feedback | "Thanks, your feedback was saved"; row in `logs/feedback.csv` |
| 8.5 | Open the app as `…/?tester=T9` and run a job | Tester ID prefilled; `logs/jobs.csv` row has `T9` |
| 8.6 | Download PNG | File opens, same size as "Output" |
| 8.7 | Phone or narrow window | Layout stacks; buttons usable; pipeline ↑ ↓ work |
| 8.8 | Status pill (top right) | "GPU · Tesla T4 · 0 in queue" (or your GPU); hover lists loaded models |
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, required=True, help="Folder of plant photos (e.g. prepared/images/val)")
    parser.add_argument("--out", type=Path, default=ROOT / "test_kit")
    parser.add_argument("--zip", action="store_true", help="Also write <out>.zip")
    args = parser.parse_args(argv)
    files = build(args.images, args.out)
    print(f"Wrote {len(files)} files to {args.out}")
    for path in files:
        print("  ", path.relative_to(args.out))
    if args.zip:
        print("ZIP:", shutil.make_archive(str(args.out), "zip", args.out.parent, args.out.name))


if __name__ == "__main__":
    main()
