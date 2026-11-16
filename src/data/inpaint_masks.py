"""Reproducible inpainting masks for the PlantSR benchmark.

Convention shared by the whole team: mask value 1 = missing / reconstruct, 0 = preserve.
Masks are saved as single-channel PNGs with 0 (preserve) and 255 (missing).

Two mask families are used, each covering roughly 5-20% of a 256 x 256 crop:
  * rectangle - one to three axis-aligned boxes (block damage, occlusion)
  * freeform  - brush strokes with round joints (scratches, irregular damage)

Validation/test masks are fixed: each one is generated from a seed derived from
(base seed, split, sample_id, generator version), so regenerating them always gives
identical pixels. Training masks are sampled on the fly after cropping/flipping, so the
image and mask always share the same geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GENERATOR_VERSION = "plantsr_inpaint_masks_v1"
COVERAGE_RANGE = (0.05, 0.20)
MASK_TYPES = ("rectangle", "freeform")
FILL_VALUE = 128  # grey written into missing pixels of the saved masked inputs
MANIFEST_FIELDS = [
    # Fields agreed in the team README, in the agreed order.
    "sample_id", "source_image_id", "split", "task", "input_path", "target_path",
    "mask_path", "crop_coordinates", "degradation_parameters", "seed",
    # Extra inpainting columns for quick filtering and integrity checks.
    "mask_type", "mask_coverage", "coverage_bin", "mask_sha256",
]


def seed_for(*parts) -> int:
    """Stable 63-bit seed from arbitrary JSON-serialisable parts."""
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()
    return int(digest[:15], 16)


def coverage(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).mean())


def coverage_bin(value: float) -> str:
    if value < 0.10:
        return "small_5_10"
    if value < 0.15:
        return "medium_10_15"
    return "large_15_20"


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _rectangles(rng: np.random.Generator, h: int, w: int, target: float, upper: float):
    mask = np.zeros((h, w), np.uint8)
    n = int(rng.integers(1, 4))
    shares = rng.dirichlet(np.ones(n)) if n > 1 else np.array([1.0])
    rects = []
    for share in shares:
        area = max(64.0, share * target * h * w)
        aspect = float(np.exp(rng.uniform(np.log(1 / 3), np.log(3))))
        rw = int(np.clip(round(math.sqrt(area * aspect)), 6, int(0.9 * w)))
        rh = int(np.clip(round(area / max(rw, 1)), 6, int(0.9 * h)))
        x = int(rng.integers(0, w - rw + 1))
        y = int(rng.integers(0, h - rh + 1))
        mask[y:y + rh, x:x + rw] = 1
        rects.append([x, y, rw, rh])
    return mask, {"rectangles_xywh": rects}


def _strokes(rng: np.random.Generator, h: int, w: int, target: float, upper: float):
    mask = np.zeros((h, w), np.uint8)
    scale = min(h, w) / 256.0
    strokes, misses = [], 0
    while coverage(mask) < target and len(strokes) < 40 and misses < 20:
        candidate = mask.copy()
        width = max(2, int(round(rng.integers(8, 25) * scale)))
        x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        angle = float(rng.uniform(0, 2 * math.pi))
        points = [(x, y)]
        for _ in range(int(rng.integers(3, 9))):
            angle += float(rng.uniform(-math.pi / 2, math.pi / 2))
            length = float(rng.uniform(10, 40)) * scale
            x = float(np.clip(x + length * math.cos(angle), 0, w - 1))
            y = float(np.clip(y + length * math.sin(angle), 0, h - 1))
            points.append((x, y))
        pts = [(int(round(px)), int(round(py))) for px, py in points]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(candidate, a, b, 1, thickness=width, lineType=cv2.LINE_8)
        for p in pts:
            cv2.circle(candidate, p, width // 2, 1, thickness=-1, lineType=cv2.LINE_8)
        if coverage(candidate) > upper:  # this stroke overshoots; try another one
            misses += 1
            continue
        mask = candidate
        strokes.append({"width": width, "vertices": len(pts)})
    return mask, {"strokes": strokes}


def make_mask(rng: np.random.Generator, height: int = 256, width: int = 256, mask_type: str | None = None,
              target: float | None = None, coverage_range=COVERAGE_RANGE, rel_tol: float = 0.15,
              max_attempts: int = 200):
    """Return (mask uint8 {0,1}, params) whose coverage is close to `target`.

    Candidates are resampled until coverage is within +/- rel_tol of the target and inside
    the global coverage range; the closest valid candidate is kept if no attempt is exact.
    """
    lo, hi = coverage_range
    if mask_type is None:
        mask_type = MASK_TYPES[int(rng.integers(0, len(MASK_TYPES)))]
    if mask_type not in MASK_TYPES:
        raise ValueError(f"Unknown mask type {mask_type!r}; use one of {MASK_TYPES}.")
    if target is None:
        target = float(rng.uniform(lo, hi))
    if not lo <= target <= hi:
        raise ValueError(f"Target coverage {target:.3f} is outside {coverage_range}.")
    draw = _rectangles if mask_type == "rectangle" else _strokes
    upper = min(hi, target * (1 + rel_tol))
    best = None
    for attempt in range(1, max_attempts + 1):
        mask, shapes = draw(rng, height, width, target, upper)
        value = coverage(mask)
        if not lo <= value <= hi:
            continue
        error = abs(value - target) / target
        if best is None or error < best[0]:
            best = (error, mask, shapes, attempt)
        if error <= rel_tol:
            break
    if best is None:
        raise RuntimeError(f"Could not generate a {mask_type} mask inside {coverage_range}.")
    _, mask, shapes, attempt = best
    value = coverage(mask)
    params = {
        "generator": GENERATOR_VERSION,
        "mask_type": mask_type,
        "target_coverage": round(float(target), 5),
        "coverage": round(value, 5),
        "attempts": attempt,
        "bbox_xywh": _bbox(mask),
        "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        **shapes,
    }
    return mask.astype(np.uint8), params


def fixed_eval_masks(sample_ids, split: str, base_seed: int = 42, size=(256, 256), coverage_range=COVERAGE_RANGE):
    """Deterministic masks for a validation/test split.

    Mask types alternate (half rectangle, half free-form) and target coverages are spread
    evenly over the coverage range within each type, so every coverage bin is represented.
    Returns {sample_id: (mask, params, seed)}.
    """
    h, w = size
    lo, hi = coverage_range
    order = sorted(sample_ids, key=lambda sid: seed_for(base_seed, split, "order", sid))
    per_type = math.ceil(len(order) / len(MASK_TYPES))
    out = {}
    for index, sid in enumerate(order):
        mask_type = MASK_TYPES[index % len(MASK_TYPES)]
        k = index // len(MASK_TYPES)
        target = lo + (hi - lo) * (k + 0.5) / per_type
        seed = seed_for(base_seed, split, sid, GENERATOR_VERSION)
        mask, params = make_mask(np.random.default_rng(seed), h, w, mask_type, target, coverage_range)
        out[sid] = (mask, params, seed)
    return out


class TrainingMaskSampler:
    """Random masks for training, sampled after the crop/flip so geometry always matches."""

    def __init__(self, seed: int = 42, coverage_range=COVERAGE_RANGE, rectangle_probability: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.coverage_range = coverage_range
        self.rectangle_probability = rectangle_probability

    def __call__(self, height: int, width: int):
        mask_type = "rectangle" if self.rng.random() < self.rectangle_probability else "freeform"
        target = float(self.rng.uniform(*self.coverage_range))
        return make_mask(self.rng, height, width, mask_type, target, self.coverage_range)


# ----------------------------------------------------------------------------- I/O helpers
def save_mask_png(mask: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(path)


def load_mask_png(path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L")) > 127).astype(np.uint8)


def apply_mask(image: np.ndarray, mask: np.ndarray, fill: int = FILL_VALUE) -> np.ndarray:
    """Copy of an RGB uint8 image with the missing pixels replaced by `fill`."""
    out = np.array(image, dtype=np.uint8, copy=True)
    out[np.asarray(mask) > 0] = fill
    return out


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha: float = 0.5) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32).copy()
    m = np.asarray(mask) > 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    edges = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edges] = color
    return out.clip(0, 255).astype(np.uint8)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rgb(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# ----------------------------------------------------------------------------- eval sets
def build_eval_mask_set(prepared_dir, rows, split: str, base_seed: int = 42,
                        out_rel_dir: str = "derived/inpainting_v1"):
    """Write fixed masks + masked inputs for one split and return manifest rows.

    All paths in the returned rows are relative to `prepared_dir`, like the team manifests.
    """
    prepared_dir = Path(prepared_dir)
    rows = [r for r in rows if r.get("eval_crop_path")]
    if not rows:
        raise ValueError(f"No saved evaluation crops found for split {split!r}.")
    first = read_rgb(prepared_dir / rows[0]["eval_crop_path"])
    masks = fixed_eval_masks([r["sample_id"] for r in rows], split, base_seed, first.shape[:2])
    manifest = []
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        sid = row["sample_id"]
        mask, params, seed = masks[sid]
        target = read_rgb(prepared_dir / row["eval_crop_path"])
        if target.shape[:2] != mask.shape:
            raise ValueError(f"{sid}: crop {target.shape[:2]} does not match mask {mask.shape}.")
        mask_rel = f"{out_rel_dir}/masks/{split}/{sid}.png"
        input_rel = f"{out_rel_dir}/inputs/{split}/{sid}.png"
        save_mask_png(mask, prepared_dir / mask_rel)
        (prepared_dir / input_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(apply_mask(target, mask)).save(prepared_dir / input_rel)
        params = {**params, "fill_value": FILL_VALUE, "base_seed": base_seed}
        manifest.append({
            "sample_id": sid,
            "source_image_id": f"{row['official_split']}/{row['relative_path']}",
            "split": split,
            "task": "inpainting",
            "input_path": input_rel,
            "target_path": row["eval_crop_path"],
            "mask_path": mask_rel,
            "crop_coordinates": json.dumps({k: int(row[f"crop_{k}"]) for k in ("x", "y", "width", "height")}),
            "degradation_parameters": json.dumps(params, separators=(",", ":")),
            "seed": seed,
            "mask_type": params["mask_type"],
            "mask_coverage": f"{params['coverage']:.5f}",
            "coverage_bin": coverage_bin(params["coverage"]),
            "mask_sha256": sha256_file(prepared_dir / mask_rel),
        })
    return manifest


def verify_eval_mask_set(prepared_dir, manifest, source_rows=None, coverage_range=COVERAGE_RANGE,
                         check_hr_alignment: bool = True):
    """Check coverage ratios and pixel alignment for a fixed inpainting set.

    Returns a list of problem strings (empty means every check passed). Checks:
      * files exist; mask, target and input have identical height/width
      * mask is strictly binary and its coverage is within range and matches the manifest
      * the mask PNG hash matches the manifest (masks have not been regenerated differently)
      * input == target on every preserved pixel and == FILL_VALUE on every missing pixel
      * optional: target crop == HR image cropped at the recorded coordinates
    """
    prepared_dir = Path(prepared_dir)
    lo, hi = coverage_range
    source = {r["sample_id"]: r for r in (source_rows or [])}
    problems = []
    for row in manifest:
        sid = row["sample_id"]
        try:
            target = read_rgb(prepared_dir / row["target_path"])
            masked = read_rgb(prepared_dir / row["input_path"])
            with Image.open(prepared_dir / row["mask_path"]) as raw:
                raw_mask = np.asarray(raw.convert("L"))
        except (OSError, FileNotFoundError) as exc:
            problems.append(f"{sid}: cannot read files ({exc})")
            continue
        if not set(np.unique(raw_mask)).issubset({0, 255}):
            problems.append(f"{sid}: mask PNG is not strictly 0/255")
        mask = raw_mask > 127
        if not (target.shape[:2] == masked.shape[:2] == mask.shape):
            problems.append(f"{sid}: size mismatch target {target.shape[:2]}, input {masked.shape[:2]}, mask {mask.shape}")
            continue
        value = float(mask.mean())
        if not lo - 1e-9 <= value <= hi + 1e-9:
            problems.append(f"{sid}: coverage {value:.4f} outside {coverage_range}")
        if abs(value - float(row["mask_coverage"])) > 1e-4:
            problems.append(f"{sid}: coverage {value:.5f} differs from manifest {row['mask_coverage']}")
        if row.get("mask_sha256") and sha256_file(prepared_dir / row["mask_path"]) != row["mask_sha256"]:
            problems.append(f"{sid}: mask hash differs from manifest")
        if not np.array_equal(masked[~mask], target[~mask]):
            problems.append(f"{sid}: preserved pixels of the input differ from the target")
        if not np.all(masked[mask] == FILL_VALUE):
            problems.append(f"{sid}: missing pixels of the input are not the fill value")
        if check_hr_alignment and sid in source and source[sid].get("hr_path"):
            box = json.loads(row["crop_coordinates"])
            with Image.open(prepared_dir / source[sid]["hr_path"]) as hr:
                region = np.asarray(hr.convert("RGB").crop(
                    (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])))
            if not np.array_equal(region, target):
                problems.append(f"{sid}: saved crop does not match the HR image at {box}")
    return problems
