"""Checks for the inpainting/app review fixes (CPU only, about a minute).

    python -m pytest -q tests/test_inpainting_fixes.py     # from the repository root

1. Mask uploads: 0/255 and 0/1 PNG masks give the same coverage through the API.
2. Jobs keep the task and method they were submitted with.
3. Training batches: each epoch draws new crops/masks (itertools.cycle replayed epoch 1).
4. The notebook's %%writefile cells match the repository files.
"""
from __future__ import annotations

import io
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.app import create_app  # noqa: E402
from src.data.inpaint_masks import binarize_mask, load_mask_png  # noqa: E402
from src.inference.factory import build_registry  # noqa: E402
from src.inference.interface import normalize_mask  # noqa: E402


def _png(array: np.ndarray, mode: str | None = None) -> bytes:
    image = Image.fromarray(array)
    if mode:
        image = image.convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _client():
    return create_app(build_registry(with_diffusion=False)).test_client()


def _wait(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError(job_id)


# --------------------------------------------------------------------------- 1. mask encodings
SQUARE = np.zeros((32, 32), np.uint8)
SQUARE[8:16, 8:16] = 1  # 8x8 on 32x32 = 6.25%


def test_binarize_accepts_both_encodings():
    for encoded in (SQUARE, SQUARE * 255, SQUARE.astype(bool), SQUARE.astype(np.float32)):
        assert binarize_mask(encoded).mean() == 0.0625
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., :3] = SQUARE[..., None] * 255
    rgba[..., 3] = 255
    assert binarize_mask(rgba).mean() == 0.0625
    assert not binarize_mask(np.zeros((32, 32), np.uint8)).any()


def test_normalize_mask_png_encodings(tmp_path):
    for name, payload in (("0_255", _png(SQUARE * 255)), ("0_1", _png(SQUARE)), ("1bit", _png(SQUARE * 255, "1"))):
        path = tmp_path / f"{name}.png"
        path.write_bytes(payload)
        assert normalize_mask(path, (32, 32)).mean() == 0.0625, name
        assert normalize_mask(Image.open(path), (32, 32)).mean() == 0.0625, name
        assert load_mask_png(path).mean() == 0.0625, name
    # a 0/1 mask at a different size is resized with nearest neighbour, keeping coverage
    assert normalize_mask(Image.fromarray(SQUARE), (64, 64)).mean() == 0.0625


def test_api_mask_upload_0_255_and_0_1_give_same_coverage():
    client = _client()
    image = _png(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    for name, mask in (("0/255", _png(SQUARE * 255)), ("0/1", _png(SQUARE))):
        response = client.post("/api/jobs", content_type="multipart/form-data", data={
            "task": "inpainting", "method": "opencv_telea", "source": "upload",
            "image": (io.BytesIO(image), "image.png"), "mask": (io.BytesIO(mask), "mask.png")})
        assert response.status_code == 202, response.get_json()
        job = _wait(client, response.get_json()["id"])
        assert job["status"] == "done", (name, job["error"])
        assert job["result"]["mask_coverage"] == 0.0625, name


# --------------------------------------------------------------------------- 2. job pairing
def test_jobs_keep_their_submitted_task_and_method():
    client = _client()
    image = _png(np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    first = client.post("/api/jobs", content_type="multipart/form-data", data={
        "task": "inpainting", "method": "opencv_ns", "source": "upload",
        "image": (io.BytesIO(image), "a.png"), "mask": (io.BytesIO(_png(SQUARE * 255)), "m.png")}).get_json()
    second = client.post("/api/jobs", content_type="multipart/form-data", data={
        "task": "colorization", "source": "upload", "image": (io.BytesIO(image), "b.png")}).get_json()
    done_first, done_second = _wait(client, first["id"]), _wait(client, second["id"])
    assert (done_first["task"], done_first["method"]) == ("inpainting", "opencv_ns")
    assert (done_first["result"]["task"], done_first["result"]["method"]) == ("inpainting", "opencv_ns")
    assert (done_second["task"], done_second["result"]["task"]) == ("colorization", "colorization")


# --------------------------------------------------------------------------- 3. training batches
def _tiny_training_set(tmp_path, n=3):
    rows = []
    rng = np.random.default_rng(2)
    for i in range(n):
        rel = f"images/train/img{i}.png"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (160, 200, 3), dtype=np.uint8)).save(tmp_path / rel)
        rows.append({"sample_id": f"tr_{i}", "hr_path": rel})
    return rows


def _epoch_signature(batches, n):
    out = {}
    for batch in itertools.islice(batches, n):
        meta = batch["meta"][0]
        out[meta["sample_id"]] = (meta["crop_x"], meta["crop_y"], meta["flip"], round(meta["coverage"], 5),
                                  float(batch["mask"].sum()))
    return out


def test_each_epoch_draws_new_crops_and_masks(tmp_path):
    import torch

    from src.train.inpaint_lora import TrainCropDataset, _collate, epoch_batches

    rows = _tiny_training_set(tmp_path)
    dataset = TrainCropDataset(tmp_path, rows, crop_size=64, work_res=64, seed=42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate,
                                         generator=torch.Generator().manual_seed(42))
    batches = epoch_batches(loader)
    epochs = [_epoch_signature(batches, len(rows)) for _ in range(3)]
    assert all(set(e) == {r["sample_id"] for r in rows} for e in epochs)  # every image once per epoch
    assert epochs[0] != epochs[1] and epochs[1] != epochs[2]  # new crops/masks each epoch

    # The old itertools.cycle(loader) replays epoch 1 exactly, which is the bug this fixes.
    old = itertools.cycle(torch.utils.data.DataLoader(
        TrainCropDataset(tmp_path, rows, crop_size=64, work_res=64, seed=42), batch_size=1, shuffle=True,
        num_workers=0, collate_fn=_collate, generator=torch.Generator().manual_seed(42)))
    assert _epoch_signature(old, len(rows)) == _epoch_signature(old, len(rows))


def test_first_epoch_unchanged_by_fix(tmp_path):
    """The 100-step trial ran inside epoch 1, so the fix does not change its batches."""
    import torch

    from src.train.inpaint_lora import TrainCropDataset, _collate, epoch_batches

    rows = _tiny_training_set(tmp_path)

    def loader():
        return torch.utils.data.DataLoader(TrainCropDataset(tmp_path, rows, crop_size=64, work_res=64, seed=42),
                                           batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate,
                                           generator=torch.Generator().manual_seed(42))

    assert _epoch_signature(epoch_batches(loader()), 3) == _epoch_signature(itertools.cycle(loader()), 3)


# --------------------------------------------------------------------------- 4. notebook sync
WRITEFILE = re.compile(r'^%%writefile\s+"?\{CODE_DIR\}/([^"\s]+)"?\s*$')


def test_notebook_writefile_cells_match_repository():
    notebooks = sorted((ROOT / "notebooks").glob("PlantSR_Inpainting_Week1*.ipynb"))
    assert notebooks, "inpainting notebook not found"
    checked = 0
    for notebook in notebooks:
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
            source = "".join(cell["source"])
            match = WRITEFILE.match(source.split("\n", 1)[0]) if cell["cell_type"] == "code" else None
            if not match:
                continue
            body = source.split("\n", 1)[1] if "\n" in source else ""
            repo_file = ROOT / match.group(1)
            assert repo_file.is_file(), f"{notebook.name}: {match.group(1)} is not in the repository"
            repo_text = repo_file.read_text(encoding="utf-8").replace("\r\n", "\n")
            assert body.rstrip("\n") == repo_text.rstrip("\n"), (
                f"{notebook.name}: the %%writefile cell for {match.group(1)} differs from the repository file")
            checked += 1
    assert checked >= 15, f"only {checked} %%writefile cells found"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
