"""Checks for the integrated four-task app (CPU only; real checkpoints are used when installed).

    python -m pytest -q tests/test_app_integration.py

Without checkpoints the model tests are skipped and the registry/app tests still run. Super-resolution
uses randomly initialised bundles in Peng's format (tests/helpers/fake_sr_bundle.py), so it checks the
plumbing, not quality.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.app import create_app  # noqa: E402
from src.inference.factory import DEFAULT_CONFIG, build_registry  # noqa: E402
from src.inference.interface import Registry, TaskHandler  # noqa: E402

RNG = np.random.default_rng(7)


def _photo(h=96, w=128) -> np.ndarray:
    """Smooth synthetic 'leaf' with edges, so restoration has something to recover."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([60 + 80 * np.sin(x / 9), 120 + 70 * np.cos(y / 11), 50 + 40 * np.sin((x + y) / 15)], -1)
    img[(x - w / 2) ** 2 / (w / 3) ** 2 + (y - h / 2) ** 2 / (h / 3) ** 2 < 1] += 40
    return np.clip(img, 0, 255).astype(np.uint8)


def _config(tmp_path, **overrides) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["include_pretrained"] = False  # never download in tests
    cfg["device"] = "cpu"
    cfg["denoising"]["checkpoint"] = str(tmp_path / "missing_dn.pth")
    cfg["colorization"]["checkpoint"] = str(tmp_path / "missing_cz.pth")
    cfg["super_resolution"]["package_dir"] = str(tmp_path / "no_sr")
    cfg["inpainting"]["lora_path"] = str(tmp_path / "no_lora")
    for key, value in overrides.items():
        cfg[key].update(value) if isinstance(value, dict) else cfg.__setitem__(key, value)
    return cfg


def _psnr(a, b) -> float:
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float(10 * np.log10(255 ** 2 / mse))


# --------------------------------------------------------------------------- registry without weights
def test_registry_without_checkpoints_falls_back_to_baselines(tmp_path):
    registry = build_registry(_config(tmp_path), with_diffusion=False)
    tasks = {t["task"]: t for t in registry.describe()}
    assert tasks["denoising"]["default_method"] == "opencv_nlmeans"
    assert tasks["colorization"]["default_method"] == "grayscale_rgb"
    assert tasks["super_resolution"]["default_method"] == "bicubic_x4"
    missing = {(u["task"], u["method"]) for u in registry.unavailable}
    assert {("denoising", "restormer_finetuned"), ("colorization", "ddcolor_finetuned"),
            ("super_resolution", "hat_l"), ("super_resolution", "real_esrgan")} <= missing
    assert all("not installed" in u["reason"] for u in registry.unavailable)


def test_installed_checkpoints_become_the_defaults():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not (ROOT / cfg["denoising"]["checkpoint"]).is_file() or not (ROOT / cfg["colorization"]["checkpoint"]).is_file():
        pytest.skip("Jinyuan's checkpoints are not installed (scripts/install_checkpoints.py)")
    tasks = {t["task"]: t for t in build_registry(cfg, with_diffusion=False).describe()}
    assert tasks["denoising"]["default_method"] == "restormer_finetuned"
    assert tasks["colorization"]["default_method"] == "ddcolor_finetuned"
    kinds = {m["method"]: m["kind"] for m in tasks["denoising"]["methods"]}
    assert kinds["restormer_finetuned"] == "finetuned" and kinds["opencv_nlmeans"] == "baseline"


# --------------------------------------------------------------------------- fixed benchmark degradations
def test_degradations_match_the_task_owners_definitions():
    from src.data.benchmark_examples import add_noise_fixed, bicubic_lr, jinyuan_seed

    digest = hashlib.sha256(b"noise|leaf_0001|15|42").digest()
    assert jinyuan_seed("noise", "leaf_0001", 15, 42) == int.from_bytes(digest[:8], "little") % 2**32
    clean = _photo()
    noisy = add_noise_fixed(clean, 15, "leaf_0001")
    assert np.array_equal(noisy, add_noise_fixed(clean, 15, "leaf_0001")), "noise must be identical on reruns"
    assert not np.array_equal(noisy, add_noise_fixed(clean, 15, "leaf_0002"))
    assert 13 < float(np.std(noisy.astype(int) - clean.astype(int))) < 16

    hr = Image.fromarray(clean)
    lr = bicubic_lr(hr)
    assert lr.size == (32, 24)
    assert np.array_equal(np.asarray(lr), np.asarray(hr.resize((32, 24), Image.Resampling.BICUBIC, reducing_gap=None)))


# --------------------------------------------------------------------------- real models (if installed)
def _real_registry():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["include_pretrained"], cfg["device"] = False, "cpu"
    return build_registry(cfg, with_diffusion=False), cfg


def test_restormer_denoises_and_tiles_without_seams():
    registry, cfg = _real_registry()
    if not (ROOT / cfg["denoising"]["checkpoint"]).is_file():
        pytest.skip("denoising checkpoint not installed")
    from src.data.benchmark_examples import add_noise_fixed

    clean = _photo(96, 128)
    noisy = add_noise_fixed(clean, 15, "t")
    result = registry.restore(noisy, "denoising", method="restormer_finetuned")
    assert result.success, result.error
    assert result.output_size == (128, 96)
    assert _psnr(clean, result.image) > _psnr(clean, noisy) + 4

    handler = registry.handler("denoising", "restormer_finetuned")
    handler.tile, handler.halo = 64, 16  # force tiling on a small image
    tiled = registry.restore(noisy, "denoising", method="restormer_finetuned")
    assert tiled.success and any("tiles" in n for n in tiled.extra.get("notes", []))
    whole = handler._denoise(noisy)
    assert np.abs(whole.astype(int) - np.asarray(tiled.image).astype(int)).mean() < 1.0


def test_ddcolor_keeps_the_input_lightness():
    registry, cfg = _real_registry()
    if not (ROOT / cfg["colorization"]["checkpoint"]).is_file():
        pytest.skip("colorization checkpoint not installed")
    from src.inference.ddcolor_colorize import is_neutral_gray, to_gray_u8

    gray = to_gray_u8(_photo(64, 80), "cpu")
    assert is_neutral_gray(gray)
    result = registry.restore(gray, "colorization", method="ddcolor_finetuned")
    assert result.success, result.error
    assert result.output_size == (80, 64) and result.processed_size == (512, 512)
    assert not result.extra.get("notes"), "an already-gray input must not be converted again"
    regray = to_gray_u8(np.asarray(result.image), "cpu")
    assert np.abs(regray.astype(int) - gray.astype(int)).mean() < 2.0, "colorization must keep L*"

    colour = registry.restore(_photo(64, 80), "colorization", method="ddcolor_finetuned")
    assert colour.success and any("converted" in n for n in colour.extra["notes"])


# --------------------------------------------------------------------------- super-resolution plumbing
@pytest.fixture(scope="module")
def fake_sr(tmp_path_factory):
    from tests.helpers.fake_sr_bundle import write_fake_bundles

    package = tmp_path_factory.mktemp("sr") / "plantsr_sr"
    package.mkdir()
    for name in ("__init__.py", "models.py", "inference.py"):
        (package / name).write_bytes((ROOT / "plantsr_sr" / name).read_bytes())
    write_fake_bundles(package / "weights", ["real_esrgan"])
    return package


def test_super_resolution_through_restore(tmp_path, fake_sr):
    cfg = _config(tmp_path, super_resolution={"package_dir": str(fake_sr), "methods": ["real_esrgan"],
                                              "default": "real_esrgan", "max_input_side": 48})
    registry = build_registry(cfg, with_diffusion=False)
    task = next(t for t in registry.describe() if t["task"] == "super_resolution")
    assert task["default_method"] == "real_esrgan"

    small = registry.restore(_photo(24, 40), "super_resolution")
    assert small.success, small.error
    assert small.output_size == (160, 96) and small.processed_size == (40, 24)

    large = registry.restore(_photo(60, 96), "super_resolution")  # longest side 96 > 48
    assert large.success, large.error
    assert large.processed_size == (48, 30) and large.output_size == (192, 120)
    assert any("reduced" in n for n in large.extra["notes"])


def test_plantsr_package_matches_pengs_notebook():
    notebook = ROOT / "notebooks" / "PlantSR_x4_Super_Resolution_Colab_combined_final.ipynb"
    if not notebook.is_file():
        pytest.skip("Peng's SR notebook is not in this checkout")
    source = "\n".join("".join(c["source"]) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"])
    blocks = dict(re.findall(r"\(PACKAGE_DIR\s*/\s*['\"]([\w.]+\.py)['\"]\)\.write_text\(r'''(.*?)''',\s*encoding=",
                             source, re.S))
    assert {"models.py", "inference.py"} <= set(blocks)
    for name in ("models.py", "inference.py"):
        repo = (ROOT / "plantsr_sr" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert repo == blocks[name], f"plantsr_sr/{name} differs from the code exported by Peng's notebook"


# --------------------------------------------------------------------------- GPU memory recovery
class _FakeGPU(TaskHandler):
    uses_gpu = True
    description = "fake"

    def __init__(self, task, method, fail_times=0):
        super().__init__()
        self.task, self.method, self.label, self.fail_times = task, method, method, fail_times

    def load(self):
        self._loaded = True

    def run(self, image, mask, seed):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return image, image.size


def test_out_of_memory_unloads_other_models_and_retries_once():
    registry = Registry(max_gpu_models=4)
    other = registry.register(_FakeGPU("colorization", "other"), default=True)
    flaky = registry.register(_FakeGPU("denoising", "flaky", fail_times=1), default=True)
    image = _photo(16, 16)
    assert registry.restore(image, "colorization").success and other.loaded
    result = registry.restore(image, "denoising")
    assert result.success, result.error
    assert not other.loaded and flaky.loaded
    assert any("unloaded" in n for n in result.extra["notes"])

    hopeless = registry.register(_FakeGPU("super_resolution", "hopeless", fail_times=5), default=True)
    failed = registry.restore(image, "super_resolution")
    assert not failed.success and "GPU out of memory" in failed.error and not hopeless.loaded


def test_gpu_model_limit_evicts_the_least_recently_used():
    registry = Registry(max_gpu_models=2)
    a, b, c = (registry.register(_FakeGPU(t, t[:3]), default=True)
               for t in ("denoising", "colorization", "super_resolution"))
    image = _photo(16, 16)
    for task in ("denoising", "colorization", "super_resolution"):
        assert registry.restore(image, task).success
    assert (a.loaded, b.loaded, c.loaded) == (False, True, True)


# --------------------------------------------------------------------------- app: logs and feedback
def _png(arr) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _wait(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_jobs_and_feedback_are_logged_for_user_testing(tmp_path):
    client = create_app(build_registry(with_models=False), log_dir=tmp_path / "logs").test_client()
    health = client.get("/api/health").get_json()
    assert "unavailable" in health and "default_models" in health

    job = client.post("/api/jobs", content_type="multipart/form-data", data={
        "task": "denoising", "source": "upload", "session": "s1", "tester": "T1",
        "image": (io.BytesIO(_png(_photo(32, 32))), "a.png")}).get_json()
    done = _wait(client, job["id"])
    assert done["status"] == "done"

    assert client.post("/api/feedback", json={"job_id": job["id"], "rating": 9}).status_code == 400
    assert client.post("/api/feedback", json={"job_id": "nope", "rating": 3}).status_code == 404
    sent = client.post("/api/feedback", json={"job_id": job["id"], "rating": 4, "comment": "Mask brush\nwas small",
                                              "session": "s1", "tester": "T1"})
    assert sent.status_code == 201

    jobs = list(csv.DictReader((tmp_path / "logs" / "jobs.csv").open(encoding="utf-8")))
    feedback = list(csv.DictReader((tmp_path / "logs" / "feedback.csv").open(encoding="utf-8")))
    assert jobs[0]["job_id"] == job["id"] and jobs[0]["tester"] == "T1" and jobs[0]["status"] == "done"
    assert jobs[0]["input_size"] == "32x32" and float(jobs[0]["run_s"]) >= 0
    assert feedback[0]["rating"] == "4" and feedback[0]["comment"] == "Mask brush was small"
    assert feedback[0]["task"] == "denoising"


def test_logging_is_off_without_a_log_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = create_app(build_registry(with_models=False)).test_client()
    job = client.post("/api/jobs", content_type="multipart/form-data", data={
        "task": "colorization", "source": "upload", "image": (io.BytesIO(_png(_photo(16, 16))), "a.png")}).get_json()
    _wait(client, job["id"])
    assert not list(tmp_path.rglob("*.csv"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
