"""PlantSR restoration web application (Flask).

Local / GPU machine:
    python app/app.py --config configs/app.json --prepared-dir /path/to/prepared --port 7860
Colab: the Week 1 notebook builds the registry and calls `create_app(...)` directly.

All GPU work goes through one background worker thread, so requests are queued instead of
competing for GPU memory; the registry keeps at most one large model loaded at a time.
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.interface import TASKS, Registry, normalize_mask  # noqa: E402

ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_PIXELS = 40_000_000
KEEP_JOBS = 100


def read_upload(storage, what: str = "image") -> Image.Image:
    data = storage.read() if storage else b""
    if not data:
        raise ValueError(f"No {what} file was received.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            if probe.width * probe.height > MAX_PIXELS:
                raise ValueError(f"The {what} is too large ({probe.width}x{probe.height}); keep it under 40 megapixels.")
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported {what} format {fmt}; upload a JPG or PNG.")
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"The {what} could not be read as a JPG or PNG file.") from exc


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class JobQueue:
    """Single GPU worker; jobs and results are kept in memory only."""

    def __init__(self, registry: Registry, examples=None):
        self.registry, self.examples = registry, examples
        self.jobs: OrderedDict[str, dict] = OrderedDict()
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.wakeup = threading.Condition(self.lock)
        threading.Thread(target=self._loop, name="restoration-worker", daemon=True).start()

    def submit(self, job: dict) -> dict:
        job.update(id=uuid.uuid4().hex[:12], status="queued", created_at=time.time(), started_at=None,
                   finished_at=None, result=None, metrics=None, error=None)
        with self.lock:
            self.jobs[job["id"]] = job
            self.order.append(job["id"])
            while len(self.jobs) > KEEP_JOBS:  # drop the oldest finished job
                oldest = next((k for k, v in self.jobs.items() if v["status"] in ("done", "error")), None)
                if oldest is None:
                    break
                self.jobs.pop(oldest)
            self.wakeup.notify()
        return job

    def position(self, job_id: str) -> int:
        with self.lock:
            queued = [j for j in self.order if self.jobs.get(j, {}).get("status") == "queued"]
        return queued.index(job_id) + 1 if job_id in queued else 0

    def pending(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] in ("queued", "running"))

    def _loop(self):
        while True:
            with self.lock:
                while not self.order:
                    self.wakeup.wait()
                job = self.jobs.get(self.order.pop(0))
                if job is None:
                    continue
                job["status"], job["started_at"] = "running", time.time()
            result = self.registry.restore(job.pop("_image"), job["task"], job.pop("_mask", None),
                                           seed=job["seed"], method=job["method"])
            job["result"] = result.metadata()
            if result.success:
                job["_png"] = png_bytes(result.image)
                if job.get("example_id") and self.examples is not None:
                    try:
                        job["metrics"] = self.examples.metrics(job["example_id"], job["task"], result.image,
                                                               job.get("_example_mask"))
                    except Exception as exc:  # noqa: BLE001 - metrics are optional
                        job["metrics"] = {"note": f"metrics unavailable: {exc}"}
            job["error"] = result.error
            job["status"] = "done" if result.success else "error"
            job["finished_at"] = time.time()


def public(job: dict, queue: JobQueue) -> dict:
    data = {k: v for k, v in job.items() if not k.startswith("_")}
    data["queue_position"] = queue.position(job["id"]) if job["status"] == "queued" else 0
    if job["status"] == "done":
        data["output_url"] = f"api/jobs/{job['id']}/output.png"
    return data


def create_app(registry: Registry, examples=None, max_upload_mb: int = 25) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    queue = JobQueue(registry, examples)
    app.extensions["plantsr_queue"] = queue

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        info = {"status": "ok", "pending_jobs": queue.pending(), "device": "cpu", "gpu": None,
                "examples": len(examples) if examples is not None else 0}
        try:
            import torch

            if torch.cuda.is_available():
                info.update(device="cuda", gpu=torch.cuda.get_device_name(0),
                            gpu_memory_allocated_gb=round(torch.cuda.memory_allocated() / 1024 ** 3, 2))
        except ImportError:
            pass
        info["loaded_models"] = [f"{t['task']}/{m['method']}" for t in registry.describe()
                                 for m in t["methods"] if m["uses_gpu"] and m["loaded"]]
        return jsonify(info)

    @app.get("/api/tasks")
    def tasks():
        return jsonify(registry.describe())

    @app.get("/api/examples")
    def example_list():
        return jsonify(examples.list() if examples is not None else [])

    @app.get("/api/examples/<sample_id>/<kind>.png")
    def example_file(sample_id, kind):
        if examples is None:
            return jsonify(error="No benchmark examples are configured."), 404
        task = request.args.get("task", "inpainting")
        try:
            image, mask = examples.degraded(sample_id, task)
        except (KeyError, ValueError) as exc:
            return jsonify(error=str(exc)), 404
        if kind == "input":
            payload = png_bytes(image)
        elif kind == "mask" and mask is not None:
            payload = png_bytes(Image.fromarray(mask * 255))
        else:
            return jsonify(error="Unknown file."), 404
        return send_file(io.BytesIO(payload), mimetype="image/png")

    @app.post("/api/jobs")
    def submit():
        form = request.form
        task = form.get("task", "")
        try:
            if task not in TASKS:
                raise ValueError(f"Choose a task: {', '.join(TASKS)}.")
            handler = registry.handler(task, form.get("method") or None)
            try:
                seed = int(form.get("seed", 42))
            except ValueError as exc:
                raise ValueError("The seed must be a whole number.") from exc
            job = {"task": task, "method": handler.method, "seed": seed, "source": form.get("source", "upload")}
            if job["source"] == "example":
                if examples is None:
                    raise ValueError("Benchmark examples are not available on this server.")
                job["example_id"] = form.get("example_id", "")
                image, mask = examples.degraded(job["example_id"], task)
                job["_example_mask"] = mask
            else:
                image = read_upload(request.files.get("image"))
                mask = None
                if handler.needs_mask:
                    if not request.files.get("mask"):
                        raise ValueError("Paint or upload a mask first (white = area to fill).")
                    mask_image = read_upload(request.files.get("mask"), "mask")
                    if mask_image.size != image.size:
                        job["mask_resized"] = True
                    mask = normalize_mask(mask_image, image.size)
            job["input_size"] = list(image.size)
            job["_image"], job["_mask"] = image, mask
        except (ValueError, KeyError) as exc:
            return jsonify(error=str(exc).strip("'\"")), 400
        job = queue.submit(job)
        return jsonify(public(job, queue)), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = queue.jobs.get(job_id)
        if job is None:
            return jsonify(error="Unknown or expired job."), 404
        return jsonify(public(job, queue))

    @app.get("/api/jobs/<job_id>/output.png")
    def job_output(job_id):
        job = queue.jobs.get(job_id)
        if job is None or "_png" not in job:
            return jsonify(error="No output for this job."), 404
        name = f"plantsr_{job['task']}_{job['method']}_{job_id}.png"
        return send_file(io.BytesIO(job["_png"]), mimetype="image/png",
                         as_attachment=request.args.get("download") == "1", download_name=name)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"Upload is larger than {max_upload_mb} MB."), 413

    return app


def main(argv=None) -> None:
    from src.inference.factory import build_registry, load_config

    parser = argparse.ArgumentParser(description="PlantSR restoration web app")
    parser.add_argument("--config", default=str(ROOT / "configs" / "app.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--prepared-dir", help="PlantSR prepared/ folder for benchmark examples")
    parser.add_argument("--inpaint-manifest", help="Fixed inpainting manifest (val) for example masks")
    parser.add_argument("--no-diffusion", action="store_true", help="Baselines only (no GPU models)")
    args = parser.parse_args(argv)

    registry = build_registry(load_config(args.config), with_diffusion=not args.no_diffusion)
    examples = None
    if args.prepared_dir:
        from src.data.benchmark_examples import BenchmarkExamples

        examples = BenchmarkExamples(args.prepared_dir, "val", args.inpaint_manifest)
    app = create_app(registry, examples)
    try:
        from waitress import serve

        print(f"Serving on http://{args.host}:{args.port} (waitress)")
        serve(app, host=args.host, port=args.port, threads=8)
    except ImportError:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
