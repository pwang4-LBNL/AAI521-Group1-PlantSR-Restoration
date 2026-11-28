"""Put the team's fine-tuned weights where the app expects them, and verify their checksums.

    python scripts/install_checkpoints.py --status
    python scripts/install_checkpoints.py --denoise Denoise.zip --colorize Colorize.zip \
        --sr PlantSR_x4_app.zip --inpaint-lora lora_trial/

Accepted inputs
  --denoise, --colorize   Jinyuan's best.pth, or the Drive ZIP that contains it (Denoise/best.pth)
  --sr                    Peng's PlantSR_x4_app.zip (or its extracted folder); --download-sr fetches it
                          from Peng's Drive link (docs/super_resolution.md) with gdown
  --inpaint-lora          a LoRA folder or ZIP with pytorch_lora_weights.safetensors (+ trial_config.json)

Destinations (all ignored by Git)
  checkpoints/denoise/best.pth, checkpoints/colorize/best.pth,
  plantsr_sr/weights/{hat_l,real_esrgan}.pt + plantsr_sr/manifest.json, checkpoints/inpaint/lora/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR_DRIVE_ID = "1j4plR5Zc-MUNYD42HYTR-lCmgLLjZ-VA"  # Peng's PlantSR_x4_app.zip (docs/super_resolution.md)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config() -> dict:
    sys.path.insert(0, str(ROOT))
    from src.inference.factory import load_config

    return load_config(ROOT / "configs" / "app.json")


def find_member(source: Path, suffix: str, workdir: Path) -> Path:
    """Return a file ending with `suffix` from a file, folder or ZIP."""
    if source.is_file() and source.suffix.lower() != ".zip":
        return source
    if source.is_dir():
        hits = sorted(p for p in source.rglob("*") if p.as_posix().endswith(suffix))
    else:
        with zipfile.ZipFile(source) as archive:
            names = sorted(n for n in archive.namelist() if n.endswith(suffix) and "__MACOSX" not in n)
            if not names:
                raise FileNotFoundError(f"No *{suffix} inside {source}")
            archive.extract(names[0], workdir)
            hits = [workdir / names[0]]
    if not hits:
        raise FileNotFoundError(f"No *{suffix} found in {source}")
    return hits[0]


def install_file(src: Path, dest: Path, expected: str | None, label: str, force: bool) -> None:
    digest = sha256(src)
    if expected and digest != expected:
        message = f"{label}: SHA-256 {digest[:12]}… does not match configs/app.json ({expected[:12]}…)."
        if not force:
            raise SystemExit(message + " Use --force to install it anyway and update the config.")
        print("WARNING:", message)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"{label:16s} -> {dest.relative_to(ROOT)}  sha256 {digest[:12]}…  ({dest.stat().st_size / 2**20:.0f} MB)")


def install_sr(source: Path, workdir: Path) -> None:
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            archive.extractall(workdir / "sr")
        source = workdir / "sr"
    package = next((p.parent for p in source.rglob("inference.py") if p.parent.name == "plantsr_sr"), None)
    if package is None:
        raise FileNotFoundError(f"No plantsr_sr/ package found in {source}")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    target = ROOT / "plantsr_sr"
    (target / "weights").mkdir(parents=True, exist_ok=True)
    for method, info in manifest["models"].items():
        src = package / info["file"]
        digest = sha256(src)
        if digest != info["sha256"]:
            raise SystemExit(f"SR {method}: SHA-256 differs from the bundle manifest; the ZIP may be corrupted.")
        shutil.copy2(src, target / info["file"])
        print(f"SR {method:12s} -> plantsr_sr/{info['file']}  epoch {info['epoch']}  sha256 {digest[:12]}…")
    shutil.copy2(package / "manifest.json", target / "manifest.json")
    for name in ("models.py", "inference.py"):  # the repo copy must be Peng's exported code
        same = (package / name).read_bytes().replace(b"\r\n", b"\n") == (target / name).read_bytes().replace(b"\r\n", b"\n")
        print(f"plantsr_sr/{name}: {'identical to the bundle' if same else 'DIFFERS from the bundle - tell Peng/Ali'}")


def install_lora(source: Path, workdir: Path) -> None:
    weights = find_member(source, "pytorch_lora_weights.safetensors", workdir)
    target = ROOT / "checkpoints" / "inpaint" / "lora"
    target.mkdir(parents=True, exist_ok=True)
    for path in weights.parent.iterdir():
        if path.suffix in {".safetensors", ".json"}:
            shutil.copy2(path, target / path.name)
    print(f"Inpainting LoRA  -> {target.relative_to(ROOT)}  sha256 {sha256(target / weights.name)[:12]}…")


def download_sr(target: Path) -> Path | None:
    """Fetch Peng's bundle; if the Drive link is not public, explain and carry on without it."""
    try:
        import gdown

        gdown.download(id=SR_DRIVE_ID, output=str(target), quiet=False)
        if target.is_file() and zipfile.is_zipfile(target):
            return target
        problem = "the download did not produce a ZIP file"
    except Exception as exc:  # noqa: BLE001 - gdown raises several types for permission problems
        problem = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"
    print("WARNING: could not download Peng's PlantSR_x4_app.zip "
          f"({problem}).\n  The Drive link is probably not shared as 'Anyone with the link'. Download it in a "
          "browser from docs/super_resolution.md and pass it with --sr. Continuing without the fine-tuned "
          "super-resolution models.")
    return None


def status() -> int:
    sys.path.insert(0, str(ROOT))
    from src.inference.factory import build_registry

    registry = build_registry(config(), with_diffusion=True)
    missing = 0
    for task in registry.describe():
        for method in task["methods"]:
            star = "*" if method["method"] == task["default_method"] else " "
            print(f"{star} {task['task']:17s} {method['method']:22s} {method['kind']}")
        for item in task["unavailable"]:
            missing += 1
            print(f"! {task['task']:17s} {item['method']:22s} {item['reason']}")
    print("(* = default method; ! = not installed)")
    return missing


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--denoise", type=Path)
    parser.add_argument("--colorize", type=Path)
    parser.add_argument("--sr", type=Path)
    parser.add_argument("--download-sr", action="store_true", help="Download Peng's bundle from Google Drive")
    parser.add_argument("--inpaint-lora", type=Path)
    parser.add_argument("--force", action="store_true", help="Install even if a checksum differs")
    parser.add_argument("--status", action="store_true", help="Only report what is installed")
    args = parser.parse_args(argv)
    if not args.status:
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            if args.denoise:
                install_file(find_member(args.denoise, "best.pth", work / "dn"), ROOT / cfg["denoising"]["checkpoint"],
                             cfg["denoising"].get("sha256"), "Denoising", args.force)
            if args.colorize:
                install_file(find_member(args.colorize, "best.pth", work / "cz"), ROOT / cfg["colorization"]["checkpoint"],
                             cfg["colorization"].get("sha256"), "Colorization", args.force)
            if args.download_sr and not args.sr:
                args.sr = download_sr(work / "PlantSR_x4_app.zip")
            if args.sr:
                install_sr(args.sr, work)
            if args.inpaint_lora:
                install_lora(args.inpaint_lora, work)
    print()
    status()


if __name__ == "__main__":
    main()
