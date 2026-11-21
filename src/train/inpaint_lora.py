"""LoRA fine-tuning of Stable Diffusion 1.5 inpainting on PlantSR crops.

Week 1 uses `train_lora_trial` for a short 50-100 step run that checks the training path:
finite losses, parameter updates, GPU memory/speed, saving, and reloading in a new process.
The same loop is the starting point for the full Week 2 fine-tuning.

Training sample: random 256 x 256 crop of a full-size training image -> random horizontal
flip -> random mask (5-20%, rectangle or free-form) drawn *after* the geometry, so image
and mask are aligned by construction -> both resized to the working resolution
(bicubic image, nearest mask). Loss: standard epsilon-prediction MSE on the 9-channel UNet.

Fresh-process reload check (run by the notebook):
    python -m src.train.inpaint_lora reload-check --lora DIR --cases cases.json --out out.npz
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.inpaint_masks import TrainingMaskSampler, load_mask_png
from src.inference.sd_inpaint import (DEFAULT_MODEL_ID, DEFAULT_NEGATIVE, DEFAULT_PROMPT, StableDiffusionInpaint,
                                      load_inpaint_pipeline, lora_file)


@dataclass
class LoraTrialConfig:
    model_id: str = DEFAULT_MODEL_ID
    steps: int = 100
    learning_rate: float = 1e-4
    rank: int = 8
    alpha: int = 8  # alpha == rank -> LoRA scale 1.0
    batch_size: int = 1
    crop_size: int = 256
    work_res: int = 512
    prompt: str = DEFAULT_PROMPT
    seed: int = 42
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    target_modules: tuple = ("to_q", "to_k", "to_v", "to_out.0")
    log_every: int = 10


@dataclass
class TrialResult:
    losses: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)
    step_seconds: list = field(default_factory=list)
    trainable_parameters: int = 0
    trainable_tensors: int = 0
    changed_tensors: int = 0
    max_abs_update: float = 0.0
    only_lora_trainable: bool = False
    peak_gpu_memory_gb: float | None = None
    amp_dtype: str = "float32"
    samples: list = field(default_factory=list)

    @property
    def all_finite(self) -> bool:
        return bool(self.losses) and bool(np.all(np.isfinite(self.losses)))

    def summary(self) -> dict:
        losses = np.asarray(self.losses, dtype=float)
        times = np.asarray(self.step_seconds[1:] or self.step_seconds, dtype=float)
        first = losses[: max(1, len(losses) // 5)].mean() if len(losses) else float("nan")
        last = losses[-max(1, len(losses) // 5):].mean() if len(losses) else float("nan")
        return {
            "steps": len(self.losses), "all_losses_finite": self.all_finite,
            "loss_first_20pct_mean": float(first), "loss_last_20pct_mean": float(last),
            "trainable_parameters": self.trainable_parameters, "trainable_tensors": self.trainable_tensors,
            "changed_tensors": self.changed_tensors, "max_abs_update": self.max_abs_update,
            "only_lora_trainable": self.only_lora_trainable,
            "median_step_seconds": float(np.median(times)) if len(times) else None,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb, "amp_dtype": self.amp_dtype,
        }


class TrainCropDataset:
    """Random crops + flips + masks from the full-size training images (map-style dataset)."""

    def __init__(self, prepared_dir, rows, crop_size: int = 256, work_res: int = 512, seed: int = 42):
        self.prepared_dir = Path(prepared_dir)
        self.rows = [r for r in rows if r.get("hr_path")]
        if not self.rows:
            raise ValueError("No training rows with hr_path were provided.")
        self.crop_size, self.work_res = crop_size, work_res
        self.rng = np.random.default_rng(seed)
        self.masks = TrainingMaskSampler(seed + 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        with Image.open(self.prepared_dir / row["hr_path"]) as src:
            image = src.convert("RGB")
        c = self.crop_size
        if min(image.size) < c:
            raise ValueError(f"{row['sample_id']} is smaller than the crop size")
        x = int(self.rng.integers(0, image.width - c + 1))
        y = int(self.rng.integers(0, image.height - c + 1))
        crop = image.crop((x, y, x + c, y + c))
        flip = bool(self.rng.random() < 0.5)
        if flip:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        mask, params = self.masks(c, c)  # drawn after crop/flip -> always aligned
        work = self.work_res
        crop_w = crop.resize((work, work), Image.BICUBIC) if work != c else crop
        mask_w = Image.fromarray(mask * 255).resize((work, work), Image.NEAREST) if work != c else Image.fromarray(mask * 255)
        pixel = torch.from_numpy(np.asarray(crop_w, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        mask_t = torch.from_numpy((np.asarray(mask_w) > 127).astype(np.float32))[None]
        return {"pixel_values": pixel, "mask": mask_t, "sample_id": row["sample_id"], "crop_x": x, "crop_y": y,
                "flip": flip, "mask_type": params["mask_type"], "coverage": params["coverage"]}


def epoch_batches(loader):
    """Yield batches forever, starting a fresh pass over `loader` each epoch.

    Unlike itertools.cycle, which replays the cached first epoch, every new pass reshuffles and
    draws new random crops, flips and masks from the dataset.
    """
    while True:
        for batch in loader:
            yield batch


def _collate(items):
    import torch

    batch = {"pixel_values": torch.stack([i["pixel_values"] for i in items]),
             "mask": torch.stack([i["mask"] for i in items])}
    batch["meta"] = [{k: v for k, v in i.items() if k not in ("pixel_values", "mask")} for i in items]
    return batch


def add_lora(unet, cfg: LoraTrialConfig):
    import torch
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, init_lora_weights="gaussian",
                                target_modules=list(cfg.target_modules)))
    if unet.dtype != torch.float32:
        cast_training_params(unet, dtype=torch.float32)  # train adapters in fp32, base stays fp16
    return [p for p in unet.parameters() if p.requires_grad]


def train_lora_trial(pipe, dataset: TrainCropDataset, cfg: LoraTrialConfig, progress=print) -> TrialResult:
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler

    device = pipe.unet.device
    torch.manual_seed(cfg.seed)
    unet, vae = pipe.unet, pipe.vae
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    params = add_lora(unet, cfg)
    names = [n for n, p in unet.named_parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in unet.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=1e-2)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    prediction_type = noise_scheduler.config.prediction_type
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(cfg.prompt, device, 1, False)

    use_amp = device.type == "cuda"
    amp_dtype = torch.float32
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()

    generator = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                                         collate_fn=_collate, generator=generator, drop_last=True)
    batches = epoch_batches(loader)
    sf = vae.config.scaling_factor
    result = TrialResult(trainable_parameters=sum(p.numel() for p in params), trainable_tensors=len(params),
                         amp_dtype=str(amp_dtype).replace("torch.", ""),
                         only_lora_trainable=bool(names) and all("lora" in n for n in names))
    for step in range(1, cfg.steps + 1):
        start = time.perf_counter()
        batch = next(batches)
        pixel = batch["pixel_values"].to(device, dtype=vae.dtype)
        mask = batch["mask"].to(device, dtype=vae.dtype)
        masked = pixel * (mask < 0.5)  # same masking as the pipeline: missing pixels -> 0 (grey)
        with torch.no_grad():
            latents = vae.encode(pixel).latent_dist.sample() * sf
            masked_latents = vae.encode(masked).latent_dist.sample() * sf
        mask_latent = F.interpolate(mask, size=latents.shape[-2:])
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        model_input = torch.cat([noisy, mask_latent, masked_latents], dim=1)
        target = noise if prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            pred = unet(model_input, timesteps,
                        encoder_hidden_states=prompt_embeds.expand(latents.shape[0], -1, -1)).sample
        loss = F.mse_loss(pred.float(), target.float())
        if not torch.isfinite(loss):
            result.losses.append(float("nan"))
            raise FloatingPointError(f"Non-finite loss at step {step}; lower the learning rate.")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if use_amp:
            torch.cuda.synchronize(device)
        result.losses.append(float(loss.detach()))
        result.grad_norms.append(float(grad_norm))
        result.step_seconds.append(time.perf_counter() - start)
        if step <= 3:
            result.samples.extend(batch["meta"])
        if progress and (step % cfg.log_every == 0 or step == 1):
            progress(f"step {step:4d}/{cfg.steps}  loss {result.losses[-1]:.4f}  "
                     f"grad-norm {result.grad_norms[-1]:.3f}  {result.step_seconds[-1]:.2f}s/step")
    unet.eval()
    if cfg.gradient_checkpointing:
        unet.disable_gradient_checkpointing()
    after = dict(unet.named_parameters())
    deltas = [float((after[n].detach() - before[n]).abs().max()) for n in names]
    result.changed_tensors = int(sum(d > 0 for d in deltas))
    result.max_abs_update = max(deltas) if deltas else 0.0
    if use_amp:
        result.peak_gpu_memory_gb = round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2)
    del optimizer
    return result


def prepare_for_inference(unet, dtype) -> None:
    """Freeze the adapters and cast them to the pipeline dtype (as a fresh reload would)."""
    from diffusers.training_utils import cast_training_params

    cast_training_params(unet, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()


def lora_state_dict(unet, adapter_name: str = "default") -> dict:
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    return convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name=adapter_name))


def save_lora(unet, out_dir, cfg: LoraTrialConfig, extra: dict | None = None, adapter_name: str = "default") -> Path:
    from diffusers import StableDiffusionInpaintPipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = lora_state_dict(unet, adapter_name)
    try:  # newer diffusers also store the LoRA config (rank/alpha) inside the file
        StableDiffusionInpaintPipeline.save_lora_weights(
            out_dir, unet_lora_layers=layers, safe_serialization=True,
            unet_lora_adapter_metadata=unet.peft_config[adapter_name].to_dict())
    except TypeError:
        StableDiffusionInpaintPipeline.save_lora_weights(out_dir, unet_lora_layers=layers, safe_serialization=True)
    record = {"config": asdict(cfg), "environment": environment(), **(extra or {})}
    (out_dir / "trial_config.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return lora_file(out_dir)


def compare_adapter_to_file(unet, lora_dir, adapter_name: str) -> dict:
    """Compare LoRA tensors held by `unet` with the saved safetensors file."""
    from safetensors.torch import load_file

    saved = {k.removeprefix("unet."): v for k, v in load_file(str(lora_file(lora_dir))).items()}
    live = {k.removeprefix("unet."): v.detach().float().cpu() for k, v in lora_state_dict(unet, adapter_name).items()}
    shared = sorted(set(saved) & set(live))
    max_diff = max((float((saved[k].float() - live[k]).abs().max()) for k in shared), default=float("nan"))
    return {"saved_tensors": len(saved), "loaded_tensors": len(live), "matching_keys": len(shared),
            "missing_in_model": len(set(saved) - set(live)), "max_abs_diff": max_diff}


def environment() -> dict:
    import importlib.metadata as md

    info = {"python": platform.python_version()}
    for name in ("torch", "diffusers", "transformers", "peft", "accelerate", "numpy", "Pillow"):
        try:
            info[name] = md.version(name)
        except md.PackageNotFoundError:
            info[name] = None
    try:
        import torch

        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def run_cases(handler: StableDiffusionInpaint, prepared_dir, cases) -> np.ndarray:
    """Generate raw (pre-composite) outputs for a list of {input_path, mask_path, seed} cases."""
    outputs = []
    for case in cases:
        with Image.open(Path(prepared_dir) / case["input_path"]) as src:
            image = src.convert("RGB")
        mask = load_mask_png(Path(prepared_dir) / case["mask_path"])
        outputs.append(np.asarray(handler.generate(image, mask, int(case["seed"]))))
    return np.stack(outputs)


def _reload_check(args) -> None:
    record = json.loads((Path(args.lora) / "trial_config.json").read_text(encoding="utf-8"))
    cfg = record["config"]
    infer = record.get("inference", {})
    start = time.perf_counter()
    pipe = load_inpaint_pipeline(cfg["model_id"])
    pipe.load_lora_weights(args.lora, adapter_name="plantsr")
    load_seconds = time.perf_counter() - start
    tensors = compare_adapter_to_file(pipe.unet, args.lora, "plantsr")
    handler = StableDiffusionInpaint("reload_check", model_id=cfg["model_id"], lora_path=args.lora, pipe=pipe,
                                     work_res=infer.get("work_res", cfg["work_res"]),
                                     prompt=infer.get("prompt", cfg["prompt"]),
                                     negative_prompt=infer.get("negative_prompt", DEFAULT_NEGATIVE),
                                     steps=infer.get("steps", 25), guidance=infer.get("guidance", 5.0))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    outputs = run_cases(handler, args.prepared_dir, cases)
    np.savez_compressed(args.out, outputs=outputs)
    report = {"load_seconds": round(load_seconds, 2), "adapter_tensors": tensors,
              "model_version": handler.version, "cases": len(cases)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("reload-check", help="Reload a saved LoRA in a fresh process and run fixed cases")
    check.add_argument("--lora", required=True)
    check.add_argument("--cases", required=True)
    check.add_argument("--prepared-dir", required=True)
    check.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "reload-check":
        _reload_check(args)


if __name__ == "__main__":
    main()
