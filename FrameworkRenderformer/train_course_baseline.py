from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_data import H5TriangleDataset, PtSceneDataset, renderformer_baseline_collate
from baseline_loss import SimpleRenderFormerLoss
from baseline_model import CourseRenderFormerWrapper, build_baseline_config, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact RenderFormer baseline for course assignments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset_format", choices=["pt", "h5"], required=True, help="Input dataset format.")
    parser.add_argument("--data_path", type=str, required=True, help="PT directory or H5 file/directory.")
    parser.add_argument("--out_dir", type=str, default="course_renderformer_baseline/runs/default")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None, help="Optional cap on PT samples for debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    parser.add_argument("--image_size", type=int, default=64, help="Only used for H5 data.")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--vis_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--view_layers", type=int, default=4)
    parser.add_argument("--view_num_heads", type=int, default=4)
    parser.add_argument("--use_dpt_decoder", action="store_true")
    parser.add_argument("--num_register_tokens", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--texture_patch_size", type=int, default=8)
    parser.add_argument("--vertex_pe_num_freqs", type=int, default=6)
    parser.add_argument("--vn_pe_num_freqs", type=int, default=6)
    parser.add_argument("--no_vn", action="store_true")
    parser.add_argument("--ffn_opt", choices=["checkpoint", "none"], default="checkpoint")
    parser.add_argument("--use_object_emb", action="store_true", help="Add per-object segment embeddings (object-level encoding).")
    parser.add_argument("--object_emb_max_objects", type=int, default=64, help="Embedding table size for object IDs.")

    parser.add_argument("--loss_type", choices=["log_l1", "l1", "mse"], default="log_l1")
    parser.add_argument("--use_lpips", action="store_true")
    parser.add_argument("--lpips_weight", type=float, default=0.05)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_mode(device: torch.device, amp_arg: str) -> Tuple[torch.dtype | None, bool]:
    if device.type != "cuda" or amp_arg == "none":
        return None, False
    if amp_arg == "bf16":
        return torch.bfloat16, False
    if amp_arg == "fp16":
        return torch.float16, True
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(sub_value, device) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def build_dataset(args: argparse.Namespace):
    if args.dataset_format == "pt":
        return PtSceneDataset(args.data_path, max_items=args.max_items)
    return H5TriangleDataset(args.data_path, render_resolution=args.image_size)


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().to(device="cpu", dtype=torch.float32)
    image = torch.clamp(image, 0.0, 1.0)
    image = torch.pow(image, 1.0 / 2.2)
    image = image.permute(1, 2, 0).numpy()
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def save_preview(prediction: torch.Tensor, target: torch.Tensor, save_path: Path) -> None:
    pred_image = tensor_to_uint8_image(prediction[0].detach())
    target_image = tensor_to_uint8_image(target[0].detach())
    canvas = np.concatenate([target_image, pred_image], axis=1)
    Image.fromarray(canvas).save(save_path)


def save_json(save_path: Path, content: Dict[str, Any]) -> None:
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2, ensure_ascii=True)


def next_batch(data_iterator, dataloader):
    try:
        batch = next(data_iterator)
        return batch, data_iterator
    except StopIteration:
        data_iterator = iter(dataloader)
        batch = next(data_iterator)
        return batch, data_iterator


def ensure_resolution_is_valid(batch: Dict[str, Any], patch_size: int) -> None:
    _, _, height, width = batch["gt_image"].shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image resolution {(height, width)} must be divisible by patch_size={patch_size}."
        )


def maybe_load_checkpoint(
    resume_from: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    if not resume_from:
        return 0

    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("step", 0))


def save_checkpoint(
    save_path: Path,
    step: int,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    torch.save(
        {
            "step": step,
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "args.json", vars(args))

    device = pick_device(args.device)
    amp_dtype, use_grad_scaler = resolve_amp_mode(device, args.amp)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=renderformer_baseline_collate,
    )

    config = build_baseline_config(
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        view_layers=args.view_layers,
        view_num_heads=args.view_num_heads,
        use_dpt_decoder=args.use_dpt_decoder,
        num_register_tokens=args.num_register_tokens,
        patch_size=args.patch_size,
        texture_patch_size=args.texture_patch_size,
        vertex_pe_num_freqs=args.vertex_pe_num_freqs,
        vn_pe_num_freqs=args.vn_pe_num_freqs,
        use_vn_encoder=not args.no_vn,
        ffn_opt=args.ffn_opt,
        use_object_emb=args.use_object_emb,
        object_emb_max_objects=args.object_emb_max_objects,
    )
    model = CourseRenderFormerWrapper(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    criterion = SimpleRenderFormerLoss(
        loss_type=args.loss_type,
        use_lpips=args.use_lpips,
        lpips_weight=args.lpips_weight,
        device=str(device),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_grad_scaler))

    start_step = maybe_load_checkpoint(args.resume_from, model, optimizer, scheduler, device)

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"AMP mode: {amp_dtype if amp_dtype is not None else 'disabled'}")
    print("ATTN_IMPL: sdpa")

    data_iterator = iter(dataloader)
    start_time = time.time()
    global_step = start_step

    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        model.train()

        step_metrics = {
            "total_loss": 0.0,
            "base_loss": 0.0,
            "lpips_loss": 0.0,
        }
        latest_prediction = None
        latest_target = None

        for _ in range(args.grad_accum_steps):
            batch, data_iterator = next_batch(data_iterator, dataloader)
            batch = move_to_device(batch, device)
            ensure_resolution_is_valid(batch, args.patch_size)
            target = batch["gt_image"].to(dtype=torch.float32)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype is not None
                else nullcontext()
            )

            with autocast_context:
                prediction = model(batch)
                loss, metrics = criterion(prediction, target)

            scaled_loss = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            for key in step_metrics:
                step_metrics[key] += float(metrics[key].item())

            latest_prediction = prediction.detach()
            latest_target = target.detach()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        global_step += 1

        for key in step_metrics:
            step_metrics[key] /= args.grad_accum_steps

        if global_step == start_step + 1:
            batch_shapes = {
                "tri_pos": tuple(batch["tri_pos"].shape),
                "gt_image": tuple(batch["gt_image"].shape),
                "ray_map": tuple(batch["ray_map"].shape),
            }
            print(f"First batch shapes: {batch_shapes}")

        if global_step % args.log_every == 0 or global_step == 1:
            elapsed = time.time() - start_time
            peak_memory = 0.0
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"[Step {global_step:05d}/{args.max_steps}] "
                f"loss={step_metrics['total_loss']:.6f} "
                f"base={step_metrics['base_loss']:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"elapsed={elapsed:.1f}s "
                f"peak_mem={peak_memory:.2f}GB"
            )

        if latest_prediction is not None and (global_step % args.vis_every == 0 or global_step <= 30):
            save_preview(latest_prediction, latest_target, vis_dir / f"step_{global_step:05d}.png")

        if global_step % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"step_{global_step:05d}.pt", global_step, args, model, optimizer, scheduler)
            save_checkpoint(ckpt_dir / "last.pt", global_step, args, model, optimizer, scheduler)

    save_checkpoint(ckpt_dir / "final.pt", global_step, args, model, optimizer, scheduler)
    save_checkpoint(ckpt_dir / "last.pt", global_step, args, model, optimizer, scheduler)
    print(f"Training finished in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
