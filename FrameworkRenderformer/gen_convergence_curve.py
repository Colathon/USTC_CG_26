"""Generate PSNR convergence curve from saved checkpoints.
Evaluates a subset of test samples for speed. Saves plot to Results/.
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("ATTN_IMPL", "sdpa")

from baseline_data import PtSceneDataset, renderformer_baseline_collate
from baseline_model import CourseRenderFormerWrapper, build_baseline_config
from torch.utils.data import DataLoader, Subset


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.clamp(pred.float(), 0.0, 1.0)
    target = torch.clamp(target.float(), 0.0, 1.0)
    pred = torch.pow(pred, 1.0 / 2.2)
    target = torch.pow(target, 1.0 / 2.2)
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0


def eval_checkpoint(ckpt_path: Path, dataloader, device) -> float:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]

    config = build_baseline_config(
        latent_dim=saved_args["latent_dim"],
        num_layers=saved_args["num_layers"],
        num_heads=saved_args["num_heads"],
        view_layers=saved_args["view_layers"],
        view_num_heads=saved_args["view_num_heads"],
        use_dpt_decoder=saved_args["use_dpt_decoder"],
        num_register_tokens=saved_args.get("num_register_tokens", 4),
        patch_size=saved_args["patch_size"],
        texture_patch_size=saved_args["texture_patch_size"],
        vertex_pe_num_freqs=saved_args.get("vertex_pe_num_freqs", 6),
        vn_pe_num_freqs=saved_args.get("vn_pe_num_freqs", 6),
        use_vn_encoder=not saved_args.get("no_vn", False),
        ffn_opt=saved_args.get("ffn_opt", "checkpoint"),
        use_object_emb=saved_args.get("use_object_emb", False),
        object_emb_max_objects=saved_args.get("object_emb_max_objects", 64),
    )
    model = CourseRenderFormerWrapper(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    psnr_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            target = batch["gt_image"].float()
            pred = model(batch).float()
            for b in range(pred.shape[0]):
                psnr_list.append(compute_psnr(pred[b], target[b]))

    return sum(psnr_list) / len(psnr_list)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = SCRIPT_DIR / "traindata/traindata/pt_dataset_datalow"
    dataset = PtSceneDataset(str(data_path))
    # use first 10 samples for speed
    n_eval = min(10, len(dataset))
    subset = Subset(dataset, list(range(n_eval)))
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=renderformer_baseline_collate)
    print(f"Evaluating on {n_eval} samples\n")

    runs_dir = SCRIPT_DIR / "runs"

    # --- define what to evaluate ---
    configs = {
        "Baseline": {
            "checkpoints": sorted((runs_dir / "full/checkpoints").glob("step_*.pt")),
            "color": "#2196F3",
            "lw": 2.5,
        },
        "w/o DPT": {
            "checkpoints": sorted((runs_dir / "ablation_no_dpt/checkpoints").glob("step_*.pt")),
            "color": "#F44336",
            "lw": 1.5,
        },
        "w/o Vertex Normal": {
            "checkpoints": sorted((runs_dir / "ablation_no_vn/checkpoints").glob("step_*.pt")),
            "color": "#FF9800",
            "lw": 1.5,
        },
        "L1 Loss": {
            "checkpoints": sorted((runs_dir / "ablation_loss_l1/checkpoints").glob("step_*.pt")),
            "color": "#9C27B0",
            "lw": 1.5,
        },
    }

    results = {}
    for name, cfg in configs.items():
        ckpts = cfg["checkpoints"]
        if not ckpts:
            print(f"  [SKIP] {name}: no checkpoints found")
            continue
        steps, psnrs = [], []
        for ckpt in ckpts:
            step = int(ckpt.stem.split("_")[1])
            print(f"  Evaluating {name} @ step {step} ...")
            psnr = eval_checkpoint(ckpt, dataloader, device)
            steps.append(step)
            psnrs.append(psnr)
            print(f"    PSNR = {psnr:.2f} dB")
        results[name] = (steps, psnrs, cfg["color"], cfg["lw"])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, (steps, psnrs, color, lw) in results.items():
        ax.plot(steps, psnrs, marker="o", markersize=5, color=color, lw=lw, label=name)

    ax.axhline(y=15.0, color="gray", linestyle="--", lw=1, label="Required threshold (15 dB)")
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Training Convergence: PSNR vs. Training Steps", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    out_path = SCRIPT_DIR / "Results/convergence_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()