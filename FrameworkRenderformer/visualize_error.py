"""
Error analysis and visualization for trained RenderFormer model.
Outputs per-sample comparison images (GT | Prediction | Error heatmap)
and a summary grid with PSNR values.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

os.environ.setdefault("ATTN_IMPL", "sdpa")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_data import PtSceneDataset, renderformer_baseline_collate
from baseline_model import CourseRenderFormerWrapper, build_baseline_config


# ── helpers ──────────────────────────────────────────────────────────────────

def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [3, H, W] float tensor to uint8 HWC array (gamma corrected)."""
    img = tensor.detach().cpu().float()
    img = torch.clamp(img, 0.0, 1.0)
    img = torch.pow(img, 1.0 / 2.2)
    return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """Apply jet colormap to a [H, W] float array in [0, 1]. Returns [H, W, 3] uint8."""
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def error_heatmap(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Compute absolute error heatmap [H, W, 3] uint8 in gamma-corrected space."""
    p = torch.clamp(pred.float(), 0.0, 1.0).pow(1.0 / 2.2)
    t = torch.clamp(target.float(), 0.0, 1.0).pow(1.0 / 2.2)
    err = (p - t).abs().mean(dim=0).cpu().numpy()  # [H, W]
    # normalize to [0, 1] using 95th percentile for robustness
    vmax = float(np.percentile(err, 95)) + 1e-6
    err_norm = np.clip(err / vmax, 0.0, 1.0)
    return jet_colormap(err_norm)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = torch.clamp(pred.float(), 0.0, 1.0).pow(1.0 / 2.2)
    t = torch.clamp(target.float(), 0.0, 1.0).pow(1.0 / 2.2)
    mse = F.mse_loss(p, t).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def add_label(img_arr: np.ndarray, text: str) -> np.ndarray:
    """Draw a small label at the top-left of an HWC uint8 array."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, len(text) * 7 + 4, 14], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 255))
    return np.array(img)


def make_comparison(gt_arr, pred_arr, err_arr, psnr: float, idx: int) -> np.ndarray:
    """Stack GT | Pred | Error side by side with labels."""
    gap = np.zeros((gt_arr.shape[0], 4, 3), dtype=np.uint8)
    gt_l   = add_label(gt_arr,   "GT")
    pred_l = add_label(pred_arr, f"Pred  PSNR={psnr:.1f}dB")
    err_l  = add_label(err_arr,  "Error (jet)")
    row = np.concatenate([gt_l, gap, pred_l, gap, err_l], axis=1)
    return row


def load_model(checkpoint_path: str, device: torch.device) -> CourseRenderFormerWrapper:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    config = build_baseline_config(
        latent_dim=a["latent_dim"],
        num_layers=a["num_layers"],
        num_heads=a["num_heads"],
        view_layers=a["view_layers"],
        view_num_heads=a["view_num_heads"],
        use_dpt_decoder=a["use_dpt_decoder"],
        num_register_tokens=a.get("num_register_tokens", 4),
        patch_size=a["patch_size"],
        texture_patch_size=a["texture_patch_size"],
        vertex_pe_num_freqs=a.get("vertex_pe_num_freqs", 6),
        vn_pe_num_freqs=a.get("vn_pe_num_freqs", 6),
        use_vn_encoder=not a.get("no_vn", False),
        ffn_opt=a.get("ffn_opt", "checkpoint"),
        use_object_emb=a.get("use_object_emb", False),
        object_emb_max_objects=a.get("object_emb_max_objects", 64),
    )
    model = CourseRenderFormerWrapper(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", default="runs/error_analysis")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  Output: {out_dir}")

    model = load_model(args.checkpoint, device)

    dataset = PtSceneDataset(args.data_path)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=renderformer_baseline_collate)

    psnr_list = []
    thumb_list = []          # collect small thumbnails for the summary grid
    THUMB = 64               # thumbnail size for summary

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            target = batch["gt_image"].float()   # [1, 3, H, W]
            pred   = model(batch).float()        # [1, 3, H, W]

            for b in range(pred.shape[0]):
                p, t = pred[b], target[b]
                psnr = compute_psnr(p, t)
                psnr_list.append(psnr)

                gt_arr   = to_uint8(t)
                pred_arr = to_uint8(p)
                err_arr  = error_heatmap(p, t)

                # ── per-sample comparison image ──────────────────────────
                row = make_comparison(gt_arr, pred_arr, err_arr, psnr, i)
                Image.fromarray(row).save(out_dir / f"sample_{i:03d}_compare.png")

                # ── thumbnail of error for summary ───────────────────────
                thumb = Image.fromarray(err_arr).resize((THUMB, THUMB), Image.BILINEAR)
                thumb_list.append((i, psnr, np.array(thumb)))

                print(f"  [{i:03d}] PSNR = {psnr:.2f} dB  →  sample_{i:03d}_compare.png")

    # ── summary grid ─────────────────────────────────────────────────────────
    n = len(thumb_list)
    cols = min(10, n)
    rows = math.ceil(n / cols)
    cell_h = THUMB + 18   # extra space for PSNR text
    cell_w = THUMB + 2
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for idx, (sample_i, psnr, thumb_arr) in enumerate(thumb_list):
        col = idx % cols
        row = idx // cols
        x = col * cell_w
        y = row * cell_h
        grid.paste(Image.fromarray(thumb_arr), (x, y))
        label = f"#{sample_i} {psnr:.1f}"
        draw.text((x + 1, y + THUMB + 2), label, fill=(220, 220, 100))

    grid.save(out_dir / "summary_grid.png")

    # ── PSNR bar chart (pure numpy → PNG) ────────────────────────────────────
    bar_w, bar_h = 20, 200
    margin = 40
    chart_w = margin + n * (bar_w + 4) + margin
    chart_h = bar_h + margin * 2 + 20
    chart = np.full((chart_h, chart_w, 3), 30, dtype=np.uint8)

    max_psnr = max(psnr_list)
    mean_psnr = sum(psnr_list) / n

    for idx, psnr in enumerate(psnr_list):
        h = int(psnr / max_psnr * bar_h)
        x0 = margin + idx * (bar_w + 4)
        x1 = x0 + bar_w
        y1 = margin + bar_h
        y0 = y1 - h
        # color: green if above mean, orange if below
        color = (80, 200, 80) if psnr >= mean_psnr else (220, 130, 50)
        chart[y0:y1, x0:x1] = color

    # draw mean line
    mean_y = margin + bar_h - int(mean_psnr / max_psnr * bar_h)
    chart[mean_y, margin:margin + n * (bar_w + 4)] = (255, 80, 80)

    img_chart = Image.fromarray(chart)
    d = ImageDraw.Draw(img_chart)
    d.text((margin, 5), f"Per-sample PSNR  (mean={mean_psnr:.2f} dB, red line)", fill=(220, 220, 220))
    d.text((margin, chart_h - 18), f"Min={min(psnr_list):.2f}  Max={max(psnr_list):.2f}  Mean={mean_psnr:.2f}", fill=(180, 180, 180))
    img_chart.save(out_dir / "psnr_chart.png")

    print(f"\n{'='*45}")
    print(f"  Mean PSNR : {mean_psnr:.2f} dB")
    print(f"  Min  PSNR : {min(psnr_list):.2f} dB")
    print(f"  Max  PSNR : {max(psnr_list):.2f} dB")
    print(f"  Samples   : {n}")
    print(f"  Output    : {out_dir.resolve()}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
