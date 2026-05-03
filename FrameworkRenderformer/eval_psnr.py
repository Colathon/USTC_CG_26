"""Evaluate PSNR on the PT dataset using a trained checkpoint."""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("ATTN_IMPL", "sdpa")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_data import PtSceneDataset, renderformer_baseline_collate
from baseline_model import CourseRenderFormerWrapper, build_baseline_config


def move_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in gamma-corrected [0,1] space (same as the vis images)."""
    pred = torch.clamp(pred.float(), 0.0, 1.0)
    target = torch.clamp(target.float(), 0.0, 1.0)
    pred = torch.pow(pred, 1.0 / 2.2)
    target = torch.pow(target, 1.0 / 2.2)
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--data_path", type=str, required=True, help="PT dataset directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    print("Model loaded.")

    dataset = PtSceneDataset(args.data_path)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=renderformer_baseline_collate,
    )
    print(f"Dataset: {len(dataset)} samples\n")

    psnr_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            target = batch["gt_image"].float()
            pred = model(batch).float()

            for b in range(pred.shape[0]):
                psnr = compute_psnr(pred[b], target[b])
                psnr_list.append(psnr)
                print(f"  [{i:03d}] PSNR = {psnr:.2f} dB")

    mean_psnr = sum(psnr_list) / len(psnr_list)
    print(f"\n{'='*40}")
    print(f"  Mean PSNR : {mean_psnr:.2f} dB")
    print(f"  Samples   : {len(psnr_list)}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
