"""
Ablation study runner with resume support.
- Already finished experiments (final.pt exists) are skipped.
- Interrupted experiments (last.pt exists) are resumed automatically.
Run: .venv\Scripts\python.exe run_ablation.py
"""

import subprocess
import sys
from pathlib import Path

PYTHON = str(Path(sys.executable))
DATA   = "traindata/traindata/pt_dataset_datalow"
STEPS  = 5000
DEVICE = "cuda"

BASE = [
    "--dataset_format", "pt",
    "--data_path", DATA,
    "--max_steps", str(STEPS),
    "--batch_size", "4",
    "--latent_dim", "256",
    "--num_layers", "4",
    "--num_heads", "4",
    "--patch_size", "8",
    "--texture_patch_size", "1",
    "--log_every", "100",
    "--vis_every", "1000",
    "--save_every", "1000",   # save every 1000 steps for safety
    "--workers", "0",
    "--device", DEVICE,
    "--seed", "42",
]

ABLATIONS = [
    {
        "name": "no_dpt",
        "desc": "No DPT decoder",
        "extra": [],
        "view_layers": "4",
        "use_dpt": False,
    },
    {
        "name": "no_vn",
        "desc": "No vertex normal encoder",
        "extra": ["--no_vn"],
        "view_layers": "6",
        "use_dpt": True,
    },
    {
        "name": "loss_l1",
        "desc": "Plain L1 loss",
        "extra": ["--loss_type", "l1"],
        "view_layers": "6",
        "use_dpt": True,
    },
    {
        "name": "object_emb",
        "desc": "Per-object segment embeddings (object-level encoding)",
        "extra": ["--use_object_emb"],
        "view_layers": "6",
        "use_dpt": True,
    },
]


def run_training(ab):
    name     = ab["name"]
    out_dir  = Path(f"runs/ablation_{name}")
    final_pt = out_dir / "checkpoints" / "final.pt"
    last_pt  = out_dir / "checkpoints" / "last.pt"

    # Already done — skip
    if final_pt.exists():
        print(f"\n[SKIP] {ab['desc']} — final.pt already exists.")
        return True

    args = list(BASE)
    args += ["--out_dir", str(out_dir)]
    args += ["--view_layers", ab["view_layers"]]
    if ab["use_dpt"]:
        args += ["--use_dpt_decoder"]
    args += ab["extra"]

    # Resume if interrupted
    if last_pt.exists():
        print(f"\n[RESUME] {ab['desc']} — resuming from last.pt")
        args += ["--resume_from", str(last_pt)]
    else:
        print(f"\n[START] {ab['desc']}")

    print(f"{'='*55}")
    result = subprocess.run([PYTHON, "train_course_baseline.py"] + args)
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {name} (code {result.returncode})")
        return False
    return True


def run_eval(ab):
    name     = ab["name"]
    final_pt = Path(f"runs/ablation_{name}/checkpoints/final.pt")
    if not final_pt.exists():
        print(f"[SKIP EVAL] {name} — no final.pt found.")
        return
    print(f"\n  Evaluating: {ab['desc']}")
    subprocess.run([
        PYTHON, "eval_psnr.py",
        "--checkpoint", str(final_pt),
        "--data_path", DATA,
        "--device", DEVICE,
    ])


def main():
    for ab in ABLATIONS:
        ok = run_training(ab)
        if ok:
            run_eval(ab)

    print(f"\n{'='*55}")
    print("  All done. Compare results with baseline PSNR 23.82 dB")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
