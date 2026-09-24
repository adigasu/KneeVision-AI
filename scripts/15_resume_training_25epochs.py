"""
KneeVision-AI: Resume Dense Training to 25 Epochs
==================================================
Resumes all 5 ConvNeXt-Tiny (or any backbone) folds from their existing
dense_288px checkpoints and trains to a target epoch count.

Usage
-----
    python scripts/15_resume_training_25epochs.py [--backbone convnext_tiny] [--target_epochs 25]

Config
------
    All paths resolved via src.config (no hardcoded paths).
    Set KNEEVISION_DATA_DIR or configs/env.yaml before running.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Resolve repo root and add to path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.config import resolve_checkpoint_dir, resolve_data_dir, get_python_bin

PYTHON = get_python_bin()
TRAIN_SCRIPT = _REPO / "scripts" / "11_train_dense_288px.py"


def parse_args():
    p = argparse.ArgumentParser(description="Resume dense training for all folds")
    p.add_argument("--backbone",      type=str, default="convnext_tiny",
                   help="Backbone name (must match existing checkpoints)")
    p.add_argument("--target_epochs", type=int, default=25,
                   help="Total epochs to train to (resumes from checkpoint epoch)")
    p.add_argument("--image_size",    type=int, default=288)
    p.add_argument("--folds",         type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Which folds to resume (default: all 5)")
    p.add_argument("--batch_size",    type=int, default=2)
    p.add_argument("--backbone_lr",   type=float, default=8e-5)
    p.add_argument("--head_lr",       type=float, default=4e-4)
    p.add_argument("--labels_path",   type=str, default="data/dense_labels_master.parquet")
    p.add_argument("--use_tta", action="store_true", default=False, help="Enable TTA")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def find_checkpoint(ckpt_dir: Path, backbone: str, fold: int) -> Path | None:
    """Find the best checkpoint file for a given backbone and fold."""
    candidates = [
        ckpt_dir / f"dense_288px_{backbone}_sagittal_fold{fold}_best.pth",
        ckpt_dir / f"dense_{backbone}_sagittal_fold{fold}_best.pth",
        ckpt_dir / f"{backbone}_sagittal_fold{fold}_best.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    args   = parse_args()
    ckpt_dir = resolve_checkpoint_dir()

    print(f"\n{'='*60}")
    print(f"KneeVision-AI: Resume Training to {args.target_epochs} Epochs")
    print(f"{'='*60}")
    print(f"  backbone      : {args.backbone}")
    print(f"  target_epochs : {args.target_epochs}")
    print(f"  img_size      : {args.image_size}")
    print(f"  folds         : {args.folds}")
    print(f"  checkpoint_dir: {ckpt_dir}")
    print(f"  python        : {PYTHON}")
    print()

    for fold in args.folds:
        ckpt = find_checkpoint(ckpt_dir, args.backbone, fold)
        if ckpt is None:
            print(f"  [SKIP] fold {fold}: no checkpoint found in {ckpt_dir}")
            continue

        # Read current epoch from checkpoint
        import torch
        meta = torch.load(ckpt, map_location="cpu")
        current_epoch = meta.get("epoch", 0)

        if current_epoch >= args.target_epochs - 1:
            print(f"  [SKIP] fold {fold}: already at epoch {current_epoch} "
                  f"(target={args.target_epochs})")
            continue

        remaining = args.target_epochs - current_epoch - 1
        print(f"\n[fold {fold}] Resuming from epoch {current_epoch} → {args.target_epochs} "
              f"({remaining} epochs remaining)")
        print(f"           Checkpoint: {ckpt.name}")

        cmd = [
            PYTHON, str(TRAIN_SCRIPT),
            "--backbone",      args.backbone,
            "--fold",          str(fold),
            "--epochs",        str(args.target_epochs),
            "--image_size",    str(args.image_size),
            "--batch_size",    str(args.batch_size),
            "--backbone_lr",   str(args.backbone_lr),
            "--head_lr",       str(args.head_lr),
            "--labels_path",   args.labels_path,
            "--resume",        str(ckpt),
        ]

        print(f"  CMD: {' '.join(cmd)}\n")

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(_REPO), check=False)
            if result.returncode != 0:
                print(f"  [ERROR] fold {fold} exited with code {result.returncode}")
            else:
                print(f"  [DONE] fold {fold} completed successfully")

    print("\nAll folds processed.")


if __name__ == "__main__":
    main()
