"""
KneeVision-AI: ConvNeXt-Small 5-Fold Training (Phase C)
=========================================================
Trains 5-fold ConvNeXt-Small (50M params) + Gated Attention MIL on
dense v4_blend soft labels at 320×320 resolution.

Expected gain over ConvNeXt-Tiny: +0.025–0.035 Macro AUC

Usage
-----
    python scripts/16_train_convnext_small_5fold.py [options]

Config
------
    All paths from src.config — set KNEEVISION_DATA_DIR or configs/env.yaml.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.config import resolve_checkpoint_dir, get_python_bin

PYTHON       = get_python_bin()
TRAIN_SCRIPT = _REPO / "scripts" / "11_train_dense_288px.py"


def parse_args():
    p = argparse.ArgumentParser(description="Train ConvNeXt-Small 5-fold from scratch")
    p.add_argument("--folds",         type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs",        type=int, default=25)
    p.add_argument("--img_size",      type=int, default=320,
                   help="320 is ConvNeXt-Small pretraining resolution")
    p.add_argument("--batch_size",    type=int, default=2,
                   help="ConvNeXt-Small uses more VRAM; keep batch_size=2")
    p.add_argument("--backbone_lr",   type=float, default=6e-5,
                   help="Lower than Tiny (8e-5) — larger model needs gentler finetuning")
    p.add_argument("--head_lr",       type=float, default=4e-4)
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Linear LR warmup before cosine annealing")
    p.add_argument("--label_file",    type=str,
                   default="data/dense_labels_master.parquet")
    p.add_argument("--start_fold",    type=int, default=0,
                   help="Resume from this fold index (skip earlier folds)")
    p.add_argument("--use_tta", action="store_true", default=False, help="Enable TTA")
    p.add_argument("--dry_run",       action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    from src.config import get_experiment_artifacts
    exp_art = get_experiment_artifacts("phase_07_convnext_small_50m")
    ckpt_dir = exp_art["checkpoints"]

    print(f"\n{'='*60}")
    print(f"KneeVision-AI: ConvNeXt-Small 5-Fold Training (Phase C)")
    print(f"{'='*60}")
    print(f"  backbone     : convnext_small  (50M params)")
    print(f"  img_size     : {args.img_size}")
    print(f"  epochs       : {args.epochs}")
    print(f"  folds        : {[f for f in args.folds if f >= args.start_fold]}")
    print(f"  backbone_lr  : {args.backbone_lr}")
    print(f"  head_lr      : {args.head_lr}")
    print(f"  warmup_epochs: {args.warmup_epochs}")
    print(f"  checkpoint_dir: {ckpt_dir}")
    print(f"  Expected ETA : ~12 h/fold on dual RTX A6000")
    print()

    for fold in args.folds:
        if fold < args.start_fold:
            print(f"  [SKIP] fold {fold}: below start_fold={args.start_fold}")
            continue

        print(f"\n[fold {fold}] Starting ConvNeXt-Small training from scratch ...")

        cmd = [
            PYTHON, str(TRAIN_SCRIPT),
            "--backbone",      "convnext_small",
            "--fold",          str(fold),
            "--epochs",        str(args.epochs),
            "--image_size",    str(args.img_size),
            "--batch_size",    str(args.batch_size),
            "--backbone_lr",   str(args.backbone_lr),
            "--head_lr",       str(args.head_lr),
            "--warmup_epochs", str(args.warmup_epochs),
            "--labels_path",   args.label_file,
            "--checkpoint_dir", str(ckpt_dir),
        ]

        print(f"  CMD: {' '.join(cmd)}\n")

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(_REPO), check=False)
            if result.returncode != 0:
                print(f"  [ERROR] fold {fold} exited with code {result.returncode}")
            else:
                print(f"  [DONE] fold {fold} completed")

    print("\n✅ All folds processed.")


if __name__ == "__main__":
    main()
