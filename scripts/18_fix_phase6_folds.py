"""
KneeVision-AI: Phase 6 Clean Completion Orchestrator (Folds 1, 3, 4)
====================================================================
1. Fold 1: Re-train with 2-epoch linear warmup & gentle differential LR (6e-5 / 3e-4) to prevent gradient shock.
2. Fold 3: Resume to 25 epochs.
3. Fold 4: Resume to 25 epochs.
"""

import os, sys, subprocess, time
from pathlib import Path
from src.config import resolve_checkpoint_dir, get_python_bin, get_experiment_artifacts

exp_artifacts = get_experiment_artifacts("phase_06_loss_and_25ep_retraining")
ckpt_dir = resolve_checkpoint_dir()
py_bin = get_python_bin()

log_file = exp_artifacts["logs"] / "train_phase_06_clean.log"

print(f"=== Starting Phase 6 Clean Completion ===")
print(f"Python       : {py_bin}")
print(f"Checkpoints  : {ckpt_dir}")
print(f"Log          : {log_file}")

tasks = [
    # Task 1: Fold 1 from clean weak-pretrained foundation with 2-epoch warmup
    {
        "fold": 1,
        "epochs": 25,
        "backbone_lr": 6e-5,
        "head_lr": 3e-4,
        "warmup_epochs": 2,
        "resume": str(ckpt_dir / "convnext_tiny_sagittal_weak_ft_fold1_best.pth") if (ckpt_dir / "convnext_tiny_sagittal_weak_ft_fold1_best.pth").exists() else None,
        "desc": "Fold 1 clean retraining (gentle LR + 2 ep warmup)"
    },
    # Task 2: Fold 3 resume
    {
        "fold": 3,
        "epochs": 25,
        "backbone_lr": 8e-5,
        "head_lr": 4e-4,
        "warmup_epochs": 1,
        "resume": str(ckpt_dir / "dense_288px_convnext_tiny_sagittal_fold3_best.pth"),
        "desc": "Fold 3 continuation to 25 epochs"
    },
    # Task 3: Fold 4 resume
    {
        "fold": 4,
        "epochs": 25,
        "backbone_lr": 8e-5,
        "head_lr": 4e-4,
        "warmup_epochs": 1,
        "resume": str(ckpt_dir / "dense_288px_convnext_tiny_sagittal_fold4_best.pth"),
        "desc": "Fold 4 continuation to 25 epochs"
    }
]

for t in tasks:
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {t['desc']}...")
    cmd = [
        py_bin, "scripts/11_train_dense_288px.py",
        "--backbone", "convnext_tiny",
        "--fold", str(t["fold"]),
        "--epochs", str(t["epochs"]),
        "--image_size", "288",
        "--batch_size", "4",
        "--backbone_lr", str(t["backbone_lr"]),
        "--head_lr", str(t["head_lr"]),
        "--warmup_epochs", str(t["warmup_epochs"]),
        "--labels_path", "data/dense_labels_master.parquet",
    ]
    if t["resume"] and os.path.exists(t["resume"]):
        cmd.extend(["--resume", t["resume"]])

    print("CMD:", " ".join(cmd))
    with open(log_file, "a") as f_log:
        f_log.write(f"\n\n{'='*60}\n{t['desc']}\n{'='*60}\n")
        f_log.flush()
        res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(f"Error on fold {t['fold']}: exit code {res.returncode}")
    else:
        print(f"Completed {t['desc']} successfully.")

print("\n=== All Phase 6 targeted folds finished! ===")
