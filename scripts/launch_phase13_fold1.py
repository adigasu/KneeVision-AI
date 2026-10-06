import subprocess
import sys
import os

py_exec = sys.executable

jobs = [
    {
        "name": "ConvNeXt-Tiny (Fold 1, GPU 0)",
        "cmd": [
            py_exec, "-u", "scripts/36_train_triplanar_mil.py",
            "--backbone", "convnext_tiny",
            "--fold", "1",
            "--slices_per_plane", "24",
            "--image_size", "288",
            "--batch_size", "2",
            "--grad_accum_steps", "8",
            "--epochs", "20",
            "--lr", "3e-4",
            "--backbone_lr_mult", "0.1",
            "--loss_type", "consensus_denoised",
            "--gold_weight", "2.0",
            "--device", "cuda:0",
            "--run_name", "phase13_triplanar_convnext_tiny_f1_72sl",
        ],
        "log": "logs/train_phase13_triplanar_convnext_tiny_f1_gpu0.log",
    },
    {
        "name": "ConvNeXt-Small (Fold 1, GPU 1)",
        "cmd": [
            py_exec, "-u", "scripts/36_train_triplanar_mil.py",
            "--backbone", "convnext_small",
            "--fold", "1",
            "--slices_per_plane", "24",
            "--image_size", "288",
            "--batch_size", "2",
            "--grad_accum_steps", "8",
            "--epochs", "20",
            "--lr", "3e-4",
            "--backbone_lr_mult", "0.1",
            "--loss_type", "consensus_denoised",
            "--gold_weight", "2.0",
            "--device", "cuda:1",
            "--run_name", "phase13_triplanar_convnext_small_f1_72sl",
        ],
        "log": "logs/train_phase13_triplanar_convnext_small_f1_gpu1.log",
    },
    {
        "name": "DINOv2-Small (Fold 1, GPU 0)",
        "cmd": [
            py_exec, "-u", "scripts/36_train_triplanar_mil.py",
            "--backbone", "vit_small_patch14_dinov2.lvd142m",
            "--fold", "1",
            "--slices_per_plane", "24",
            "--image_size", "280",
            "--batch_size", "2",
            "--grad_accum_steps", "8",
            "--epochs", "20",
            "--lr", "2e-4",
            "--backbone_lr_mult", "0.1",
            "--loss_type", "consensus_denoised",
            "--gold_weight", "2.0",
            "--device", "cuda:0",
            "--run_name", "phase13_triplanar_dinov2_small_f1_72sl",
        ],
        "log": "logs/train_phase13_triplanar_dinov2_small_f1_gpu0.log",
    },
]

print("=== Launching Detached Phase 13 Fold 1 Jobs ===")
for j in jobs:
    f_log = open(j["log"], "w")
    proc = subprocess.Popen(
        j["cmd"],
        stdout=f_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Fully detaches process into its own session
    )
    print(f"✓ Dispatched {j['name']} with PID {proc.pid} -> Log: {j['log']}")

print("All 3 jobs launched detached!")
