"""
RSNA Knee Abnormality Detection - Multi-GPU Parallel Training Orchestrator.
Spawns parallel cross-validation training jobs across dual RTX A6000 GPUs:
- GPU 0: ResNet-34 2.5D Gated MIL (5-Fold Full Cohort)
- GPU 1: DINOv2 (ViT-Small Patch14) 2.5D Gated MIL (5-Fold Full Cohort)
"""

import os
import sys
import subprocess
import time
from rich.console import Console

console = Console()

def run_parallel_training():
    console.print("[bold green]=== KneeVision-AI: Multi-GPU Parallel Training Launch ===[/bold green]")
    console.print("Launching concurrent 5-fold cross validation:")
    console.print(" • [bold cyan]GPU 0[/bold cyan]: [magenta]ResNet-34 (2.5D MIL)[/magenta] across all 5 folds")
    console.print(" • [bold cyan]GPU 1[/bold cyan]: [yellow]DINOv2 ViT-Small (2.5D MIL)[/yellow] across all 5 folds\n")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Command for GPU 0 (ResNet-34)
    cmd_gpu0 = [
        sys.executable,
        "scripts/04_train_5fold_cv.py",
        "--backbone", "resnet34",
        "--fold", "-1",
        "--epochs", "10",
        "--batch_size", "4",
        "--grad_accum_steps", "2",
        "--target_slices", "24",
        "--lr", "5e-4",
        "--loss", "bce",
        "--checkpoint_dir", "checkpoints",
    ]

    # Command for GPU 1 (DINOv2)
    cmd_gpu1 = [
        sys.executable,
        "scripts/04_train_5fold_cv.py",
        "--backbone", "vit_small_patch14_dinov2.lvd142m",
        "--fold", "-1",
        "--epochs", "10",
        "--batch_size", "4",
        "--grad_accum_steps", "2",
        "--target_slices", "24",
        "--lr", "3e-4",
        "--loss", "bce",
        "--checkpoint_dir", "checkpoints",
    ]

    env_gpu0 = os.environ.copy()
    env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"

    env_gpu1 = os.environ.copy()
    env_gpu1["CUDA_VISIBLE_DEVICES"] = "1"

    log_gpu0 = open("logs/train_gpu0_resnet34.log", "w")
    log_gpu1 = open("logs/train_gpu1_dinov2.log", "w")

    console.print("[bold blue]Starting GPU 0 process...[/bold blue]")
    p0 = subprocess.Popen(cmd_gpu0, env=env_gpu0, stdout=log_gpu0, stderr=subprocess.STDOUT)

    console.print("[bold blue]Starting GPU 1 process...[/bold blue]")
    p1 = subprocess.Popen(cmd_gpu1, env=env_gpu1, stdout=log_gpu1, stderr=subprocess.STDOUT)

    console.print(f"\n[bold green]✓ Both processes spawned![/bold green]")
    console.print(f" • GPU 0 PID: [cyan]{p0.pid}[/cyan] (Log: [dim]logs/train_gpu0_resnet34.log[/dim])")
    console.print(f" • GPU 1 PID: [cyan]{p1.pid}[/cyan] (Log: [dim]logs/train_gpu1_dinov2.log[/dim])")

    # Monitor loop
    while p0.poll() is None or p1.poll() is None:
        time.sleep(15)

    log_gpu0.close()
    log_gpu1.close()

    console.print("\n[bold green]=== Multi-GPU Training Completed ===[/bold green]")
    console.print(f"GPU 0 Exit Code: {p0.returncode}")
    console.print(f"GPU 1 Exit Code: {p1.returncode}")


if __name__ == "__main__":
    run_parallel_training()
