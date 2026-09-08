"""
RSNA Knee Abnormality Detection - Phase 6 Parallel Dense 288px Training Orchestrator.
Spawns concurrent 5-fold cross-validation jobs across dual NVIDIA RTX A6000 GPUs:
- GPU 0: ResNet-34 (288x288 Gated Attention MIL) across 5 folds
- GPU 1: ConvNeXt-Tiny (288x288 Gated Attention MIL) across 5 folds
"""

import os
import sys
import subprocess
import time
from rich.console import Console

console = Console()


def run_parallel_dense_training():
    console.print("[bold green]=== KneeVision-AI: Phase 6 Multi-GPU Dense 288px Retraining Launch ===[/bold green]")
    console.print("Launching concurrent 5-fold cross-validation on dense [0.0, 1.0] soft targets:")
    console.print(" • [bold cyan]GPU 0[/bold cyan]: [magenta]ResNet-34 (288x288 2.5D MIL)[/magenta] across 5 folds")
    console.print(" • [bold cyan]GPU 1[/bold cyan]: [yellow]ConvNeXt-Tiny (288x288 2.5D MIL)[/yellow] across 5 folds\n")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Command for GPU 0 (ResNet-34 288px)
    cmd_gpu0 = [
        sys.executable,
        "scripts/11_train_dense_288px.py",
        "--backbone", "resnet34",
        "--image_size", "288",
        "--fold", "-1",
        "--epochs", "8",
        "--batch_size", "8",
        "--grad_accum_steps", "2",
        "--target_slices", "24",
        "--lr", "4e-4",
        "--checkpoint_dir", "checkpoints",
    ]

    # Command for GPU 1 (ConvNeXt-Tiny 288px)
    cmd_gpu1 = [
        sys.executable,
        "scripts/11_train_dense_288px.py",
        "--backbone", "convnext_tiny",
        "--image_size", "288",
        "--fold", "-1",
        "--epochs", "8",
        "--batch_size", "8",
        "--grad_accum_steps", "2",
        "--target_slices", "24",
        "--lr", "3e-4",
        "--checkpoint_dir", "checkpoints",
    ]

    env_gpu0 = os.environ.copy()
    env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"

    env_gpu1 = os.environ.copy()
    env_gpu1["CUDA_VISIBLE_DEVICES"] = "1"

    log_gpu0 = open("logs/train_dense_288_gpu0_resnet34.log", "w")
    log_gpu1 = open("logs/train_dense_288_gpu1_convnext.log", "w")

    console.print("[bold blue]Starting GPU 0 (ResNet-34 288px) process...[/bold blue]")
    p0 = subprocess.Popen(cmd_gpu0, env=env_gpu0, stdout=log_gpu0, stderr=subprocess.STDOUT)

    console.print("[bold blue]Starting GPU 1 (ConvNeXt-Tiny 288px) process...[/bold blue]")
    p1 = subprocess.Popen(cmd_gpu1, env=env_gpu1, stdout=log_gpu1, stderr=subprocess.STDOUT)

    console.print(f"\n[bold green]✓ Both processes spawned![/bold green]")
    console.print(f" • GPU 0 PID: [cyan]{p0.pid}[/cyan] (Log: [dim]logs/train_dense_288_gpu0_resnet34.log[/dim])")
    console.print(f" • GPU 1 PID: [cyan]{p1.pid}[/cyan] (Log: [dim]logs/train_dense_288_gpu1_convnext.log[/dim])")

    while p0.poll() is None or p1.poll() is None:
        time.sleep(15)

    log_gpu0.close()
    log_gpu1.close()

    console.print("\n[bold green]=== Multi-GPU Dense Retraining Completed ===[/bold green]")
    console.print(f"GPU 0 Exit Code: {p0.returncode}")
    console.print(f"GPU 1 Exit Code: {p1.returncode}")


if __name__ == "__main__":
    run_parallel_dense_training()
