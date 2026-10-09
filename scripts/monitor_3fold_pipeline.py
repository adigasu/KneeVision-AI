"""
RSNA Knee Abnormality Detection - Phase 17: Multi-Contrast 3-Fold Training Monitor
Displays GPU status, active training processes, checkpoint states, and recent log outputs.
"""

import os
import subprocess
from rich.console import Console
from rich.table import Table

console = Console()
console.print("[bold blue]=== RSNA Knee: Phase 17 3-Fold Pipeline Monitor ===[/bold blue]")

# 1. Checkpoints status
table = Table(title="6-Model Training & Checkpoint Status")
table.add_column("Model / Stream", style="cyan")
table.add_column("Fold", style="white", justify="center")
table.add_column("Checkpoint File", style="yellow")
table.add_column("Status", justify="center")

models = [
    ("T1 Anatomical (GPU 0)", 0, "phase17_decoupled_t1_convnext_small_384px_f0"),
    ("T1 Anatomical (GPU 0)", 1, "phase17_decoupled_t1_convnext_small_384px_f1"),
    ("T1 Anatomical (GPU 0)", 2, "phase17_decoupled_t1_convnext_small_384px_f2"),
    ("T2 FS         (GPU 1)", 0, "phase17_decoupled_t2_convnext_small_384px_f0"),
    ("T2 FS         (GPU 1)", 1, "phase17_decoupled_t2_convnext_small_384px_f1"),
    ("T2 FS         (GPU 1)", 2, "phase17_decoupled_t2_convnext_small_384px_f2"),
]

for stream, fold, run_name in models:
    ckpt = f"checkpoints/{run_name}_best.pth"
    preds = f"checkpoints/{run_name}_val_preds.npy"
    if os.path.exists(ckpt) and os.path.exists(preds):
        status = "[bold green]✓ Completed[/bold green]"
    elif os.path.exists(ckpt):
        status = "[yellow]Saving...[/yellow]"
    else:
        status = "[dim]Pending / Training[/dim]"
    table.add_row(stream, f"Fold {fold}", ckpt, status)

console.print(table)

# 2. Check running processes
try:
    ps_out = subprocess.check_output(["pgrep", "-af", "53_train_decoupled_multicontrast_3fold.py"]).decode()
    console.print("\n[bold green]Active Training Processes:[/bold green]")
    for line in ps_out.strip().splitlines():
        console.print(f"  • {line}")
except Exception:
    console.print("\n[yellow]No active 53_train_decoupled_multicontrast_3fold.py processes found.[/yellow]")
