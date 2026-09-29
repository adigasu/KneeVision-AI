"""
Unified Real-Time Progress Monitor for Parallel 2-GPU Single-Plane & Tri-Planar Training.
"""

import os
import glob
import json
from rich.console import Console
from rich.table import Table

def check_progress():
    console = Console()
    dirs = [
        ("Phase 09: Single-Plane Parallel (Sagittal 288px)", "artifacts/experiments/phase_09_parallel_finetune"),
        ("Phase 10: Tri-Planar Multi-View Parallel (Sag + Cor + Ax)", "artifacts/experiments/phase_10_triplanar_parallel"),
    ]

    for title, base_dir in dirs:
        if not os.path.exists(base_dir):
            continue

        exp_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d) and os.path.basename(d) != "logs"]
        if not exp_dirs:
            continue

        table = Table(title=f"Training Suite: {title}", show_header=True, header_style="bold magenta")
        table.add_column("Backbone Architecture", style="cyan", width=42)
        table.add_column("Folds Done", justify="center", width=12)
        table.add_column("Current Status", justify="center", width=22)
        table.add_column("Best Val AUC", justify="center", style="bold green", width=15)
        table.add_column("Gold AUC", justify="center", style="bold yellow", width=15)

        for exp_dir in sorted(exp_dirs):
            name = os.path.basename(exp_dir)
            fold_dirs = sorted(glob.glob(os.path.join(exp_dir, "fold_*")))
            
            folds_done = 0
            best_val = 0.0
            best_gold = 0.0
            cur_status = "In Progress"

            for f_dir in fold_dirs:
                hist_file = os.path.join(f_dir, "training_history.json")
                if os.path.exists(hist_file):
                    try:
                        with open(hist_file) as fp:
                            history = json.load(fp)
                        if history:
                            last_ep = history[-1]["epoch"]
                            cur_status = f"{os.path.basename(f_dir)} (Ep {last_ep}/25)"
                            for entry in history:
                                if entry.get("val_auc", 0) > best_val:
                                    best_val = entry.get("val_auc", 0)
                                    best_gold = entry.get("gold_auc", 0)
                            if last_ep >= 25:
                                folds_done += 1
                    except Exception:
                        pass

            table.add_row(
                name,
                f"{folds_done}/5",
                cur_status,
                f"{best_val:.4f}" if best_val > 0 else "Initializing...",
                f"{best_gold:.4f}" if best_gold > 0 else "Initializing...",
            )

        console.print(table)
        console.print("")

if __name__ == "__main__":
    check_progress()
