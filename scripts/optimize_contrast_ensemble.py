"""
RSNA Knee Abnormality Detection - Pathology-Specific Multi-Contrast Ensemble Optimizer.
Optimizes contrast blending weights across T1 Anatomical, T2 FS, and Joint models
to directly maximize Multilabel Macro AUC.
Output: checkpoints/contrast_ensemble_weights.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc


def optimize_ensemble(t2_preds_path: str, t1_preds_path: str, targets_path: str, output_json: str):
    console = Console()
    console.print("[bold blue]=== RSNA Knee: Multi-Contrast Pathology-Specific Ensemble Optimizer ===[/bold blue]")

    p_t2 = np.load(t2_preds_path)
    p_t1 = np.load(t1_preds_path)
    targets = np.load(targets_path)

    n_samples, n_classes = p_t2.shape
    console.print(f"Loaded {n_samples:,} validation predictions across {n_classes} classes.")

    # 1. Individual metrics
    m_t2 = compute_macro_auc(targets, p_t2)
    m_t1 = compute_macro_auc(targets, p_t1)
    p_uniform = 0.5 * p_t2 + 0.5 * p_t1
    m_uniform = compute_macro_auc(targets, p_uniform)

    # 2. Per-class bounded optimization: find w in [0, 1] for w * p_t2 + (1 - w) * p_t1
    optimal_weights = {}
    p_opt = np.zeros_like(p_t2)

    for c_idx, col in enumerate(TARGET_COLUMNS):
        y_true = targets[:, c_idx]
        s_t2 = p_t2[:, c_idx]
        s_t1 = p_t1[:, c_idx]

        # Best weight search over fine grid [0.0 to 1.0]
        best_w = 0.5
        best_auc = 0.0

        for w in np.linspace(0.0, 1.0, 101):
            s_blend = w * s_t2 + (1.0 - w) * s_t1
            from sklearn.metrics import roc_auc_score
            try:
                # Handle binary or continuous soft targets
                y_bin = (y_true >= 0.5).astype(int)
                if len(np.unique(y_bin)) > 1:
                    auc = roc_auc_score(y_bin, s_blend)
                else:
                    auc = 0.5
            except Exception:
                auc = 0.5

            if auc > best_auc:
                best_auc = auc
                best_w = w

        optimal_weights[col] = {
            "weight_t2_fs": float(round(best_w, 3)),
            "weight_t1_anatomical": float(round(1.0 - best_w, 3)),
            "optimized_auc": float(round(best_auc, 4)),
            "baseline_t2_auc": float(round(m_t2["per_class_auc"].get(col, 0.0), 4)),
            "baseline_t1_auc": float(round(m_t1["per_class_auc"].get(col, 0.0), 4)),
        }
        p_opt[:, c_idx] = best_w * s_t2 + (1.0 - best_w) * s_t1

    m_opt = compute_macro_auc(targets, p_opt)

    # Comparison Table
    table = Table(title="Contrast Ensembling Benchmark & Optimal Pathology Weights")
    table.add_column("Abnormality", style="cyan")
    table.add_column("T2 FS Alone", justify="right", style="green")
    table.add_column("T1 Alone", justify="right", style="magenta")
    table.add_column("Uniform 50/50", justify="right", style="yellow")
    table.add_column("Optimized Blend", justify="right", style="bold green")
    table.add_column("Optimal T2 : T1 Weight", justify="center", style="bold white")

    for col in TARGET_COLUMNS:
        w_t2 = optimal_weights[col]["weight_t2_fs"]
        w_t1 = optimal_weights[col]["weight_t1_anatomical"]
        table.add_row(
            col,
            f"{m_t2['per_class_auc'].get(col, 0.0):.4f}",
            f"{m_t1['per_class_auc'].get(col, 0.0):.4f}",
            f"{m_uniform['per_class_auc'].get(col, 0.0):.4f}",
            f"{m_opt['per_class_auc'].get(col, 0.0):.4f}",
            f"{w_t2:.2f} : {w_t1:.2f}",
        )

    console.print(table)
    console.print(f"\n[bold]Baseline T2 Macro AUC:[/bold]      {m_t2['macro_auc']:.4f}")
    console.print(f"[bold]Baseline T1 Macro AUC:[/bold]      {m_t1['macro_auc']:.4f}")
    console.print(f"[bold]Uniform Ensemble Macro AUC:[/bold] {m_uniform['macro_auc']:.4f} (+{m_uniform['macro_auc'] - m_t2['macro_auc']:+.4f})")
    console.print(f"[bold green]Optimized Blend Macro AUC:[/bold green]  [bold green]{m_opt['macro_auc']:.4f}[/bold green] (+{m_opt['macro_auc'] - m_t2['macro_auc']:+.4f})")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({
            "macro_auc_t2": float(m_t2["macro_auc"]),
            "macro_auc_t1": float(m_t1["macro_auc"]),
            "macro_auc_uniform": float(m_uniform["macro_auc"]),
            "macro_auc_optimized": float(m_opt["macro_auc"]),
            "per_class": optimal_weights,
        }, f, indent=2)

    console.print(f"\nSaved optimal ensemble weights to: [green]{output_json}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2_preds", type=str, required=True)
    parser.add_argument("--t1_preds", type=str, required=True)
    parser.add_argument("--targets", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="checkpoints/contrast_ensemble_weights.json")
    args = parser.parse_args()

    optimize_ensemble(args.t2_preds, args.t1_preds, args.targets, args.output_json)
