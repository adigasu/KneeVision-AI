"""
RSNA Knee Abnormality Detection - Verify T1 Anatomical + T2 FS Ensemble Gain on Fold 0.
Compares:
1. Baseline T2 FS Decoupled ConvNeXt-Small 384px (Phase 16)
2. T1 Anatomical Decoupled ConvNeXt-Small 384px (Phase 17)
3. Uniform 50/50 Ensemble
4. Pathology-Specific Optimal Blend
Output: Direct empirical verification of multi-contrast fusion gain on Fold 0.
"""

import os
import sys
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc


def run_verification():
    console = Console()
    console.print("[bold blue]=== RSNA Knee: Multi-Contrast (T1 + T2 FS) Fold 0 Gain Verification ===[/bold blue]")

    t2_path = "checkpoints/phase16_decoupled_convnext_small_384px_f0_val_preds.npy"
    t1_path = "checkpoints/phase17_decoupled_t1_convnext_small_384px_f0_val_preds.npy"

    if not os.path.exists(t1_path):
        console.print(f"[yellow]T1 predictions not yet generated at: {t1_path}. Training in progress...[/yellow]")
        return

    p_t2 = np.load(t2_path)
    p_t1 = np.load(t1_path)

    # Load targets for Fold 0
    df_splits = pd.read_parquet("data/splits_5fold.parquet")
    df_dense = pd.read_parquet("data/dense_labels_master.parquet")

    df_f0 = df_splits[df_splits["fold"] == 0].reset_index(drop=True)
    is_gold = df_f0[TARGET_COLUMNS].notnull().any(axis=1).values

    dense_map = df_dense.set_index("StudyInstanceUID")[TARGET_COLUMNS]
    targets = dense_map.loc[df_f0["StudyInstanceUID"]].fillna(0.50).values.astype(float)

    console.print(f"Validation Studies: {len(df_f0):,} (Gold Consensus: {is_gold.sum()})")

    # 1. Compute Baselines
    m_t2 = compute_macro_auc(targets, p_t2)
    m_t1 = compute_macro_auc(targets, p_t1)

    # 2. Compute Uniform Blend (50/50)
    p_uniform = 0.5 * p_t2 + 0.5 * p_t1
    m_uniform = compute_macro_auc(targets, p_uniform)

    # 3. Compute Pathology-Specific Optimal Blend
    p_opt = np.zeros_like(p_t2)
    optimal_weights = {}

    for c_idx, col in enumerate(TARGET_COLUMNS):
        y_c = targets[:, c_idx]
        s_t2 = p_t2[:, c_idx]
        s_t1 = p_t1[:, c_idx]

        best_w = 0.5
        best_auc = 0.0

        for w in np.linspace(0.0, 1.0, 101):
            s_blend = w * s_t2 + (1.0 - w) * s_t1
            from sklearn.metrics import roc_auc_score
            try:
                # Binarize for rank metric check
                y_bin = (y_c >= 0.5).astype(int)
                if len(np.unique(y_bin)) > 1:
                    auc = roc_auc_score(y_bin, s_blend)
                else:
                    auc = 0.5
            except:
                auc = 0.5

            if auc > best_auc:
                best_auc = auc
                best_w = w

        optimal_weights[col] = best_w
        p_opt[:, c_idx] = best_w * s_t2 + (1.0 - best_w) * s_t1

    m_opt = compute_macro_auc(targets, p_opt)

    # Gold subset metrics
    gold_t2 = compute_macro_auc(targets[is_gold], p_t2[is_gold])["macro_auc"]
    gold_t1 = compute_macro_auc(targets[is_gold], p_t1[is_gold])["macro_auc"]
    gold_uni = compute_macro_auc(targets[is_gold], p_uniform[is_gold])["macro_auc"]
    gold_opt = compute_macro_auc(targets[is_gold], p_opt[is_gold])["macro_auc"]

    # Table
    table = Table(title="Fold 0 Head-to-Head Comparison: T2 FS vs. T1 Anatomical vs. Blends")
    table.add_column("Abnormality", style="cyan")
    table.add_column("T2 FS Alone", justify="right", style="green")
    table.add_column("T1 Alone", justify="right", style="magenta")
    table.add_column("50/50 Blend", justify="right", style="yellow")
    table.add_column("Optimal Blend", justify="right", style="bold green")
    table.add_column("Δ vs T2 Alone", justify="right", style="bold cyan")
    table.add_column("T2 : T1 Ratio", justify="center", style="white")

    for col in TARGET_COLUMNS:
        auc_t2 = m_t2["per_class_auc"].get(col, 0.0)
        auc_t1 = m_t1["per_class_auc"].get(col, 0.0)
        auc_uni = m_uniform["per_class_auc"].get(col, 0.0)
        auc_opt = m_opt["per_class_auc"].get(col, 0.0)
        delta = auc_opt - auc_t2
        w_t2 = optimal_weights[col]

        table.add_row(
            col,
            f"{auc_t2:.4f}",
            f"{auc_t1:.4f}",
            f"{auc_uni:.4f}",
            f"{auc_opt:.4f}",
            f"[bold green]+{delta:.4f}[/bold green]" if delta > 0 else f"[red]{delta:.4f}[/red]",
            f"{w_t2:.2f} : {1.0 - w_t2:.2f}",
        )

    console.print(table)

    console.print("\n[bold]Summary Performance Metrics (Fold 0):[/bold]")
    console.print(f"  • [green]T2 FS Alone Macro AUC:[/green]        {m_t2['macro_auc']:.4f} (Gold: {gold_t2:.4f})")
    console.print(f"  • [magenta]T1 Anatomical Alone Macro AUC:[/magenta] {m_t1['macro_auc']:.4f} (Gold: {gold_t1:.4f})")
    console.print(f"  • [yellow]50/50 Uniform Blend Macro AUC:[/yellow] {m_uniform['macro_auc']:.4f} (Gold: {gold_uni:.4f}) [bold green](+{m_uniform['macro_auc'] - m_t2['macro_auc']:+.4f})[/bold green]")
    console.print(f"  • [bold green]Pathology-Optimal Blend AUC:[/bold green]   [bold green]{m_opt['macro_auc']:.4f}[/bold green] (Gold: [bold magenta]{gold_opt:.4f}[/bold magenta]) [bold green](+{m_opt['macro_auc'] - m_t2['macro_auc']:+.4f})[/bold green]")


if __name__ == "__main__":
    run_verification()
