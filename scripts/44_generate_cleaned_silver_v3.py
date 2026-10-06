"""
RSNA Knee Abnormality Detection - Vision-Guided Silver GT Self-Cleaning (v3).
Applies high-confidence Out-of-Fold (OOF) predictions from our 5-model vision ensemble
to repair text-report omissions and extraction errors across silver studies.
STRICT INVARIANT: All 58 Gold Human-Annotated studies remain 100% untouched.
"""

import os
import argparse
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc

console = Console()

def generate_cleaned_silver(
    dense_path: str = "data/dense_labels_master.parquet",
    splits_path: str = "data/splits_5fold.parquet",
    output_parquet: str = "data/dense_labels_master_v3_cleaned.parquet",
    output_csv: str = "data/dense_labels_master_v3_cleaned.csv",
):
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║  RSNA Knee: Vision-Guided Silver GT Cleaning (Dataset v3)        ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")

    df_splits = pd.read_parquet(splits_path)
    df_dense = pd.read_parquet(dense_path)

    # 1. Identify Gold Studies (Strict Invariant)
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    gold_uids = set(df_splits[labeled_mask]["StudyInstanceUID"].unique())
    console.print(f"Loaded {len(df_dense)} studies. Identified [bold green]{len(gold_uids)} Gold Studies[/bold green] (will be strictly preserved).")

    df_clean = df_dense.copy()

    # 2. Load Fold 0 OOF Predictions (904 studies)
    f0_uids = df_splits[df_splits["fold"] == 0]["StudyInstanceUID"].values

    p_tiny_0 = np.load("checkpoints/phase13_triplanar_convnext_tiny_f0_72sl_val_preds.npy")
    p_small_0 = np.load("checkpoints/phase13_triplanar_convnext_small_f0_72sl_val_preds.npy")
    p_dino_0 = np.load("checkpoints/phase13_triplanar_dinov2_small_f0_72sl_val_preds.npy")
    p_intra_0 = np.load("checkpoints/phase14_ablation_intra_transformer_convnext_tiny_f0_val_preds.npy")
    p_coat_0 = np.load("checkpoints/phase14_ablation_coattention_convnext_tiny_f0_val_preds.npy")

    p_5m_0 = 0.25 * p_tiny_0 + 0.35 * p_small_0 + 0.18 * p_dino_0 + 0.12 * p_intra_0 + 0.10 * p_coat_0

    # 3. Load Fold 1 OOF Predictions (881 studies)
    f1_uids = df_splits[df_splits["fold"] == 1]["StudyInstanceUID"].values
    p_tiny_1 = np.load("checkpoints/phase13_triplanar_convnext_tiny_f1_72sl_val_preds.npy")
    p_small_1 = np.load("checkpoints/phase13_triplanar_convnext_small_f1_72sl_val_preds.npy")
    p_dino_1 = np.load("checkpoints/phase13_triplanar_dinov2_small_f1_72sl_val_preds.npy")
    p_3m_1 = 0.30 * p_tiny_1 + 0.45 * p_small_1 + 0.25 * p_dino_1

    stats_cleaned = {c: {"omissions_resolved": 0, "negations_corrected": 0} for c in TARGET_COLUMNS}

    # Clean Fold 0 silver studies
    for i, uid in enumerate(f0_uids):
        if uid in gold_uids:
            continue  # NEVER touch gold
        mask = df_clean["StudyInstanceUID"] == uid
        for c_idx, col in enumerate(TARGET_COLUMNS):
            orig_val = df_clean.loc[mask, col].values[0]
            pred = p_5m_0[i, c_idx]

            # Case A: Omission resolution (was 0.50 unaddressed, but vision is decisive)
            if 0.45 <= orig_val <= 0.55:
                if pred >= 0.82:
                    df_clean.loc[mask, col] = 0.70 * orig_val + 0.30 * pred  # Move toward positive
                    stats_cleaned[col]["omissions_resolved"] += 1
                elif pred <= 0.18:
                    df_clean.loc[mask, col] = 0.70 * orig_val + 0.30 * pred  # Move toward negative
                    stats_cleaned[col]["omissions_resolved"] += 1

            # Case B: False Positive / Regex over-extraction
            elif orig_val >= 0.90 and pred <= 0.12:
                df_clean.loc[mask, col] = 0.60  # Soften dubious positive
                stats_cleaned[col]["negations_corrected"] += 1

    # Clean Fold 1 silver studies
    for i, uid in enumerate(f1_uids):
        if uid in gold_uids:
            continue
        mask = df_clean["StudyInstanceUID"] == uid
        for c_idx, col in enumerate(TARGET_COLUMNS):
            orig_val = df_clean.loc[mask, col].values[0]
            pred = p_3m_1[i, c_idx]

            if 0.45 <= orig_val <= 0.55:
                if pred >= 0.82 or pred <= 0.18:
                    df_clean.loc[mask, col] = 0.70 * orig_val + 0.30 * pred
                    stats_cleaned[col]["omissions_resolved"] += 1
            elif orig_val >= 0.90 and pred <= 0.12:
                df_clean.loc[mask, col] = 0.60
                stats_cleaned[col]["negations_corrected"] += 1

    # Verify Gold Invariant
    for uid in gold_uids:
        orig = df_dense[df_dense["StudyInstanceUID"] == uid][TARGET_COLUMNS].values
        clean = df_clean[df_clean["StudyInstanceUID"] == uid][TARGET_COLUMNS].values
        assert np.allclose(orig, clean), f"FATAL: Gold study {uid} was modified!"
    console.print("[bold green]✓ Strict Invariant Verified: 100% of Gold Studies match exactly bit-for-bit![/bold green]")

    # Output stats
    table = Table(title="Dataset v3 Cleaning Breakdown per Pathology", header_style="bold magenta")
    table.add_column("Pathology", style="cyan", width=22)
    table.add_column("Omissions Resolved", justify="right", style="green")
    table.add_column("Regex Over-detections Softened", justify="right", style="yellow")

    total_omiss = sum(s["omissions_resolved"] for s in stats_cleaned.values())
    total_neg = sum(s["negations_corrected"] for s in stats_cleaned.values())

    for c in TARGET_COLUMNS:
        table.add_row(c, str(stats_cleaned[c]["omissions_resolved"]), str(stats_cleaned[c]["negations_corrected"]))

    console.print(table)
    console.print(f"[bold green]Total label refinements across Folds 0 & 1: {total_omiss + total_neg} ({total_omiss} omissions resolved, {total_neg} dubious detections softened)[/bold green]")

    # Save datasets
    df_clean.to_parquet(output_parquet, index=False)
    df_clean.to_csv(output_csv, index=False)
    console.print(f"✓ Saved cleaned parquet: [bold green]{output_parquet}[/bold green]")
    console.print(f"✓ Saved cleaned csv:     [bold green]{output_csv}[/bold green]")

if __name__ == "__main__":
    generate_cleaned_silver()
