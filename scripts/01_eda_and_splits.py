"""
RSNA Knee Abnormality Detection - Exploratory Data Analysis & Leak-Free Multilabel Splits.
Analyzes label distributions, series anatomy, and partitions labeled studies into 5 folds.
"""

import os
import argparse
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS
from src.utils.common import seed_everything


def run_eda_and_splits(
    train_csv: str = "../rsna-knee-abnormality-detection/train.csv",
    train_series_csv: str = "../rsna-knee-abnormality-detection/train_series.csv",
    output_splits_path: str = "./data/splits_5fold.csv",
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    seed_everything(seed)
    console = Console()

    console.print("[bold blue]=== RSNA Knee Abnormality Detection: EDA & Split Generation ===[/bold blue]")

    df_train = pd.read_csv(train_csv)
    df_series = pd.read_csv(train_series_csv)

    console.print(f"Total Studies in train.csv: [green]{len(df_train):,}[/green]")
    console.print(f"Total Series in train_series.csv: [green]{len(df_series):,}[/green]")

    # Check labeled vs unlabeled studies
    labeled_mask = df_train[TARGET_COLUMNS].notnull().any(axis=1)
    df_labeled = df_train[labeled_mask].copy()
    df_unlabeled = df_train[~labeled_mask].copy()

    console.print(f"Studies with Ground Truth Labels: [bold green]{len(df_labeled):,}[/bold green]")
    console.print(f"Studies with Only Radiology Reports (Weak Supervision): [yellow]{len(df_unlabeled):,}[/yellow]")

    # Table of label frequencies
    table = Table(title="Abnormality Label Distribution (Ground Truth Labeled Studies)")
    table.add_column("Abnormality Target", style="cyan")
    table.add_column("Positive Count", justify="right", style="magenta")
    table.add_column("Prevalence (%)", justify="right", style="green")

    for col in TARGET_COLUMNS:
        pos_count = int(df_labeled[col].sum()) if col in df_labeled.columns else 0
        total_eval = int(df_labeled[col].count())
        prev = (pos_count / total_eval * 100) if total_eval > 0 else 0.0
        table.add_row(col, f"{pos_count:,}", f"{prev:.2f}%")

    console.print(table)

    # Perform Multilabel Stratified Split on labeled studies
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X = df_labeled["StudyInstanceUID"].values
        Y = df_labeled[TARGET_COLUMNS].fillna(0).values

        df_labeled["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(mskf.split(X, Y)):
            df_labeled.iloc[val_idx, df_labeled.columns.get_loc("fold")] = fold_idx
    except ImportError:
        # Fallback: pseudo-stratified random split
        console.print("[yellow]iterative-stratification not found, using pseudo-stratified KFold.[/yellow]")
        df_labeled["fold"] = np.random.RandomState(seed).randint(0, n_splits, size=len(df_labeled))

    # Assign unlabeled studies evenly across folds (for semi-supervised / self-supervised training)
    df_unlabeled["fold"] = np.random.RandomState(seed).randint(0, n_splits, size=len(df_unlabeled))

    # Combine back into a master splits dataframe
    df_master = pd.concat([df_labeled, df_unlabeled], ignore_index=True)
    os.makedirs(os.path.dirname(output_splits_path), exist_ok=True)
    df_master.to_csv(output_splits_path, index=False)

    console.print(f"[bold green]✓ Successfully generated and saved {n_splits}-fold splits to: {output_splits_path}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="../rsna-knee-abnormality-detection/train.csv")
    parser.add_argument("--train_series_csv", type=str, default="../rsna-knee-abnormality-detection/train_series.csv")
    parser.add_argument("--output_splits", type=str, default="./data/splits_5fold.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_eda_and_splits(
        train_csv=args.train_csv,
        train_series_csv=args.train_series_csv,
        output_splits_path=args.output_splits,
        n_splits=args.n_splits,
        seed=args.seed,
    )
