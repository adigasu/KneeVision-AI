"""
RSNA Knee Abnormality Detection - Generate Leak-Free 3-Fold Multilabel Stratified Splits.
Partitions 4,407 knee studies into 3 balanced folds:
- Gold-labeled studies (N=58) partitioned via MultilabelStratifiedKFold across all 12 pathologies.
- Weak/report-only studies (N=4,349) partitioned evenly across the 3 folds.
Output: data/splits_3fold.parquet and data/splits_3fold.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.metrics.auc_metrics import TARGET_COLUMNS
from src.utils.common import seed_everything


def generate_3fold_splits(
    train_csv: str = "../Datasets/rsna-knee-abnormality-detection/train.csv",
    output_parquet: str = "data/splits_3fold.parquet",
    output_csv: str = "data/splits_3fold.csv",
    n_splits: int = 3,
    seed: int = 42,
) -> None:
    seed_everything(seed)
    console = Console()
    console.print("[bold blue]=== RSNA Knee: Generating 3-Fold Multilabel Stratified Splits ===[/bold blue]")

    if not os.path.exists(train_csv):
        alt_csv = "../rsna-knee-abnormality-detection/train.csv"
        if os.path.exists(alt_csv):
            train_csv = alt_csv

    df_train = pd.read_csv(train_csv)
    console.print(f"Total Studies in train.csv: [green]{len(df_train):,}[/green]")

    labeled_mask = df_train[TARGET_COLUMNS].notnull().any(axis=1)
    df_labeled = df_train[labeled_mask].copy().reset_index(drop=True)
    df_unlabeled = df_train[~labeled_mask].copy().reset_index(drop=True)

    console.print(f"Gold Labeled Studies: [bold green]{len(df_labeled):,}[/bold green]")
    console.print(f"Weak/Unlabeled Studies: [yellow]{len(df_unlabeled):,}[/yellow]")

    # 1. Stratify gold studies
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df_labeled["StudyInstanceUID"].values
    Y = df_labeled[TARGET_COLUMNS].fillna(0).values

    df_labeled["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(mskf.split(X, Y)):
        df_labeled.iloc[val_idx, df_labeled.columns.get_loc("fold")] = fold_idx

    # 2. Partition unlabeled studies evenly
    rng = np.random.RandomState(seed)
    unlabeled_folds = rng.permutation(len(df_unlabeled)) % n_splits
    df_unlabeled["fold"] = unlabeled_folds

    # 3. Merge together
    df_master = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_master.to_parquet(output_parquet, index=False)
    df_master.to_csv(output_csv, index=False)

    console.print(f"[bold green]Saved 3-fold splits to:[/bold green] {output_parquet} and {output_csv}")

    # Summary table
    table = Table(title="3-Fold Distribution Summary")
    table.add_column("Fold", style="cyan")
    table.add_column("Total Studies", justify="right", style="green")
    table.add_column("Gold Studies", justify="right", style="magenta")
    table.add_column("Weak Studies", justify="right", style="yellow")

    for f in range(n_splits):
        fold_data = df_master[df_master["fold"] == f]
        gold_count = int(fold_data[TARGET_COLUMNS].notnull().any(axis=1).sum())
        weak_count = len(fold_data) - gold_count
        table.add_row(f"Fold {f}", f"{len(fold_data):,}", f"{gold_count:,}", f"{weak_count:,}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="../Datasets/rsna-knee-abnormality-detection/train.csv")
    parser.add_argument("--output_parquet", type=str, default="data/splits_3fold.parquet")
    parser.add_argument("--output_csv", type=str, default="data/splits_3fold.csv")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_3fold_splits(
        train_csv=args.train_csv,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
        n_splits=args.n_splits,
        seed=args.seed,
    )
