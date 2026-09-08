"""
RSNA Knee Abnormality Detection - Generate Multilingual Dense Labels (v2).
Applies the multilingual dense semantic extractor across all 4,407 radiology reports,
preserves the 58 gold human-labeled ground-truth targets, evaluates label quality vs gold,
and writes clean master datasets.
"""

import os
import argparse
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.nlp.llm_dense_extractor import MultilingualDenseNLPExtractor


def generate_dense_labels(
    train_csv: str = "../Datasets/rsna-knee-abnormality-detection/train.csv",
    splits_path: str = "./data/splits_5fold.parquet",
    output_csv: str = "./data/dense_labels_master.csv",
    output_parquet: str = "./data/dense_labels_master.parquet",
):
    console = Console()
    console.print("[bold blue]=== RSNA Knee: Generating Multilingual Dense [0.0, 1.0] Targets (v2) ===[/bold blue]")

    df_train = pd.read_csv(train_csv)
    console.print(f"Loaded train dataset: [green]{len(df_train):,}[/green] total studies.")

    # Load fold splits
    if os.path.exists(splits_path):
        df_splits = pd.read_parquet(splits_path) if splits_path.endswith(".parquet") else pd.read_csv(splits_path)
        if "fold" in df_splits.columns and "StudyInstanceUID" in df_splits.columns:
            df_train = df_train.merge(df_splits[["StudyInstanceUID", "fold"]], on="StudyInstanceUID", how="left")
    elif os.path.exists("./data/splits_5fold.csv"):
        df_splits = pd.read_csv("./data/splits_5fold.csv")
        df_train = df_train.merge(df_splits[["StudyInstanceUID", "fold"]], on="StudyInstanceUID", how="left")

    if "fold" not in df_train.columns:
        df_train["fold"] = np.random.randint(0, 5, size=len(df_train))

    gold_mask = df_train[TARGET_COLUMNS].notnull().all(axis=1)
    gold_df = df_train[gold_mask].copy()
    console.print(f"Identified [bold green]{len(gold_df)}[/bold green] Gold Human-Annotated Studies.")

    extractor = MultilingualDenseNLPExtractor()
    dense_records = []
    nlp_predictions_for_gold = []

    for idx, row in df_train.iterrows():
        study_id = row["StudyInstanceUID"]
        report = row.get("Report", "")
        fold = row.get("fold", 0)
        is_gold_case = bool(gold_mask.iloc[idx])

        # Always extract NLP probabilities from text
        probs = extractor.extract_probabilities(report)

        if is_gold_case:
            nlp_predictions_for_gold.append([probs[col] for col in TARGET_COLUMNS])

        record = {
            "StudyInstanceUID": study_id,
            "fold": int(fold),
            "is_gold": is_gold_case,
            "Report": report,
        }

        if is_gold_case:
            for col in TARGET_COLUMNS:
                record[col] = float(row[col])
        else:
            for col in TARGET_COLUMNS:
                record[col] = float(probs[col])

        dense_records.append(record)

    df_dense = pd.DataFrame(dense_records)

    # Measure Label Matrix Quality vs the 58 Gold Studies
    if len(nlp_predictions_for_gold) > 0:
        gold_targets = gold_df[TARGET_COLUMNS].values
        gold_nlp_preds = np.array(nlp_predictions_for_gold)
        label_eval = compute_macro_auc(y_true=gold_targets, y_pred=gold_nlp_preds)
        console.print(f"\n[bold yellow]⭐ Label Extractor Macro AUC vs 58 Gold Annotations: {label_eval['macro_auc']:.4f}[/bold yellow]\n")

        eval_table = Table(title="Label Key vs Gold Human Benchmark Per-Class Breakdown")
        eval_table.add_column("Abnormality Finding", style="cyan")
        eval_table.add_column("Key vs Gold AUC", justify="right", style="yellow")
        for col in TARGET_COLUMNS:
            auc = label_eval["per_class_auc"].get(col, np.nan)
            eval_table.add_row(col, f"{auc:.4f}" if not np.isnan(auc) else "N/A")
        console.print(eval_table)

    # Save to disk
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_dense.to_csv(output_csv, index=False)
    df_dense.to_parquet(output_parquet, index=False)
    console.print(f"\n[bold green]✓ Master Dense Labels saved to:[/bold green]\n  - {output_csv}\n  - {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="../Datasets/rsna-knee-abnormality-detection/train.csv")
    parser.add_argument("--splits_path", type=str, default="./data/splits_5fold.parquet")
    parser.add_argument("--output_csv", type=str, default="./data/dense_labels_master.csv")
    parser.add_argument("--output_parquet", type=str, default="./data/dense_labels_master.parquet")
    args = parser.parse_args()

    generate_dense_labels(
        train_csv=args.train_csv,
        splits_path=args.splits_path,
        output_csv=args.output_csv,
        output_parquet=args.output_parquet,
    )
