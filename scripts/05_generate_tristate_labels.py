"""
RSNA Knee Abnormality Detection - Generate Multilingual Tri-State Labels.
Extracts +1.0 (positive), 0.0 (negated/normal), and NaN (unmentioned) labels
across all 4,407 radiology reports (Spanish, Greek, Cyrillic, English).
Validates accuracy against the 58 gold ground-truth studies.
"""

import os
import argparse
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.nlp.multilingual_extractor import MultilingualKneeNLPExtractor
from src.metrics.auc_metrics import TARGET_COLUMNS


def detect_script(text: str) -> str:
    if not isinstance(text, str):
        return "EMPTY"
    scripts = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "GREEK" in name:
            return "Greek"
        elif "CYRILLIC" in name:
            return "Cyrillic"
    return "Latin"


def run_extraction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../Datasets/rsna-knee-abnormality-detection/train.csv")
    parser.add_argument("--output_parquet", type=str, default="data/tristate_labels_master.parquet")
    parser.add_argument("--output_csv", type=str, default="data/tristate_labels_master.csv")
    args = parser.parse_args()

    console = Console()
    console.print("[bold green]=== RSNA Knee: Multilingual Tri-State Label Extraction ===[/bold green]")

    df = pd.read_csv(args.train_path)
    console.print(f"Total Studies to Process: [cyan]{len(df):,}[/cyan]")

    extractor = MultilingualKneeNLPExtractor()

    records = []
    script_types = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Multilingual Labels"):
        s_uid = row["StudyInstanceUID"]
        report = row.get("Report", "")
        script = detect_script(report)
        script_types.append(script)

        # Extract tri-state findings
        findings = extractor.extract_from_report(report)
        findings["StudyInstanceUID"] = s_uid
        findings["Script"] = script
        records.append(findings)

    df_tristate = pd.DataFrame(records)

    # Validate against the 58 Gold Studies
    gold_mask = df[TARGET_COLUMNS].notnull().any(axis=1)
    df_gold = df[gold_mask].copy()
    gold_tristate = df_tristate[df_tristate["StudyInstanceUID"].isin(df_gold["StudyInstanceUID"])].copy()

    # Compare Gold Ground-Truth vs Extracted
    gold_merged = pd.merge(df_gold[["StudyInstanceUID"] + TARGET_COLUMNS], gold_tristate, on="StudyInstanceUID", suffixes=("_gt", "_ext"))

    correct_matches = 0
    total_evaluated = 0

    for col in TARGET_COLUMNS:
        y_true = gold_merged[f"{col}_gt"].values
        y_ext = gold_merged[f"{col}_ext"].values
        valid = ~np.isnan(y_ext)
        correct_matches += np.sum(y_true[valid] == (y_ext[valid] == 1.0))
        total_evaluated += np.sum(valid)

    accuracy = correct_matches / max(1, total_evaluated)
    console.print(f"\n[bold green]Gold Set Alignment Accuracy: {accuracy * 100:.2f}% on {total_evaluated} evaluated pairs[/bold green]")

    # Summary Statistics Table across full 4,407 dataset
    table = Table(title="Multilingual Tri-State Label Distribution across 4,407 Studies")
    table.add_column("Target Pathology", style="cyan")
    table.add_column("Positive (+1.0)", justify="right", style="green")
    table.add_column("Normal/Negated (0.0)", justify="right", style="blue")
    table.add_column("Unmentioned (NaN)", justify="right", style="yellow")
    table.add_column("Positive % (Observed)", justify="right", style="magenta")

    for col in TARGET_COLUMNS:
        n_pos = (df_tristate[col] == 1.0).sum()
        n_neg = (df_tristate[col] == 0.0).sum()
        n_nan = df_tristate[col].isna().sum()
        obs_total = n_pos + n_neg
        pos_rate = (n_pos / obs_total * 100) if obs_total > 0 else 0.0
        table.add_row(col, f"{n_pos:,}", f"{n_neg:,}", f"{n_nan:,}", f"{pos_rate:.1f}%")

    console.print(table)

    # Script Distribution Table
    script_table = Table(title="Language / Script Extraction Coverage")
    script_table.add_column("Script / Language", style="cyan")
    script_table.add_column("Studies", justify="right", style="magenta")
    script_table.add_column("Avg Positive Targets / Study", justify="right", style="green")
    script_table.add_column("Avg Mentioned Targets / Study", justify="right", style="yellow")

    for s_name in ["Latin", "Greek", "Cyrillic"]:
        sub = df_tristate[df_tristate["Script"] == s_name]
        if len(sub) > 0:
            avg_pos = (sub[TARGET_COLUMNS] == 1.0).sum(axis=1).mean()
            avg_obs = (sub[TARGET_COLUMNS].notnull()).sum(axis=1).mean()
            script_table.add_row(s_name, f"{len(sub):,}", f"{avg_pos:.2f}", f"{avg_obs:.2f}")

    console.print(script_table)

    # Merge with manual gold labels to preserve ground-truth
    final_master = df_tristate.copy()
    for _, row in df_gold.iterrows():
        s_uid = row["StudyInstanceUID"]
        match_idx = final_master[final_master["StudyInstanceUID"] == s_uid].index
        if len(match_idx) > 0:
            for col in TARGET_COLUMNS:
                if pd.notnull(row[col]):
                    final_master.loc[match_idx, col] = float(row[col])

    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    final_master.to_parquet(args.output_parquet, engine="pyarrow", compression="snappy", index=False)
    final_master.to_csv(args.output_csv, index=False)
    console.print(f"\n✓ Saved Tri-State Master Labels to [bold green]{args.output_parquet}[/bold green] and [bold green]{args.output_csv}[/bold green]")


if __name__ == "__main__":
    run_extraction()
