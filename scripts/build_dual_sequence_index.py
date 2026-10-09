"""
RSNA Knee Abnormality Detection - Build Dual-Sequence Manifest with Diagnostic Window Selection.
Selects primary series based on the standard clinical diagnostic window (24-38 slices)
and preserves secondary series for Multi-Series Test-Time Augmentation (TTA).
Output: data/dual_sequence_index.parquet & data/dual_sequence_index.csv.
"""

import os
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table


def score_series(row):
    """
    Scores series for standard knee MRI whole-joint diagnostic quality:
    - Ideal diagnostic 2D slice range: 24 to 38 slices (score: 90 - 100).
    - Acceptable range: 18 to 48 slices (score: 60 - 89).
    - Extreme partial/oblique scouts (<18) or 3D thin volumes (>48): score < 50.
    """
    ns = row["num_slices"]
    if 24 <= ns <= 38:
        score = 100 - abs(ns - 30)
    elif 18 <= ns <= 48:
        score = 70 - abs(ns - 30)
    else:
        score = max(0, 30 - min(abs(ns - 30), 30))
    return score


def build_dual_sequence_index(
    manifest_path: str = "data/preprocessed_index.parquet",
    output_parquet: str = "data/dual_sequence_index.parquet",
    output_csv: str = "data/dual_sequence_index.csv",
) -> None:
    console = Console()
    console.print("[bold blue]=== RSNA Knee: Building Dual-Sequence Index (Diagnostic Window Selection) ===[/bold blue]")

    df = pd.read_parquet(manifest_path)
    df["diag_score"] = df.apply(score_series, axis=1)

    # Sort primarily by diagnostic score descending, then by num_slices descending
    df_sorted = df.sort_values(by=["diag_score", "num_slices"], ascending=[False, False])

    studies = sorted(df["StudyInstanceUID"].unique())
    records = []

    # Filter into T2 and T1 subsets
    df_t2 = df_sorted[(df_sorted["Fluid_Sensitive"] == 1) & (df_sorted["Fat_Suppression"] == 1)]
    df_t1 = df_sorted[(df_sorted["Fluid_Sensitive"] == 0) & (df_sorted["Fat_Suppression"] == 0)]

    def extract_primary_and_secondary(sub_df):
        prim = {}
        sec = {}
        for (s_uid, plane), group in sub_df.groupby(["StudyInstanceUID", "Anatomical_Plane"]):
            prim[(s_uid, plane)] = group.iloc[0]["cache_path"]
            if len(group) > 1:
                sec[(s_uid, plane)] = group.iloc[1]["cache_path"]
            else:
                sec[(s_uid, plane)] = None
        return prim, sec

    t2_prim, t2_sec = extract_primary_and_secondary(df_t2)
    t1_prim, t1_sec = extract_primary_and_secondary(df_t1)

    for s_uid in studies:
        t2_sag = t2_prim.get((s_uid, "Sagittal"), None)
        t2_cor = t2_prim.get((s_uid, "Coronal"), None)
        t2_ax  = t2_prim.get((s_uid, "Axial"), None)

        t1_sag = t1_prim.get((s_uid, "Sagittal"), None)
        t1_cor = t1_prim.get((s_uid, "Coronal"), None)
        t1_ax  = t1_prim.get((s_uid, "Axial"), None)

        # Secondary series for Multi-Series TTA
        t2_sag_sec = t2_sec.get((s_uid, "Sagittal"), None)
        t2_cor_sec = t2_sec.get((s_uid, "Coronal"), None)
        t2_ax_sec  = t2_sec.get((s_uid, "Axial"), None)

        t1_sag_sec = t1_sec.get((s_uid, "Sagittal"), None)
        t1_cor_sec = t1_sec.get((s_uid, "Coronal"), None)
        t1_ax_sec  = t1_sec.get((s_uid, "Axial"), None)

        records.append({
            "StudyInstanceUID": s_uid,
            # Primary series paths (used for training)
            "t2_sag_path": t2_sag,
            "t2_cor_path": t2_cor,
            "t2_ax_path": t2_ax,
            "t1_sag_path": t1_sag,
            "t1_cor_path": t1_cor,
            "t1_ax_path": t1_ax,
            # Secondary series paths (used for Multi-Series TTA during inference)
            "t2_sag_sec_path": t2_sag_sec,
            "t2_cor_sec_path": t2_cor_sec,
            "t2_ax_sec_path": t2_ax_sec,
            "t1_sag_sec_path": t1_sag_sec,
            "t1_cor_sec_path": t1_cor_sec,
            "t1_ax_sec_path": t1_ax_sec,
            # Availability flags
            "has_t2_sag": bool(t2_sag is not None),
            "has_t2_cor": bool(t2_cor is not None),
            "has_t2_ax": bool(t2_ax is not None),
            "has_t1_sag": bool(t1_sag is not None),
            "has_t1_cor": bool(t1_cor is not None),
            "has_t1_ax": bool(t1_ax is not None),
            "has_any_secondary": any([
                t2_sag_sec is not None, t2_cor_sec is not None, t2_ax_sec is not None,
                t1_sag_sec is not None, t1_cor_sec is not None, t1_ax_sec is not None,
            ]),
        })

    dual_df = pd.DataFrame(records)
    dual_df.to_parquet(output_parquet, index=False)
    dual_df.to_csv(output_csv, index=False)

    console.print(f"[bold green]Saved updated index to:[/bold green] {output_parquet} and {output_csv}")

    # Summary Table
    table = Table(title="Sequence Coverage & Multi-Series TTA Candidates (4,407 Studies)")
    table.add_column("Plane", style="cyan")
    table.add_column("T2 Primary", justify="right", style="green")
    table.add_column("T2 Secondary (TTA)", justify="right", style="yellow")
    table.add_column("T1 Primary", justify="right", style="magenta")
    table.add_column("T1 Secondary (TTA)", justify="right", style="yellow")

    for plane in ["Sagittal", "Coronal", "Axial"]:
        p_lower = "ax" if plane == "Axial" else plane.lower()[:3]
        t2_cov = dual_df[f"has_t2_{p_lower}"].sum()
        t2_sec_cov = dual_df[f"t2_{p_lower}_sec_path"].notnull().sum()
        t1_cov = dual_df[f"has_t1_{p_lower}"].sum()
        t1_sec_cov = dual_df[f"t1_{p_lower}_sec_path"].notnull().sum()
        table.add_row(
            plane,
            f"{t2_cov:,} ({t2_cov/len(dual_df)*100:.1f}%)",
            f"{t2_sec_cov:,} ({t2_sec_cov/len(dual_df)*100:.1f}%)",
            f"{t1_cov:,} ({t1_cov/len(dual_df)*100:.1f}%)",
            f"{t1_sec_cov:,} ({t1_sec_cov/len(dual_df)*100:.1f}%)",
        )

    console.print(table)
    console.print(f"Studies with at least one Secondary series for TTA: [bold green]{dual_df['has_any_secondary'].sum():,} ({dual_df['has_any_secondary'].sum()/len(dual_df)*100:.1f}%)[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="data/preprocessed_index.parquet")
    parser.add_argument("--output_parquet", type=str, default="data/dual_sequence_index.parquet")
    parser.add_argument("--output_csv", type=str, default="data/dual_sequence_index.csv")
    args = parser.parse_args()

    build_dual_sequence_index(
        manifest_path=args.manifest_path,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
    )
