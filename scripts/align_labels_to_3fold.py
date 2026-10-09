"""
RSNA Knee Abnormality Detection - Align All Label Masters to New 3-Fold Splits.
Updates 'fold' column across:
- data/dense_labels_master.parquet & .csv
- data/tristate_labels_master.parquet & .csv
- data/dense_labels_master_v3_cleaned.parquet & .csv
to perfectly match data/splits_3fold.parquet (Folds 0, 1, 2).
"""

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
console.print("[bold blue]=== RSNA Knee: Aligning Label Masters to New 3-Fold Splits ===[/bold blue]")

splits_3f = pd.read_parquet("data/splits_3fold.parquet")
fold_map = splits_3f.set_index("StudyInstanceUID")["fold"].to_dict()

files_to_update = [
    "data/dense_labels_master",
    "data/tristate_labels_master",
    "data/dense_labels_master_v3_cleaned",
]

table = Table(title="Label Masters 3-Fold Realignment Summary")
table.add_column("Dataset", style="cyan")
table.add_column("Total Rows", justify="right", style="green")
table.add_column("Fold 0", justify="right", style="magenta")
table.add_column("Fold 1", justify="right", style="magenta")
table.add_column("Fold 2", justify="right", style="magenta")

for base in files_to_update:
    p_pq = f"{base}.parquet"
    p_csv = f"{base}.csv"

    df = pd.read_parquet(p_pq)
    df["fold"] = df["StudyInstanceUID"].map(fold_map)

    # Save aligned files
    df.to_parquet(p_pq, index=False)
    df.to_csv(p_csv, index=False)

    counts = df["fold"].value_counts().to_dict()
    table.add_row(
        base,
        f"{len(df):,}",
        f"{counts.get(0, 0):,}",
        f"{counts.get(1, 0):,}",
        f"{counts.get(2, 0):,}",
    )

console.print(table)
console.print("[bold green]✓ All label masters successfully aligned to the New 3-Fold Splits![/bold green]")
