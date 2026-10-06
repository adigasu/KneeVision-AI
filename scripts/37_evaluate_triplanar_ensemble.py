import numpy as np
import pandas as pd
from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from rich.console import Console
from rich.table import Table

console = Console()

# 1. Load fold 0 ground truth
df_splits = pd.read_parquet("data/splits_5fold.parquet")
df_tri = pd.read_parquet("data/tristate_labels_master.parquet")
df_dense = pd.read_parquet("data/dense_labels_master.parquet")

df_all = df_splits[["StudyInstanceUID", "fold"]].copy()
labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
df_all["is_gold"] = labeled_mask

tri_renamed = df_tri[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"hard_{c}" for c in TARGET_COLUMNS})
dense_renamed = df_dense[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})
df_all = df_all.merge(tri_renamed, on="StudyInstanceUID", how="left")
df_all = df_all.merge(dense_renamed, on="StudyInstanceUID", how="left")

val_df = df_all[df_all["fold"] == 0].reset_index(drop=True)
targets = np.array([[float(row[f"soft_{c}"]) if f"soft_{c}" in row and pd.notnull(row[f"soft_{c}"]) else 0.50 for c in TARGET_COLUMNS] for _, row in val_df.iterrows()])
is_gold = val_df["is_gold"].values

# 2. Load out-of-fold validation predictions
p_tiny = np.load("checkpoints/phase13_triplanar_convnext_tiny_f0_72sl_val_preds.npy")
p_small = np.load("checkpoints/phase13_triplanar_convnext_small_f0_72sl_val_preds.npy")
p_dino = np.load("checkpoints/phase13_triplanar_dinov2_small_f0_72sl_val_preds.npy")

console.print(f"[bold cyan]Validation Studies: {len(targets)} | Gold Consensus: {is_gold.sum()}[/bold cyan]\n")

def get_metrics(preds):
    full = compute_macro_auc(targets, preds)
    gold = compute_macro_auc(targets[is_gold], preds[is_gold]) if is_gold.sum() > 5 else {"macro_auc": 0.0}
    return full["macro_auc"], gold["macro_auc"], full["per_class_auc"]

m_tiny, g_tiny, c_tiny = get_metrics(p_tiny)
m_small, g_small, c_small = get_metrics(p_small)
m_dino, g_dino, c_dino = get_metrics(p_dino)

# 2-Model Ensembles
p_2m_eq = 0.5 * p_tiny + 0.5 * p_small
m_2m, g_2m, c_2m = get_metrics(p_2m_eq)

# 3-Model Equal
p_3m_eq = (p_tiny + p_small + p_dino) / 3.0
m_3m_eq, g_3m_eq, c_3m_eq = get_metrics(p_3m_eq)

# 3-Model Optimized Grid Search
best_auc = 0.0
best_w = (1/3, 1/3, 1/3)
for w1 in np.linspace(0.1, 0.8, 15):
    for w2 in np.linspace(0.1, 0.8, 15):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * p_tiny + w2 * p_small + w3 * p_dino
        auc, _, _ = get_metrics(blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)

p_3m_opt = best_w[0] * p_tiny + best_w[1] * p_small + best_w[2] * p_dino
m_3m_opt, g_3m_opt, c_3m_opt = get_metrics(p_3m_opt)

# Summary Table
table = Table(title="Phase 13 Tri-Planar: Individual & Ensemble Validation Macro AUC", header_style="bold magenta")
table.add_column("Model / Configuration", style="bold")
table.add_column("Val Macro AUC", justify="right", style="green")
table.add_column("Gold Macro AUC", justify="right", style="cyan")
table.add_column("Gain vs Tiny Baseline", justify="right", style="yellow")

table.add_row("ConvNeXt-Tiny (F0, 72sl)", f"{m_tiny:.4f}", f"{g_tiny:.4f}", "-")
table.add_row("ConvNeXt-Small (F0, 72sl)", f"{m_small:.4f}", f"{g_small:.4f}", f"+{m_small - m_tiny:.4f}")
table.add_row("DINOv2-Small (F0, 72sl)", f"{m_dino:.4f}", f"{g_dino:.4f}", f"{m_dino - m_tiny:+.4f}")
table.add_row("─" * 30, "─" * 12, "─" * 12, "─" * 15)
table.add_row("2-Model Equal (Tiny + Small)", f"{m_2m:.4f}", f"{g_2m:.4f}", f"+{m_2m - m_tiny:.4f}")
table.add_row("3-Model Equal (Tiny + Small + DINOv2)", f"{m_3m_eq:.4f}", f"{g_3m_eq:.4f}", f"+{m_3m_eq - m_tiny:.4f}")
table.add_row(
    f"3-Model Optimized (w: {best_w[0]:.2f}/{best_w[1]:.2f}/{best_w[2]:.2f})",
    f"[bold]{m_3m_opt:.4f}[/bold]",
    f"[bold]{g_3m_opt:.4f}[/bold]",
    f"[bold]+{m_3m_opt - m_tiny:.4f}[/bold]",
)

console.print(table)

# Per-Class Table comparing Best Individual vs Best Ensemble
c_table = Table(title="Per-Class AUC: Single Model vs 3-Model Tri-Planar Ensemble", header_style="bold blue")
c_table.add_column("Pathology", style="bold")
c_table.add_column("ConvNeXt-Tiny", justify="right")
c_table.add_column("ConvNeXt-Small", justify="right")
c_table.add_column("DINOv2-Small", justify="right")
c_table.add_column("3-Model Blend", justify="right", style="bold green")

for col in TARGET_COLUMNS:
    c_table.add_row(
        col,
        f"{c_tiny.get(col, 0):.3f}",
        f"{c_small.get(col, 0):.3f}",
        f"{c_dino.get(col, 0):.3f}",
        f"{c_3m_opt.get(col, 0):.3f}",
    )

console.print(c_table)

# Save ensemble predictions
np.save("checkpoints/phase13_triplanar_3model_opt_val_preds.npy", p_3m_opt)
np.save("checkpoints/phase13_triplanar_3model_equal_val_preds.npy", p_3m_eq)
print(f"Optimal Weights: ConvNeXt-Tiny={best_w[0]:.3f}, ConvNeXt-Small={best_w[1]:.3f}, DINOv2-Small={best_w[2]:.3f}")
