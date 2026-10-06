"""
RSNA Knee Abnormality Detection - Evaluate Peak Attention Checkpoints & Ensembling.
Extracts validation predictions from Phase 14 Arm 3 (Intra-Transformer) & Arm 4 (Co-Attention)
and tests multi-model ensembling with Phase 13 backbones (Tiny, Small, DINOv2).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from rich.console import Console
from rich.table import Table

import importlib.util
spec = importlib.util.spec_from_file_location("train_ablation", "scripts/39_train_ablation.py")
_mod = importlib.util.module_from_spec(spec)
sys.modules["train_ablation"] = _mod
spec.loader.exec_module(_mod)
TriPlanarKneeDataset = _mod.TriPlanarKneeDataset
AblationTriPlanarMILModel = _mod.AblationTriPlanarMILModel
triplanar_collate = _mod.triplanar_collate
get_validation_transforms = _mod.get_validation_transforms
evaluate = _mod.evaluate
from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc

console = Console()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║  Stage 1: Peak Attention Checkpoints Extraction & Ensembling     ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")

    # 1. Load Data
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

    console.print(f"Validation Studies: {len(val_df)} | Gold Studies: {is_gold.sum()}")

    val_transforms = get_validation_transforms((288, 288))
    val_ds = TriPlanarKneeDataset(
        val_df,
        slices_per_plane=24,
        transforms=val_transforms,
        is_training=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=triplanar_collate,
    )

    # 2. Extract Arm 3 (Intra-Transformer) Predictions
    intra_pred_file = "checkpoints/phase14_ablation_intra_transformer_convnext_tiny_f0_val_preds.npy"
    if os.path.exists(intra_pred_file):
        console.print(f"[green]✓ Loading existing: {intra_pred_file}[/green]")
        p_intra = np.load(intra_pred_file)
    else:
        console.print("[yellow]Extracting Intra-Transformer validation predictions...[/yellow]")
        intra_ckpt = "checkpoints/phase14_ablation_intra_transformer_convnext_tiny_f0_best.pth"
        model_intra = AblationTriPlanarMILModel(
            backbone_name="convnext_tiny",
            pretrained=False,
            num_classes=12,
            mil_hidden_dim=128,
            dropout=0.0,
            chunk_size=32,
            slices_per_plane=24,
            use_intra_transformer=True,
            use_coattention=False,
        ).to(device)
        state = torch.load(intra_ckpt, map_location=device)
        model_intra.load_state_dict(state["model_state_dict"])
        _, _, _, p_intra = evaluate(model_intra, val_loader, device)
        np.save(intra_pred_file, p_intra)
        console.print(f"[green]✓ Saved {intra_pred_file}[/green]")

    # 3. Extract Arm 4 (Co-Attention) Predictions
    coat_pred_file = "checkpoints/phase14_ablation_coattention_convnext_tiny_f0_val_preds.npy"
    if os.path.exists(coat_pred_file):
        console.print(f"[green]✓ Loading existing: {coat_pred_file}[/green]")
        p_coat = np.load(coat_pred_file)
    else:
        console.print("[yellow]Extracting Co-Attention validation predictions...[/yellow]")
        coat_ckpt = "checkpoints/phase14_ablation_coattention_convnext_tiny_f0_best.pth"
        model_coat = AblationTriPlanarMILModel(
            backbone_name="convnext_tiny",
            pretrained=False,
            num_classes=12,
            mil_hidden_dim=128,
            dropout=0.0,
            chunk_size=32,
            slices_per_plane=24,
            use_intra_transformer=False,
            use_coattention=True,
        ).to(device)
        state = torch.load(coat_ckpt, map_location=device)
        model_coat.load_state_dict(state["model_state_dict"])
        _, _, _, p_coat = evaluate(model_coat, val_loader, device)
        np.save(coat_pred_file, p_coat)
        console.print(f"[green]✓ Saved {coat_pred_file}[/green]")

    # 4. Load Phase 13 Predictions
    p_tiny = np.load("checkpoints/phase13_triplanar_convnext_tiny_f0_72sl_val_preds.npy")
    p_small = np.load("checkpoints/phase13_triplanar_convnext_small_f0_72sl_val_preds.npy")
    p_dino = np.load("checkpoints/phase13_triplanar_dinov2_small_f0_72sl_val_preds.npy")

    def calc_metrics(preds):
        full = compute_macro_auc(targets, preds)
        gold = compute_macro_auc(targets[is_gold], preds[is_gold]) if is_gold.sum() > 5 else {"macro_auc": 0.0}
        return full["macro_auc"], gold["macro_auc"], full["per_class_auc"]

    # Individual Metrics
    models = {
        "ConvNeXt-Tiny (F0 Baseline)": p_tiny,
        "ConvNeXt-Small (F0)": p_small,
        "DINOv2-Small (F0)": p_dino,
        "Arm 3: Intra-Transformer (Peak)": p_intra,
        "Arm 4: Co-Attention (Peak)": p_coat,
    }

    # Ensembles
    # A. 3-Model Phase 13 Baseline Blend
    p_3m_opt = 0.30 * p_tiny + 0.45 * p_small + 0.25 * p_dino
    # B. 4-Model Blend (+ Intra-Transformer)
    p_4m = 0.28 * p_tiny + 0.38 * p_small + 0.20 * p_dino + 0.14 * p_intra
    # C. 5-Model Full Architecture Blend (+ Intra + Co-Attention)
    p_5m = 0.25 * p_tiny + 0.35 * p_small + 0.18 * p_dino + 0.12 * p_intra + 0.10 * p_coat

    # Rank Blending for 5-Model
    def to_ranks(mat):
        res = np.zeros_like(mat)
        for c in range(mat.shape[1]):
            res[:, c] = rankdata(mat[:, c]) / len(mat)
        return res

    p_5m_rank = 0.25 * to_ranks(p_tiny) + 0.35 * to_ranks(p_small) + 0.18 * to_ranks(p_dino) + 0.12 * to_ranks(p_intra) + 0.10 * to_ranks(p_coat)

    # Logit Blending for 5-Model
    eps = 1e-6
    def to_logit(p):
        clp = np.clip(p, eps, 1 - eps)
        return np.log(clp / (1 - clp))
    l_5m = 0.25 * to_logit(p_tiny) + 0.35 * to_logit(p_small) + 0.18 * to_logit(p_dino) + 0.12 * to_logit(p_intra) + 0.10 * to_logit(p_coat)
    p_5m_logit = 1 / (1 + np.exp(-l_5m))

    all_evals = {
        **models,
        "─── Ensembles ───": None,
        "Phase 13 3-Model Blend (Tiny+Small+DINO)": p_3m_opt,
        "Phase 14 4-Model Blend (+ Intra-Transformer)": p_4m,
        "Phase 14 5-Model Blend (+ Intra + Co-Attention)": p_5m,
        "Phase 14 5-Model Logit-Space Blend": p_5m_logit,
        "Phase 14 5-Model Rank-Average Blend": p_5m_rank,
    }

    table = Table(title="Phase 14 Advanced Ensemble Benchmarks on Fold 0", header_style="bold magenta")
    table.add_column("Configuration", style="cyan", width=42)
    table.add_column("Val Macro AUC", justify="right", style="bold green")
    table.add_column("Gold Macro AUC", justify="right", style="bold yellow")
    table.add_column("Gain vs Baseline", justify="right")

    base_val = compute_macro_auc(targets, p_tiny)["macro_auc"]

    for name, p in all_evals.items():
        if p is None:
            table.add_row("─" * 40, "─" * 12, "─" * 12, "─" * 15)
            continue
        v_m, g_m, _ = calc_metrics(p)
        delta = v_m - base_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        table.add_row(name, f"{v_m:.4f}", f"{g_m:.4f}", delta_str)

    console.print(table)

if __name__ == "__main__":
    main()
