"""
RSNA Knee Abnormality Detection - Comprehensive Evaluation of Tracks 3, 4, 5.
Applies Track 3 (Peak Attention Blending), Track 4 (Anatomically-Safe TTA),
and Track 5 (Rank/Logit Calibration) to:
1. ConvNeXt-Tiny alone (Single Model & Pure Tiny-Family)
2. 3-Model Blend (ConvNeXt-Tiny + ConvNeXt-Small + DINOv2)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from src.data.transforms import get_validation_transforms
from src.data.triplanar_dataset import TriPlanarKneeDataset, triplanar_collate
from src.models.triplanar_mil import TriPlanarLabelSpecificMILModel

console = Console()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def evaluate_with_tta(model, loader, device):
    model.eval()
    all_preds, all_soft, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            plane_ids = batch["plane_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                # View 1: Original
                p1 = torch.sigmoid(model(imgs, plane_ids=plane_ids, mask=mask)["logits"])
                # View 2: Contrast Boost (+8%)
                imgs_boost = torch.clamp(imgs * 1.08, 0.0, 1.0)
                p2 = torch.sigmoid(model(imgs_boost, plane_ids=plane_ids, mask=mask)["logits"])
                # View 3: Contrast Dim (-8%)
                imgs_dim = torch.clamp(imgs * 0.92, 0.0, 1.0)
                p3 = torch.sigmoid(model(imgs_dim, plane_ids=plane_ids, mask=mask)["logits"])

                probs = ((p1 + p2 + p3) / 3.0).cpu().numpy()

            all_preds.append(probs)
            all_soft.append(batch["soft_targets"].numpy())
            all_is_gold.append(batch["is_gold"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_soft, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    full_metrics = compute_macro_auc(targets, preds)
    gold_macro = 0.0
    if is_gold.sum() > 5:
        gold_metrics = compute_macro_auc(targets[is_gold], preds[is_gold])
        gold_macro = gold_metrics["macro_auc"]

    return full_metrics["macro_auc"], gold_macro, preds

def main():
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║  Evaluation of Tracks 3, 4, 5 on Tiny alone & 3-Model Blend      ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")

    # 1. Load Data
    df_splits = pd.read_parquet("data/splits_5fold.parquet")
    df_dense = pd.read_parquet("data/dense_labels_master.parquet")

    df_all = df_splits[["StudyInstanceUID", "fold"]].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all["is_gold"] = labeled_mask

    dense_renamed = df_dense[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})
    df_all = df_all.merge(dense_renamed, on="StudyInstanceUID", how="left")

    val_df = df_all[df_all["fold"] == 0].reset_index(drop=True)
    targets = np.array([[float(row[f"soft_{c}"]) if f"soft_{c}" in row and pd.notnull(row[f"soft_{c}"]) else 0.50 for c in TARGET_COLUMNS] for _, row in val_df.iterrows()])
    is_gold = val_df["is_gold"].values

    console.print(f"Validation Studies: {len(val_df)} | Gold Consensus: {is_gold.sum()}")

    val_transforms = get_validation_transforms((288, 288))
    val_ds = TriPlanarKneeDataset(val_df, slices_per_plane=24, transforms=val_transforms, is_training=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=triplanar_collate)

    # 2. Load Raw Baseline Predictions
    p_tiny = np.load("checkpoints/phase13_triplanar_convnext_tiny_f0_72sl_val_preds.npy")
    p_small = np.load("checkpoints/phase13_triplanar_convnext_small_f0_72sl_val_preds.npy")
    p_dino = np.load("checkpoints/phase13_triplanar_dinov2_small_f0_72sl_val_preds.npy")
    p_intra = np.load("checkpoints/phase14_ablation_intra_transformer_convnext_tiny_f0_val_preds.npy")
    p_coat = np.load("checkpoints/phase14_ablation_coattention_convnext_tiny_f0_val_preds.npy")

    # 3. Track 4: Extract TTA predictions for Tiny, Small, DINO
    # Tiny TTA
    tiny_tta_path = "checkpoints/phase13_triplanar_convnext_tiny_f0_72sl_tta_preds.npy"
    if os.path.exists(tiny_tta_path):
        console.print(f"[green]✓ Loading existing Tiny TTA: {tiny_tta_path}[/green]")
        p_tiny_tta = np.load(tiny_tta_path)
    else:
        console.print("[yellow]Running Anatomically-Safe TTA on ConvNeXt-Tiny...[/yellow]")
        model_tiny = TriPlanarLabelSpecificMILModel(backbone_name="convnext_tiny", pretrained=False, num_classes=12, mil_hidden_dim=128, dropout=0.0).to(device)
        st = torch.load("checkpoints/phase13_triplanar_convnext_tiny_f0_72sl_best.pth", map_location=device)
        model_tiny.load_state_dict(st["model_state_dict"])
        _, _, p_tiny_tta = evaluate_with_tta(model_tiny, val_loader, device)
        np.save(tiny_tta_path, p_tiny_tta)
        console.print(f"[green]✓ Saved {tiny_tta_path}[/green]")

    # Small TTA
    small_tta_path = "checkpoints/phase13_triplanar_convnext_small_f0_72sl_tta_preds.npy"
    if os.path.exists(small_tta_path):
        console.print(f"[green]✓ Loading existing Small TTA: {small_tta_path}[/green]")
        p_small_tta = np.load(small_tta_path)
    else:
        console.print("[yellow]Running Anatomically-Safe TTA on ConvNeXt-Small...[/yellow]")
        model_small = TriPlanarLabelSpecificMILModel(backbone_name="convnext_small", pretrained=False, num_classes=12, mil_hidden_dim=128, dropout=0.0).to(device)
        st = torch.load("checkpoints/phase13_triplanar_convnext_small_f0_72sl_best.pth", map_location=device)
        model_small.load_state_dict(st["model_state_dict"])
        _, _, p_small_tta = evaluate_with_tta(model_small, val_loader, device)
        np.save(small_tta_path, p_small_tta)
        console.print(f"[green]✓ Saved {small_tta_path}[/green]")

    # DINO TTA
    dino_tta_path = "checkpoints/phase13_triplanar_dinov2_small_f0_72sl_tta_preds.npy"
    if os.path.exists(dino_tta_path):
        console.print(f"[green]✓ Loading existing DINOv2 TTA: {dino_tta_path}[/green]")
        p_dino_tta = np.load(dino_tta_path)
    else:
        console.print("[yellow]Running Anatomically-Safe TTA on DINOv2-Small...[/yellow]")
        val_transforms_dino = get_validation_transforms((280, 280))
        val_ds_dino = TriPlanarKneeDataset(val_df, slices_per_plane=24, transforms=val_transforms_dino, is_training=False)
        val_loader_dino = DataLoader(val_ds_dino, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=triplanar_collate)
        model_dino = TriPlanarLabelSpecificMILModel(backbone_name="vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=12, mil_hidden_dim=128, dropout=0.0).to(device)
        st = torch.load("checkpoints/phase13_triplanar_dinov2_small_f0_72sl_best.pth", map_location=device)
        model_dino.load_state_dict(st["model_state_dict"])
        _, _, p_dino_tta = evaluate_with_tta(model_dino, val_loader_dino, device)
        np.save(dino_tta_path, p_dino_tta)
        console.print(f"[green]✓ Saved {dino_tta_path}[/green]")

    def get_auc(preds):
        full = compute_macro_auc(targets, preds)
        gold = compute_macro_auc(targets[is_gold], preds[is_gold]) if is_gold.sum() > 5 else {"macro_auc": 0.0}
        return full["macro_auc"], gold["macro_auc"]

    # Utilities for Track 5 (Rank & Logit)
    def to_ranks(mat):
        res = np.zeros_like(mat)
        for c in range(mat.shape[1]):
            res[:, c] = rankdata(mat[:, c]) / len(mat)
        return res

    eps = 1e-6
    def to_logit(p):
        clp = np.clip(p, eps, 1 - eps)
        return np.log(clp / (1 - clp))

    # =========================================================================
    # PART A: ConvNeXt-Tiny Alone Variations
    # =========================================================================
    # 1. Baseline Tiny
    v_tiny_base, g_tiny_base = get_auc(p_tiny)
    # 2. Tiny + Track 4 (TTA)
    v_tiny_tta, g_tiny_tta = get_auc(p_tiny_tta)
    # 3. Tiny + Track 5 (Rank Percentile of Tiny alone gives identical AUC, but logit scaling tested)
    # 4. Tiny + Track 3 (Pure Tiny-Family Attention Blend: Tiny + Tiny-Intra + Tiny-CoAttention)
    p_tiny_family = 0.50 * p_tiny + 0.30 * p_intra + 0.20 * p_coat
    v_tiny_fam, g_tiny_fam = get_auc(p_tiny_family)
    # 5. Tiny-Family + Track 4 (TTA Tiny + Intra + CoAttention)
    p_tiny_fam_tta = 0.50 * p_tiny_tta + 0.30 * p_intra + 0.20 * p_coat
    v_tiny_fam_tta, g_tiny_fam_tta = get_auc(p_tiny_fam_tta)
    # 6. Tiny-Family Logit Space Blend
    l_tiny_fam = 0.50 * to_logit(p_tiny) + 0.30 * to_logit(p_intra) + 0.20 * to_logit(p_coat)
    p_tiny_fam_logit = 1 / (1 + np.exp(-l_tiny_fam))
    v_tiny_fam_logit, g_tiny_fam_logit = get_auc(p_tiny_fam_logit)

    # =========================================================================
    # PART B: 3-Model Blend Variations
    # =========================================================================
    # 1. Baseline 3-Model Blend
    p_3m_base = 0.30 * p_tiny + 0.45 * p_small + 0.25 * p_dino
    v_3m_base, g_3m_base = get_auc(p_3m_base)
    # 2. 3-Model Blend + Track 4 (TTA)
    p_3m_tta = 0.30 * p_tiny_tta + 0.45 * p_small_tta + 0.25 * p_dino_tta
    v_3m_tta, g_3m_tta = get_auc(p_3m_tta)
    # 3. 3-Model Blend + Track 5 (Logit-space)
    l_3m = 0.30 * to_logit(p_tiny) + 0.45 * to_logit(p_small) + 0.25 * to_logit(p_dino)
    p_3m_logit = 1 / (1 + np.exp(-l_3m))
    v_3m_logit, g_3m_logit = get_auc(p_3m_logit)
    # 4. 3-Model Blend + Track 5 (Rank Percentile)
    p_3m_rank = 0.30 * to_ranks(p_tiny) + 0.45 * to_ranks(p_small) + 0.25 * to_ranks(p_dino)
    v_3m_rank, g_3m_rank = get_auc(p_3m_rank)
    # 5. 3-Model + Track 3 (5-Model Full Architecture Blend)
    p_5m_base = 0.25 * p_tiny + 0.35 * p_small + 0.18 * p_dino + 0.12 * p_intra + 0.10 * p_coat
    v_5m_base, g_5m_base = get_auc(p_5m_base)
    # 6. Full Grand Ensemble: Track 3 + 4 + 5 (5-Model with TTA in Logit Space)
    l_5m_tta = 0.25 * to_logit(p_tiny_tta) + 0.35 * to_logit(p_small_tta) + 0.18 * to_logit(p_dino_tta) + 0.12 * to_logit(p_intra) + 0.10 * to_logit(p_coat)
    p_grand = 1 / (1 + np.exp(-l_5m_tta))
    v_grand, g_grand = get_auc(p_grand)

    # Output Table
    table = Table(title="Tracks 3, 4, 5 Benchmark: ConvNeXt-Tiny alone vs 3-Model Blend", header_style="bold magenta")
    table.add_column("Configuration", style="cyan", width=46)
    table.add_column("Val Macro AUC", justify="right", style="bold green")
    table.add_column("Gold Macro AUC", justify="right", style="bold yellow")
    table.add_column("Δ vs Tiny Baseline", justify="right")

    def row(name, v, g):
        diff = v - v_tiny_base
        d_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        table.add_row(name, f"{v:.4f}", f"{g:.4f}", d_str)

    table.add_row("[bold]A. ConvNeXt-Tiny Alone & Tiny-Family[/bold]", "", "", "")
    row("  1. ConvNeXt-Tiny Baseline (No TTA/Calib)", v_tiny_base, g_tiny_base)
    row("  2. ConvNeXt-Tiny + Track 4 (Anatomical TTA)", v_tiny_tta, g_tiny_tta)
    row("  3. Pure Tiny-Family + Track 3 (Tiny+Intra+CoAtt)", v_tiny_fam, g_tiny_fam)
    row("  4. Pure Tiny-Family + Track 3 + Track 4 (TTA)", v_tiny_fam_tta, g_tiny_fam_tta)
    row("  5. Pure Tiny-Family + Track 3 + Track 5 (Logit)", v_tiny_fam_logit, g_tiny_fam_logit)

    table.add_row("─" * 46, "─" * 12, "─" * 14, "─" * 18)
    table.add_row("[bold]B. Multi-Model Blend (Tiny + Small + DINO)[/bold]", "", "", "")
    row("  1. 3-Model Baseline Blend", v_3m_base, g_3m_base)
    row("  2. 3-Model + Track 4 (Anatomical TTA on all 3)", v_3m_tta, g_3m_tta)
    row("  3. 3-Model + Track 5 (Logit-Space Calibration)", v_3m_logit, g_3m_logit)
    row("  4. 3-Model + Track 5 (Rank-Average Calibration)", v_3m_rank, g_3m_rank)
    row("  5. 3-Model + Track 3 (5-Model Architecture Blend)", v_5m_base, g_5m_base)
    row("  6. Grand Ensemble (Tracks 3 + 4 + 5 Combined)", v_grand, g_grand)

    console.print(table)

if __name__ == "__main__":
    main()
