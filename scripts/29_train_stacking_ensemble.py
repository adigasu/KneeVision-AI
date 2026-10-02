"""
RSNA Knee Abnormality Detection - Phase 11 Stacking Meta-Learner (Fold 0)
Loads pre-trained Fold 0 models (ConvNeXt-Tiny, ConvNeXt-Small, DINOv2-Small),
extracts validation predictions, and optimizes per-pathology weights
using the Consensus-Denoised Loss (2.0x Gold Weighting + Noise Mitigation).
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table

from src.data.mri_aware_dataset import MRIAwareKneeDataset, mri_aware_collate
from src.data.transforms import get_validation_transforms
from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
import timm

console = Console()

class LabelSpecificKneeMILModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', pretrained=False, num_classes=12, mil_hidden_dim=128, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        extra_kwargs = {}
        if 'dinov2' in backbone_name.lower():
            extra_kwargs = {'img_size': 336, 'dynamic_img_size': True}
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=3, **extra_kwargs)
        self.num_features = self.backbone.num_features
        self.mil_pool = LabelSpecificGatedAttentionMILPool(in_features=self.num_features, num_classes=num_classes, hidden_dim=mil_hidden_dim, dropout=dropout)

    def forward(self, images):
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        feats_flat = self.backbone(x_flat)
        slice_feats = feats_flat.view(B, D, self.num_features)
        res = self.mil_pool(slice_feats)
        logits = res[0] if isinstance(res, (tuple, list)) else res
        return logits

def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    st = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(st)
    return model.to(device).eval().half()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]")
    console.print(f"[bold green]║ Phase 11: STACKING META-LEARNER (DENOISED LOSS) ON FOLD {args.fold}     ║[/bold green]")
    console.print(f"[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]")

    # 1. Dataset setup
    df_splits = pd.read_parquet('data/splits_5fold.parquet')
    df_tri = pd.read_parquet('data/tristate_labels_master.parquet')
    df_dense = pd.read_parquet('data/dense_labels_master.parquet')

    df_all = df_splits[['StudyInstanceUID', 'fold']].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all['is_gold'] = labeled_mask

    tri_renamed = df_tri[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'hard_{c}' for c in TARGET_COLUMNS})
    dense_renamed = df_dense[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'soft_{c}' for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on='StudyInstanceUID', how='left').merge(dense_renamed, on='StudyInstanceUID', how='left')

    val_df = df_all[df_all['fold'] == args.fold].reset_index(drop=True)
    val_transforms = get_validation_transforms((288, 288))
    val_ds = MRIAwareKneeDataset(val_df, target_slices=32, transforms=val_transforms, is_training=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)

    # 2. Load the 3 models
    ckpts = {
        'tiny': 'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_tiny_mri_aware_fold0_best.pth',
        'small': 'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_small_mri_aware_fold0_best.pth',
        'dinov2': 'kaggle_upload/datasets/phase_11_tri_ensemble/dinov2_small_mri_aware_fold0_best.pth',
    }

    console.print("[cyan]Loading Phase 11 best checkpoints (Tiny + Small + DINOv2)...[/cyan]")
    m_tiny = load_ckpt(LabelSpecificKneeMILModel('convnext_tiny'), ckpts['tiny'], device)
    m_small = load_ckpt(LabelSpecificKneeMILModel('convnext_small'), ckpts['small'], device)
    m_dino = load_ckpt(LabelSpecificKneeMILModel('vit_small_patch14_dinov2.lvd142m'), ckpts['dinov2'], device)

    # 3. Extract validation predictions
    console.print(f"Extracting validation predictions on {len(val_df)} studies...")
    preds_tiny, preds_small, preds_dino, targets, soft_list, is_gold_list = [], [], [], [], [], []

    with torch.inference_mode():
        for batch in val_loader:
            vols = batch['images'].to(device).half()
            B, D, C, H, W = vols.shape
            vols_dino = F.interpolate(vols.view(B * D, C, H, W), size=(280, 280), mode='bilinear', align_corners=False).view(B, D, C, 280, 280)

            with torch.amp.autocast('cuda'):
                p_t = torch.sigmoid(m_tiny(vols)['logits']).cpu().float().numpy()
                p_s = torch.sigmoid(m_small(vols)['logits']).cpu().float().numpy()
                p_d = torch.sigmoid(m_dino(vols_dino)['logits']).cpu().float().numpy()

            preds_tiny.append(p_t)
            preds_small.append(p_s)
            preds_dino.append(p_d)
            targets.append(batch['hard_labels'].numpy())
            soft_list.append(batch['soft_labels'].numpy())
            is_gold_list.append(batch['is_gold'].numpy())

    preds_tiny = np.concatenate(preds_tiny, axis=0)
    preds_small = np.concatenate(preds_small, axis=0)
    preds_dino = np.concatenate(preds_dino, axis=0)
    targets = np.concatenate(targets, axis=0)
    soft_targets = np.concatenate(soft_list, axis=0)
    is_gold = np.concatenate(is_gold_list, axis=0)

    # Benchmark individual models
    auc_tiny = compute_macro_auc(targets, preds_tiny)
    auc_small = compute_macro_auc(targets, preds_small)
    auc_dino = compute_macro_auc(targets, preds_dino)

    # Fixed blend: 0.50 / 0.30 / 0.20
    preds_fixed = 0.50 * preds_tiny + 0.30 * preds_small + 0.20 * preds_dino
    auc_fixed = compute_macro_auc(targets, preds_fixed)

    # 4. Optimize per-pathology weights using Consensus-Denoised Objective
    console.print("\n[bold yellow]Optimizing per-pathology weights using Consensus-Denoised Loss (2.0x Gold Weight)...[/bold yellow]")
    optimal_weights = np.zeros((12, 3))
    preds_opt = np.zeros_like(preds_tiny)

    for k in range(12):
        y_hard = targets[:, k]
        y_soft = soft_targets[:, k]
        p_mat = np.stack([preds_tiny[:, k], preds_small[:, k], preds_dino[:, k]], axis=1)

        # Sample weights: 2.0x for verified gold annotations, 0.0 for active contradictions
        has_hard = ~np.isnan(y_hard)
        is_conflict = has_hard & (((y_hard == 1.0) & (y_soft < 0.30)) | ((y_hard == 0.0) & (y_soft > 0.70)))
        sample_w = np.where(is_gold, 2.0, 1.0)
        sample_w[is_conflict] = 0.0  # mask contradictions

        effective_target = np.where(has_hard, y_hard, y_soft)
        valid_mask = sample_w > 0

        p_mat_valid = p_mat[valid_mask]
        t_valid = effective_target[valid_mask]
        w_valid = sample_w[valid_mask]

        def loss_fn(w):
            w = w / np.sum(w)
            p_blend = np.dot(p_mat_valid, w)
            p_blend = np.clip(p_blend, 1e-6, 1.0 - 1e-6)
            bce = -(t_valid * np.log(p_blend) + (1.0 - t_valid) * np.log(1.0 - p_blend))
            return np.average(bce, weights=w_valid)

        res = minimize(loss_fn, x0=[0.5, 0.3, 0.2], bounds=[(0.0, 1.0)]*3, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        w_norm = res.x / np.sum(res.x)
        optimal_weights[k] = w_norm
        preds_opt[:, k] = np.dot(p_mat, w_norm)

    auc_opt = compute_macro_auc(targets, preds_opt)

    table = Table(title="Phase 11 Stacking Meta-Learner (Consensus-Denoised) - Fold 0")
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Val Macro AUC", style="bold green")
    table.add_column("Delta vs Tiny", style="yellow")
    table.add_column("Delta vs Fixed Blend", style="magenta")

    table.add_row("ConvNeXt-Tiny Standalone", f"{auc_tiny['macro_auc']:.4f}", "-", f"{auc_tiny['macro_auc'] - auc_fixed['macro_auc']:+.4f}")
    table.add_row("ConvNeXt-Small Standalone", f"{auc_small['macro_auc']:.4f}", f"{auc_small['macro_auc'] - auc_tiny['macro_auc']:+.4f}", f"{auc_small['macro_auc'] - auc_fixed['macro_auc']:+.4f}")
    table.add_row("DINOv2-Small Standalone", f"{auc_dino['macro_auc']:.4f}", f"{auc_dino['macro_auc'] - auc_tiny['macro_auc']:+.4f}", f"{auc_dino['macro_auc'] - auc_fixed['macro_auc']:+.4f}")
    table.add_row("Fixed Blend (0.50/0.30/0.20)", f"{auc_fixed['macro_auc']:.4f}", f"{auc_fixed['macro_auc'] - auc_tiny['macro_auc']:+.4f}", "-")
    table.add_row("Stacking Denoised Optimized ⭐", f"{auc_opt['macro_auc']:.4f}", f"{auc_opt['macro_auc'] - auc_tiny['macro_auc']:+.4f}", f"{auc_opt['macro_auc'] - auc_fixed['macro_auc']:+.4f}")
    console.print(table)

    # Save weights JSON
    os.makedirs('checkpoints', exist_ok=True)
    weights_dict = {
        col: {
            'tiny': float(optimal_weights[idx, 0]),
            'small': float(optimal_weights[idx, 1]),
            'dino': float(optimal_weights[idx, 2]),
        }
        for idx, col in enumerate(TARGET_COLUMNS)
    }
    with open('checkpoints/phase11_stacking_weights_fold0.json', 'w') as f:
        json.dump(weights_dict, f, indent=2)

    console.print("\n✅ Saved optimal weights to [bold]checkpoints/phase11_stacking_weights_fold0.json[/bold]")

if __name__ == '__main__':
    main()
