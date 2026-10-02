"""
RSNA Knee Abnormality Detection - Phase 11 Pre-Classifier Embedding Fusion (Fold 0)
Fuses the pre-classifier pooled pathology representations h_norm in R^{12 x F}
across ConvNeXt-Tiny (768), ConvNeXt-Small (768), and DINOv2-Small (384)
into a unified 1920-dimensional embedding per abnormality.

Initializes the unified head using Warm-Start Weight Transplantation from the 3
pre-trained models, guaranteeing Epoch 0 matches the best ensemble (0.8487 / 0.745),
and trains the head to find cross-backbone feature synergies.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass
from rich.console import Console
from rich.table import Table

from src.data.mri_aware_dataset import MRIAwareKneeDataset, mri_aware_collate
from src.data.transforms import get_validation_transforms
from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
import timm

console = Console()

class BaseMILFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', mil_hidden_dim=128):
        super().__init__()
        self.backbone_name = backbone_name
        extra_kwargs = {}
        if 'dinov2' in backbone_name.lower():
            extra_kwargs = {'img_size': 336, 'dynamic_img_size': True}
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, in_chans=3, **extra_kwargs)
        self.num_features = self.backbone.num_features
        self.mil_pool = LabelSpecificGatedAttentionMILPool(in_features=self.num_features, num_classes=12, hidden_dim=mil_hidden_dim)

    def forward(self, images):
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        feats_flat = self.backbone(x_flat)
        slice_feats = feats_flat.view(B, D, self.num_features)

        # Extract normalized pooled representations:
        A_V = self.mil_pool.attention_V(slice_feats)
        A_U = self.mil_pool.attention_U(slice_feats)
        raw_attn = self.mil_pool.attention_weights(A_V * A_U).transpose(1, 2)
        attn = torch.softmax(raw_attn, dim=-1)
        h = torch.bmm(attn, slice_feats)
        h_norm = self.mil_pool.norm(h)  # (B, 12, F)
        logits = torch.einsum('bkf,kf->bk', h_norm, self.mil_pool.classifier_w) + self.mil_pool.classifier_b
        return logits, h_norm

def load_extractor(backbone_name, ckpt_path, device):
    model = BaseMILFeatureExtractor(backbone_name)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(st)
    return model.to(device).eval().half()

class PreClassifierFusionHead(nn.Module):
    def __init__(self, in_features=1920, num_classes=12, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.in_features = in_features
        self.num_classes = num_classes
        self.classifier_w = nn.Parameter(torch.empty(num_classes, in_features))
        self.classifier_b = nn.Parameter(torch.empty(num_classes))

    def init_from_pretraineds(self, w_tiny, b_tiny, w_small, b_small, w_dino, b_dino):
        """Warm-start transplantation: exactly reproduces the 0.50/0.30/0.20 ensemble at step 0."""
        with torch.no_grad():
            w_fused = torch.cat([0.50 * w_tiny, 0.30 * w_small, 0.20 * w_dino], dim=-1)
            b_fused = 0.50 * b_tiny + 0.30 * b_small + 0.20 * b_dino
            self.classifier_w.copy_(w_fused)
            self.classifier_b.copy_(b_fused)

    def forward(self, h_fused):
        # h_fused: (B, 12, 1920)
        h_in = self.dropout(h_fused)
        logits = torch.einsum('bkf,kf->bk', h_in, self.classifier_w) + self.classifier_b
        return logits

def extract_embeddings(loader, m_tiny, m_small, m_dino, device, desc=""):
    console.print(f"[cyan]Extracting pre-classifier representations ({desc})...[/cyan]")
    fused_embeds, hard_list, soft_list, is_gold_list, study_uids = [], [], [], [], []
    logits_tiny_list, logits_small_list, logits_dino_list = [], [], []

    with torch.inference_mode():
        for batch in loader:
            vols = batch['images'].to(device).half()
            B, D, C, H, W = vols.shape
            vols_dino = F.interpolate(vols.view(B * D, C, H, W), size=(280, 280), mode='bilinear', align_corners=False).view(B, D, C, 280, 280)

            with torch.amp.autocast('cuda'):
                l_tiny, h_tiny = m_tiny(vols)
                l_small, h_small = m_small(vols)
                l_dino, h_dino = m_dino(vols_dino)

                # Concatenate normalized pooled representations: 768 + 768 + 384 = 1920
                h_fused = torch.cat([h_tiny, h_small, h_dino], dim=-1)  # (B, 12, 1920)

            fused_embeds.append(h_fused.cpu().float())
            logits_tiny_list.append(l_tiny.cpu().float())
            logits_small_list.append(l_small.cpu().float())
            logits_dino_list.append(l_dino.cpu().float())
            hard_list.append(batch['hard_labels'].detach().cpu().clone())
            soft_list.append(batch['soft_labels'].detach().cpu().clone())
            is_gold_list.append(batch['is_gold'].detach().cpu().clone())
            study_uids.extend(batch.get('study_uids', batch.get('study_uid', [])))
            del batch

    return {
        'h_fused': torch.cat(fused_embeds, dim=0),
        'logits_tiny': torch.cat(logits_tiny_list, dim=0),
        'logits_small': torch.cat(logits_small_list, dim=0),
        'logits_dino': torch.cat(logits_dino_list, dim=0),
        'hard': torch.cat(hard_list, dim=0),
        'soft': torch.cat(soft_list, dim=0),
        'is_gold': torch.cat(is_gold_list, dim=0),
        'study_uids': study_uids,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]")
    console.print(f"[bold green]║ Phase 11: PRE-CLASSIFIER EMBEDDING FUSION (1920d) ON FOLD {args.fold}     ║[/bold green]")
    console.print(f"[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]")

    # 1. Dataset splits
    df_splits = pd.read_parquet('data/splits_5fold.parquet')
    df_tri = pd.read_parquet('data/tristate_labels_master.parquet')
    df_dense = pd.read_parquet('data/dense_labels_master.parquet')

    df_all = df_splits[['StudyInstanceUID', 'fold']].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all['is_gold'] = labeled_mask

    tri_renamed = df_tri[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'hard_{c}' for c in TARGET_COLUMNS})
    dense_renamed = df_dense[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'soft_{c}' for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on='StudyInstanceUID', how='left').merge(dense_renamed, on='StudyInstanceUID', how='left')

    train_df = df_all[df_all['fold'] != args.fold].reset_index(drop=True)
    val_df   = df_all[df_all['fold'] == args.fold].reset_index(drop=True)

    val_transforms = get_validation_transforms((288, 288))
    train_ds = MRIAwareKneeDataset(train_df, target_slices=32, transforms=val_transforms, is_training=False)
    val_ds   = MRIAwareKneeDataset(val_df, target_slices=32, transforms=val_transforms, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)

    # 2. Load base models with pretrained weights
    ckpts = {
        'tiny': 'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_tiny_mri_aware_fold0_best.pth',
        'small': 'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_small_mri_aware_fold0_best.pth',
        'dinov2': 'kaggle_upload/datasets/phase_11_tri_ensemble/dinov2_small_mri_aware_fold0_best.pth',
    }

    console.print("[cyan]Loading Phase 11 best checkpoints (Tiny + Small + DINOv2)...[/cyan]")
    m_tiny = load_extractor('convnext_tiny', ckpts['tiny'], device)
    m_small = load_extractor('convnext_small', ckpts['small'], device)
    m_dino = load_extractor('vit_small_patch14_dinov2.lvd142m', ckpts['dinov2'], device)

    # Save classifier weights for warm-start transplantation
    w_tiny = m_tiny.mil_pool.classifier_w.detach().cpu().float()
    b_tiny = m_tiny.mil_pool.classifier_b.detach().cpu().float()
    w_small = m_small.mil_pool.classifier_w.detach().cpu().float()
    b_small = m_small.mil_pool.classifier_b.detach().cpu().float()
    w_dino = m_dino.mil_pool.classifier_w.detach().cpu().float()
    b_dino = m_dino.mil_pool.classifier_b.detach().cpu().float()

    # 3. Extract pre-classifier embeddings
    train_data = extract_embeddings(train_loader, m_tiny, m_small, m_dino, device, desc="Train Set")
    val_data   = extract_embeddings(val_loader, m_tiny, m_small, m_dino, device, desc="Val Set")

    del m_tiny, m_small, m_dino
    torch.cuda.empty_cache()

    train_feat_ds = TensorDataset(train_data['h_fused'], train_data['hard'], train_data['soft'], train_data['is_gold'])
    train_feat_loader = DataLoader(train_feat_ds, batch_size=32, shuffle=True)

    # Baselines
    val_targets = val_data['hard'].numpy()
    p_tiny = torch.sigmoid(val_data['logits_tiny']).numpy()
    p_small = torch.sigmoid(val_data['logits_small']).numpy()
    p_dino = torch.sigmoid(val_data['logits_dino']).numpy()
    p_fixed = 0.50 * p_tiny + 0.30 * p_small + 0.20 * p_dino

    auc_tiny = compute_macro_auc(val_targets, p_tiny)['macro_auc']
    auc_small = compute_macro_auc(val_targets, p_small)['macro_auc']
    auc_dino = compute_macro_auc(val_targets, p_dino)['macro_auc']
    auc_fixed = compute_macro_auc(val_targets, p_fixed)['macro_auc']

    # 4. Initialize PreClassifierFusionHead with Warm-Start Transplantation
    head = PreClassifierFusionHead(in_features=1920, num_classes=12, dropout=args.dropout).to(device)
    head.init_from_pretraineds(w_tiny.to(device), b_tiny.to(device), w_small.to(device), b_small.to(device), w_dino.to(device), b_dino.to(device))

    val_h = val_data['h_fused'].to(device)
    head.eval()
    with torch.no_grad():
        step0_probs = torch.sigmoid(head(val_h)).cpu().numpy()
    step0_auc = compute_macro_auc(val_targets, step0_probs)['macro_auc']
    console.print(f"[bold green]✓ Step 0 (Warm-Start Initialized) Val Macro AUC: {step0_auc:.4f}[/bold green] (Baseline Fixed Blend: {auc_fixed:.4f})")

    # 5. Train
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = ConsensusDenoisedBCEWithLogitsLoss(gold_weight=2.0)

    best_val_auc = step0_auc
    os.makedirs('checkpoints', exist_ok=True)

    console.print(f"\n[bold yellow]Optimizing Pre-Classifier Fusion Head for {args.epochs} epochs...[/bold yellow]")
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        for h_b, hard_b, soft_b, is_gold_b in train_feat_loader:
            h_b, hard_b, soft_b, is_gold_b = h_b.to(device), hard_b.to(device), soft_b.to(device), is_gold_b.to(device)
            optimizer.zero_grad()
            logits = head(h_b)
            loss = criterion(logits, hard_b, soft_b, is_gold_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(h_b)

        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(head(val_h)).cpu().numpy()
        val_auc = compute_macro_auc(val_targets, val_probs)['macro_auc']

        is_best = val_auc > best_val_auc
        star = "⭐ BEST" if is_best else ""
        if epoch % 5 == 0 or is_best:
            console.print(f"Epoch [{epoch:>02}/{args.epochs}] | Loss: {total_loss/len(train_feat_ds):.4f} | Val Macro AUC: {val_auc:.4f} {star}")

        if is_best:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'state_dict': head.state_dict(),
                'val_macro_auc': val_auc,
                'in_features': 1920,
            }, f'checkpoints/pre_classifier_fusion_fold{args.fold}_best.pth')

    table = Table(title="Pre-Classifier Embedding Fusion vs. Baselines (Fold 0)")
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Val Macro AUC", style="bold green")
    table.add_column("Delta vs Tiny", style="yellow")
    table.add_column("Delta vs Fixed Blend", style="magenta")

    table.add_row("ConvNeXt-Tiny Standalone", f"{auc_tiny:.4f}", "-", f"{auc_tiny - auc_fixed:+.4f}")
    table.add_row("ConvNeXt-Small Standalone", f"{auc_small:.4f}", f"{auc_small - auc_tiny:+.4f}", f"{auc_small - auc_fixed:+.4f}")
    table.add_row("DINOv2-Small Standalone", f"{auc_dino:.4f}", f"{auc_dino - auc_tiny:+.4f}", f"{auc_dino - auc_fixed:+.4f}")
    table.add_row("Fixed Blend (0.50/0.30/0.20)", f"{auc_fixed:.4f}", f"{auc_fixed - auc_tiny:+.4f}", "-")
    table.add_row("Pre-Classifier 1920d Fusion Head ⭐", f"{best_val_auc:.4f}", f"{best_val_auc - auc_tiny:+.4f}", f"{best_val_auc - auc_fixed:+.4f}")
    console.print(table)

    console.print(f"\n✅ Pre-Classifier Fusion Model saved to: [bold green]checkpoints/pre_classifier_fusion_fold{args.fold}_best.pth[/bold green]")

if __name__ == '__main__':
    main()
