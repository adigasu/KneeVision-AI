"""
RSNA Knee Abnormality Detection - Phase 11 Joint Multi-Backbone Feature Fusion
Concatenates slice-level features from ConvNeXt-Tiny (768d), ConvNeXt-Small (768d),
and DINOv2-Small ViT (384d) into a 1920d unified representation.
Fuses them via a 12-branch Label-Specific Gated Attention MIL Pool.
Trained on Fold 0 using Consensus-Denoised Multi-Tier Loss.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
import timm

from src.data.mri_aware_dataset import MRIAwareKneeDataset, mri_aware_collate
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS

console = Console()

class JointMultiBackboneMILModel(nn.Module):
    def __init__(self, num_classes=12, mil_hidden_dim=256, dropout=0.25):
        super().__init__()
        self.num_classes = num_classes

        # 1. Backbones
        self.bb_tiny  = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, in_chans=3)
        self.bb_small = timm.create_model('convnext_small', pretrained=False, num_classes=0, in_chans=3)
        self.bb_dino  = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, in_chans=3, img_size=336, dynamic_img_size=True)

        self.f_tiny  = self.bb_tiny.num_features    # 768
        self.f_small = self.bb_small.num_features   # 768
        self.f_dino  = self.bb_dino.num_features    # 384
        self.f_total = self.f_tiny + self.f_small + self.f_dino  # 1920

        # 2. Joint 1920-dim Label-Specific MIL Pool
        self.mil_pool = LabelSpecificGatedAttentionMILPool(
            in_features=self.f_total,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

    def load_pretraineds(self, ckpt_tiny, ckpt_small, ckpt_dino):
        console.print("[cyan]Preloading backbone weights from single-model checkpoints...[/cyan]")
        for bb, path, name in [(self.bb_tiny, ckpt_tiny, 'Tiny'), (self.bb_small, ckpt_small, 'Small'), (self.bb_dino, ckpt_dino, 'DINOv2')]:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            st = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            bb_st = {k.replace('backbone.', ''): v for k, v in st.items() if k.startswith('backbone.')}
            bb.load_state_dict(bb_st, strict=True)
            console.print(f"  ✓ {name} backbone loaded cleanly")

    def forward(self, images):
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)

        # CNN forward passes
        f1 = self.bb_tiny(x_flat)   # (B*D, 768)
        f2 = self.bb_small(x_flat)  # (B*D, 768)

        # ViT forward pass (bilinear interpolate to 280px)
        x_dino = F.interpolate(x_flat, size=(280, 280), mode='bilinear', align_corners=False)
        f3 = self.bb_dino(x_dino)   # (B*D, 384)

        # Concatenate slice features
        f_concat = torch.cat([f1, f2, f3], dim=-1)  # (B*D, 1920)
        slice_feats = f_concat.view(B, D, self.f_total)

        res = self.mil_pool(slice_feats)
        logits = res[0] if isinstance(res, (tuple, list)) else res
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum_steps', type=int, default=16)
    parser.add_argument('--lr_head', type=float, default=1.5e-4)
    parser.add_argument('--lr_backbone', type=float, default=1.5e-5)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]")
    console.print(f"[bold green]║ Phase 11: JOINT MULTI-BACKBONE FUSION (1920d MIL) ON FOLD {args.fold}     ║[/bold green]")
    console.print(f"[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]")

    # Datasets
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

    train_ds = MRIAwareKneeDataset(train_df, target_slices=32, transforms=get_training_transforms((288, 288)), is_training=True)
    val_ds   = MRIAwareKneeDataset(val_df, target_slices=32, transforms=get_validation_transforms((288, 288)), is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=mri_aware_collate, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=mri_aware_collate, pin_memory=True)

    model = JointMultiBackboneMILModel(num_classes=12, mil_hidden_dim=256)
    model.load_pretraineds(
        'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_tiny_mri_aware_fold0_best.pth',
        'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_small_mri_aware_fold0_best.pth',
        'kaggle_upload/datasets/phase_11_tri_ensemble/dinov2_small_mri_aware_fold0_best.pth',
    )
    model = model.to(device)

    # Param groups: low lr for backbones, higher lr for joint MIL head
    params = [
        {'params': model.bb_tiny.parameters(), 'lr': args.lr_backbone},
        {'params': model.bb_small.parameters(), 'lr': args.lr_backbone},
        {'params': model.bb_dino.parameters(), 'lr': args.lr_backbone},
        {'params': model.mil_pool.parameters(), 'lr': args.lr_head},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    criterion = ConsensusDenoisedBCEWithLogitsLoss(gold_weight=2.0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_auc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            hard = batch['hard_labels'].to(device)
            soft = batch['soft_labels'].to(device)
            is_gold = batch['is_gold'].to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, hard, soft, is_gold) / args.grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.inference_mode():
            for batch in val_loader:
                images = batch['images'].to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(images)
                val_preds.append(torch.sigmoid(logits).cpu().float().numpy())
                val_targets.append(batch['hard_labels'].numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_auc_res = compute_macro_auc(val_targets, val_preds)
        val_auc = val_auc_res['macro_auc']

        is_best = val_auc > best_val_auc
        star = "⭐ BEST" if is_best else ""
        console.print(f"Epoch [{epoch:>02}/{args.epochs}] | Loss: {total_loss/len(train_loader):.4f} | Val Macro AUC: {val_auc:.4f} {star}")

        if is_best:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_macro_auc': val_auc,
            }, f'checkpoints/joint_multibackbone_fusion_fold{args.fold}_best.pth')

    console.print(f"\n✅ Training complete! Best Val Macro AUC: [bold green]{best_val_auc:.4f}[/bold green]")

if __name__ == '__main__':
    main()
