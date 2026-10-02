#!/usr/bin/env python3
"""
Phase 11 Foundation Model Head-to-Head Benchmark:
Evaluates candidate foundation backbones on Fold 0 Knee MRI Abnormality Detection:
1. RadImageNet (ResNet-50, 2048-dim, 1.35M Radiology Scans)
2. DINOv3 (vit_small_patch16_dinov3, 384-dim, Meta AI)
3. DINOv2-Reg4 (vit_small_patch14_reg4_dinov2, 384-dim with registers)
4. BioMedCLIP (ViT-B/16, 512-dim, Microsoft PubMed)
5. ConvNeXt-V2-Tiny (convnextv2_tiny, 768-dim with GRN)
6. ConvNeXt-Tiny (Phase 11 Baseline)
7. DINOv2-Small (Phase 11 Baseline)
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.table import Table

import timm
from src.data.mri_aware_dataset import MRIAwareKneeDataset, mri_aware_collate
from src.data.transforms import get_validation_transforms
from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything
from src.benchmarks.foundation_extractors import RadImageNetExtractor, BioMedCLIPExtractor

console = Console()

class GenericFeatureExtractor(nn.Module):
    def __init__(self, name: str, img_size: int = 288):
        super().__init__()
        self.name = name
        self.img_size = img_size

        if name == "radimagenet":
            self.extractor = RadImageNetExtractor(img_size=img_size)
            self.embed_dim = 2048
            self.input_size = 224
        elif name == "biomedclip":
            self.extractor = BioMedCLIPExtractor(img_size=224)
            self.embed_dim = 512
            self.input_size = 224
        elif name == "dinov3_small":
            self.model = timm.create_model("vit_small_patch16_dinov3", pretrained=True, num_classes=0, img_size=288, dynamic_img_size=True)
            self.embed_dim = self.model.num_features
            self.input_size = 288
            self.extractor = None
        elif name == "dinov2_reg4":
            self.model = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=True, num_classes=0, img_size=280, dynamic_img_size=True)
            self.embed_dim = self.model.num_features
            self.input_size = 280
            self.extractor = None
        elif name == "dinov2_small":
            self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=280, dynamic_img_size=True)
            self.embed_dim = self.model.num_features
            self.input_size = 280
            self.extractor = None
        elif name == "convnextv2_tiny":
            self.model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=0)
            self.embed_dim = self.model.num_features
            self.input_size = 288
            self.extractor = None
        elif name == "convnext_tiny":
            self.model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.embed_dim = self.model.num_features
            self.input_size = 288
            self.extractor = None
        elif name in ("efficientnet_b0", "efficientnet_b2"):
            self.model = timm.create_model(name, pretrained=True, num_classes=0)
            self.embed_dim = self.model.num_features
            self.input_size = 288
            self.extractor = None
        elif name in ("coatnet_0_rw_224", "coatnet_1_rw_224"):
            self.model = timm.create_model(name, pretrained=True, num_classes=0)
            self.embed_dim = self.model.num_features
            self.input_size = 224
            self.extractor = None
        elif name in ("vit_small_patch16_224", "vit_base_patch16_224.mae"):
            self.model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224, dynamic_img_size=True)
            self.embed_dim = self.model.num_features
            self.input_size = 224
            self.extractor = None
        else:
            raise ValueError(f"Unknown backbone: {name}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, C, H, W)
        B, D, C, H, W = x.shape
        if (H, W) != (self.input_size, self.input_size):
            x = F.interpolate(
                x.view(B * D, C, H, W),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False
            ).view(B, D, C, self.input_size, self.input_size)

        if self.extractor is not None:
            return self.extractor.extract_slices(x)

        x_flat = x.view(B * D, C, self.input_size, self.input_size)
        # Process in chunks of 32 to avoid OOM
        chunk_size = 32
        if B * D <= chunk_size:
            feats = self.model(x_flat)
        else:
            feats_list = []
            for i in range(0, B * D, chunk_size):
                feats_list.append(self.model(x_flat[i : i + chunk_size]))
            feats = torch.cat(feats_list, dim=0)

        return feats.view(B, D, self.embed_dim)


class CachedFeaturesDataset(Dataset):
    def __init__(self, features: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor, is_gold: torch.Tensor):
        self.features = features
        self.hard = hard
        self.soft = soft
        self.is_gold = is_gold

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'hard': self.hard[idx],
            'soft': self.soft[idx],
            'is_gold': self.is_gold[idx],
        }


def extract_all_features(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = ""):
    model.eval()
    feat_list, hard_list, soft_list, is_gold_list = [], [], [], []
    t0 = time.time()

    with torch.inference_mode():
        for batch in loader:
            vols = batch['images'].to(device).half()
            feats = model(vols)  # (B, D, F)
            feat_list.append(feats.cpu())
            hard_list.append(batch['hard_labels'])
            soft_list.append(batch['soft_labels'])
            is_gold_list.append(batch['is_gold'])

    elapsed = time.time() - t0
    all_feats = torch.cat(feat_list, dim=0)
    all_hard  = torch.cat(hard_list, dim=0)
    all_soft  = torch.cat(soft_list, dim=0)
    all_gold  = torch.cat(is_gold_list, dim=0)
    ms_per_study = (elapsed / len(all_feats)) * 1000

    return all_feats, all_hard, all_soft, all_gold, ms_per_study


def train_and_eval_mil_head(train_feats, train_hard, train_soft, train_gold,
                            val_feats, val_hard, val_soft, val_gold,
                            embed_dim: int, device: torch.device, epochs: int = 15):
    train_ds = CachedFeaturesDataset(train_feats, train_hard, train_soft, train_gold)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)

    mil_pool = LabelSpecificGatedAttentionMILPool(
        in_features=embed_dim,
        num_classes=12,
        hidden_dim=128,
        dropout=0.25
    ).to(device)

    criterion = ConsensusDenoisedBCEWithLogitsLoss(gold_weight=2.0)
    optimizer = torch.optim.AdamW(mil_pool.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auc = 0.0
    best_gold_auc = 0.0
    val_targets = val_hard.numpy()
    val_gold_mask = val_gold.numpy().astype(bool)

    val_feats_dev = val_feats.to(device).float()

    for epoch in range(epochs):
        mil_pool.train()
        for batch in train_loader:
            x = batch['features'].to(device).float()
            hard = batch['hard'].to(device)
            soft = batch['soft'].to(device)
            gold = batch['is_gold'].to(device)

            optimizer.zero_grad()
            logits, _, _ = mil_pool(x)
            loss = criterion(logits, hard, soft, gold)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        mil_pool.eval()
        with torch.no_grad():
            val_logits, _, _ = mil_pool(val_feats_dev)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_res = compute_macro_auc(val_targets, val_probs)
        macro_auc = val_res['macro_auc']

        # Gold AUC
        if val_gold_mask.sum() > 0:
            gold_res = compute_macro_auc(val_targets[val_gold_mask], val_probs[val_gold_mask])
            gold_auc = gold_res['macro_auc']
        else:
            gold_auc = 0.0

        if macro_auc > best_val_auc:
            best_val_auc = macro_auc
            best_gold_auc = gold_auc

    return best_val_auc, best_gold_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=[
        'radimagenet',
        'dinov3_small',
        'dinov2_reg4',
        'convnextv2_tiny',
        'convnext_tiny',
        'dinov2_small',
    ], help='Models to benchmark')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    console.print(f"[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]")
    console.print(f"[bold green]║ Foundation Model Head-to-Head Benchmark (Fold {args.fold})            ║[/bold green]")
    console.print(f"[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]")

    # Load master labels
    df_splits = pd.read_parquet('data/splits_5fold.parquet')
    df_tri = pd.read_parquet('data/tristate_labels_master.parquet')
    df_dense = pd.read_parquet('data/dense_labels_master.parquet')

    df_all = df_splits[['StudyInstanceUID', 'fold']].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all['is_gold'] = labeled_mask

    tri_renamed = df_tri[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'hard_{c}' for c in TARGET_COLUMNS})
    dense_renamed = df_dense[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'soft_{c}' for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on='StudyInstanceUID', how='left')
    df_all = df_all.merge(dense_renamed, on='StudyInstanceUID', how='left')

    train_df = df_all[df_all['fold'] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all['fold'] == args.fold].reset_index(drop=True)

    console.print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} (Gold in Val: {val_df['is_gold'].sum()})")

    transforms = get_validation_transforms((288, 288))
    train_ds = MRIAwareKneeDataset(train_df, target_slices=32, transforms=transforms, is_training=False)
    val_ds = MRIAwareKneeDataset(val_df, target_slices=32, transforms=transforms, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)

    results = []

    for model_name in args.models:
        console.print(f"\n[bold yellow]▶ Evaluating: {model_name.upper()}[/bold yellow]")
        try:
            extractor = GenericFeatureExtractor(model_name, img_size=288).to(device).half()
            
            console.print(f"  Extracting features ({extractor.embed_dim}d)...")
            train_feats, train_hard, train_soft, train_gold, t_ms = extract_all_features(extractor, train_loader, device, "Train")
            val_feats, val_hard, val_soft, val_gold, v_ms = extract_all_features(extractor, val_loader, device, "Val")

            console.print(f"  Feature extraction speed: {v_ms:.1f} ms/study")
            console.print(f"  Training 12-branch Label-Specific MIL head ({args.epochs} epochs)...")
            
            val_auc, gold_auc = train_and_eval_mil_head(
                train_feats, train_hard, train_soft, train_gold,
                val_feats, val_hard, val_soft, val_gold,
                embed_dim=extractor.embed_dim,
                device=device,
                epochs=args.epochs
            )

            console.print(f"  [bold green]✓ {model_name:<16}: Val Macro AUC = {val_auc:.4f} | Gold AUC = {gold_auc:.4f}[/bold green]")
            results.append({
                "Model": model_name,
                "Pretrain Domain": "Medical/MRI (1.35M)" if "rad" in model_name else ("Vision (Meta AI)" if "dino" in model_name else ("Hybrid Conv+Attn" if "coat" in model_name else ("Self-Supervised MAE" if "mae" in model_name else "Vision/ImageNet"))),
                "Dim": extractor.embed_dim,
                "Val AUC": round(val_auc, 4),
                "Gold AUC": round(gold_auc, 4),
                "Speed (ms)": round(v_ms, 1),
            })
            del extractor, train_feats, val_feats
            torch.cuda.empty_cache()

        except Exception as e:
            console.print(f"  [red]✗ Error evaluating {model_name}: {e}[/red]")
            import traceback
            traceback.print_exc()

    # Print summary leaderboard table
    table = Table(title="Foundation Model Head-to-Head Benchmark Leaderboard (Fold 0)")
    table.add_column("Model", style="bold cyan")
    table.add_column("Pretrain Domain", style="magenta")
    table.add_column("Dim", justify="right")
    table.add_column("Val Macro AUC", justify="right", style="bold green")
    table.add_column("Gold Consensus AUC", justify="right", style="bold yellow")
    table.add_column("Latency (ms/study)", justify="right")

    # Sort results by Val AUC descending
    results.sort(key=lambda x: x["Val AUC"], reverse=True)
    for r in results:
        table.add_row(
            r["Model"],
            r["Pretrain Domain"],
            str(r["Dim"]),
            f"{r['Val AUC']:.4f}",
            f"{r['Gold AUC']:.4f}",
            str(r["Speed (ms)"])
        )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    main()
