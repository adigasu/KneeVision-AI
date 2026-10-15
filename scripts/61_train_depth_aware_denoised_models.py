"""
Phase 18: Train Anatomical Depth-Conditioned Tri-Planar MIL Model
Supervised on Confident Learning Denoised Master Labels (v5).
Features:
- Anatomical Depth Conditioning (Sinusoidal Depth Embeddings + Gaussian Pathology Priors)
- Sample-Weighted Loss (clean studies get full weight; noisy studies downweighted)
- Strict Zero-Flip Augmentation (preserving anatomical laterality)
- Native T2 FS Multi-Planar Series
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.data.triplanar_dataset import TriPlanarKneeDataset, triplanar_collate
from src.models.depth_aware_mil import AnatomicalDepthAwareTriPlanarMILModel

console = Console()


class SampleWeightedDenoisedLoss(nn.Module):
    """BCEWithLogitsLoss weighted by Confident Learning sample confidence weights."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss_matrix = self.bce(logits, targets)  # (B, 12)
        if weights is not None:
            weighted_loss = loss_matrix * weights
            return weighted_loss.sum() / (weights.sum() + 1e-8)
        return loss_matrix.mean()


def train_epoch(model, loader, optimizer, scaler, criterion, device, grad_accum_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        depths = batch.get("relative_depths", None)
        if depths is not None:
            depths = depths.to(device, non_blocking=True)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        targets = batch["targets"].to(device, non_blocking=True)
        weights = batch.get("weights", None)
        if weights is not None:
            weights = weights.to(device, non_blocking=True)

        with autocast():
            out = model(images, relative_depths=depths, mask=mask)
            loss = criterion(out["logits"], targets, weights) / grad_accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        depths = batch.get("relative_depths", None)
        if depths is not None:
            depths = depths.to(device, non_blocking=True)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        with autocast():
            out = model(images, relative_depths=depths, mask=mask)
            probs = torch.sigmoid(out["logits"]).cpu().float().numpy()

        all_preds.append(probs)
        all_targets.append(batch["targets"].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_macro_auc(targets, preds)
    return metrics["macro_auc"], preds, targets


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 18 Depth-Conditioned Denoised Training")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="convnext_small")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_name", type=str, default="phase18_depth_convnext_small_384px_f0")
    return parser.parse_args()


def main():
    args = parse_args()
    console.print(f"[bold green]Starting Phase 18 Depth-Conditioned Training: {args.run_name}[/bold green]")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Denoised Master Dataset
    data_path = "data/dense_labels_master_v5_denoised.parquet"
    splits_path = "data/splits_5fold.parquet"

    df_data = pd.read_parquet(data_path)
    df_splits = pd.read_parquet(splits_path)

    df_all = df_data.copy()
    if "fold" not in df_all.columns:
        df_all["fold"] = df_splits["fold"].values

    train_df = df_all[df_all["fold"] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == args.fold].reset_index(drop=True)

    console.print(f"Train Studies: {len(train_df)} | Val Studies: {len(val_df)}")

    # Transforms (Strictly zero horizontal flip to protect laterality)
    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_ds = TriPlanarKneeDataset(train_df, slices_per_plane=24, transforms=train_transforms, is_training=True)
    val_ds = TriPlanarKneeDataset(val_df, slices_per_plane=24, transforms=val_transforms, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=triplanar_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=triplanar_collate)

    # Initialize Depth-Conditioned Model
    model = AnatomicalDepthAwareTriPlanarMILModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=128,
        dropout=0.3,
        slices_per_plane=24,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    criterion = SampleWeightedDenoisedLoss()

    best_macro = 0.0
    save_path = f"checkpoints/{args.run_name}_best.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, args.grad_accum_steps)
        val_macro, val_preds, _ = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_macro > best_macro
        if is_best:
            best_macro = val_macro
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_macro": val_macro, "args": vars(args)}, save_path)
            np.save(f"checkpoints/{args.run_name}_val_preds.npy", val_preds)

        tag = "⭐ BEST" if is_best else ""
        console.print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Macro AUC: {val_macro:.4f} {tag} ({elapsed:.1f}s)")

    console.print(f"\n[bold green]Training complete! Best Val Macro AUC: {best_macro:.4f} saved to {save_path}[/bold green]")


if __name__ == "__main__":
    main()
