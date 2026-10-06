"""
RSNA Knee Abnormality Detection - Phase 16: Anatomically-Decoupled Tri-Planar MIL Training.
Trains PlaneDecoupledTriPlanarMILModel on Fold 0 with:
1. Decoupled intra-plane pooling for Sagittal (0:24), Coronal (24:48), and Axial (48:72).
2. Pathology-specific cross-plane routing (concentrates Coronal attention for MCL to fix the 0.661 bottleneck).
3. Bottleneck-weighted Consensus Denoised BCE Loss (2.2x weight on MCL, 1.5x on Lateral OA/Fracture/Synovitis).
4. Proven baseline hyperparameters: 288x288 resolution, 24 slices/plane (72 total), 12 epochs.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.data.triplanar_dataset import TriPlanarKneeDataset, triplanar_collate
from src.models.plane_decoupled_mil import PlaneDecoupledTriPlanarMILModel
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss


# Bottleneck class weights targeting lagging pathologies
BOTTLENECK_WEIGHT_MAP = {
    "MCL": 2.2,
    "Lateral OA": 1.5,
    "Synovitis": 1.5,
    "Fracture": 1.5,
    "Lateral Meniscus": 1.4,
    "PF OA": 1.2,
    "ACL": 1.0,
    "Medial Meniscus": 1.0,
    "Effusion": 1.0,
    "Medial OA": 0.6,
    "Baker's": 0.6,
    "Contusion": 0.6,
}


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_soft, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, mask=mask)
                probs = torch.sigmoid(out["logits"]).cpu().numpy()

            all_preds.append(probs)
            all_soft.append(batch["soft_targets"].numpy())
            all_is_gold.append(batch["is_gold"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_soft, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    # Full macro AUC across all validation studies
    full_metrics = compute_macro_auc(targets, preds)
    full_macro = full_metrics["macro_auc"]
    full_per_label = full_metrics["per_class_auc"]

    # Gold subset macro AUC (strict human consensus annotations)
    gold_macro = 0.0
    gold_per_label = {}
    if is_gold.sum() > 5:
        gold_metrics = compute_macro_auc(targets[is_gold], preds[is_gold])
        gold_macro = gold_metrics["macro_auc"]
        gold_per_label = gold_metrics["per_class_auc"]

    return full_macro, full_per_label, gold_macro, gold_per_label, preds


def main():
    parser = argparse.ArgumentParser(description="Phase 16 Decoupled Tri-Planar MIL Training")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--slices_per_plane", type=int, default=24)
    parser.add_argument("--image_size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--gold_weight", type=float, default=2.0)
    parser.add_argument("--pos_thresh", type=float, default=0.70)
    parser.add_argument("--neg_thresh", type=float, default=0.30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--use_bottleneck_weights", type=int, default=1, choices=[0, 1], help="1 to enable, 0 to disable")
    parser.add_argument("--classifier_type", type=str, default="multi_head_mlp", choices=["multi_head_mlp", "linear"])
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--grad_checkpointing", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()

    if not args.run_name:
        args.run_name = f"phase16_decoupled_{args.backbone}_f{args.fold}"

    console = Console()
    console.print(f"[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║  Phase 16: Decoupled Tri-Planar MIL Training ({args.run_name})     ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print(f"Device: {args.device} | Backbone: {args.backbone} | Fold: {args.fold}")
    console.print(f"Slices: {args.slices_per_plane}x3 = {args.slices_per_plane*3} total | Resolution: {args.image_size}x{args.image_size}")
    console.print(f"Effective Batch: {args.batch_size * args.grad_accum_steps} (Batch {args.batch_size} x Accum {args.grad_accum_steps})")

    # 1. Load Data
    splits_path = "data/splits_5fold.parquet"
    tri_path = "data/tristate_labels_master.parquet"
    dense_path = "data/dense_labels_master.parquet"

    df_splits = pd.read_parquet(splits_path)
    df_tri = pd.read_parquet(tri_path)
    df_dense = pd.read_parquet(dense_path)

    df_all = df_splits[["StudyInstanceUID", "fold"]].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all["is_gold"] = labeled_mask

    tri_renamed = df_tri[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"hard_{c}" for c in TARGET_COLUMNS})
    dense_renamed = df_dense[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on="StudyInstanceUID", how="left")
    df_all = df_all.merge(dense_renamed, on="StudyInstanceUID", how="left")

    train_df = df_all[df_all["fold"] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == args.fold].reset_index(drop=True)

    console.print(f"Train Studies: {len(train_df)} (Gold: {train_df['is_gold'].sum()})")
    console.print(f"Val Studies:   {len(val_df)} (Gold: {val_df['is_gold'].sum()})")

    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_ds = TriPlanarKneeDataset(
        train_df,
        slices_per_plane=args.slices_per_plane,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = TriPlanarKneeDataset(
        val_df,
        slices_per_plane=args.slices_per_plane,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=triplanar_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=triplanar_collate,
        pin_memory=True,
    )

    # 2. Model
    model = PlaneDecoupledTriPlanarMILModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=len(TARGET_COLUMNS),
        slices_per_plane=args.slices_per_plane,
        chunk_size=args.chunk_size,
        classifier_type=args.classifier_type,
        head_hidden_dim=args.head_hidden_dim,
    ).to(args.device)

    if args.grad_checkpointing == 1 and hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(True)
        console.print('[bold green]Gradient checkpointing enabled on backbone (memory-optimized)![/bold green]')

    # 3. Loss Function with Bottleneck Class Weights
    class_weights_tensor = None
    if args.use_bottleneck_weights == 1:
        weights_list = [BOTTLENECK_WEIGHT_MAP.get(c, 1.0) for c in TARGET_COLUMNS]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        console.print(f"Bottleneck Weights: MCL={BOTTLENECK_WEIGHT_MAP['MCL']}x | LatOA/Fracture/Synovitis=1.5x | Saturated=0.6x")
    else:
        console.print("[bold yellow]Running UNWEIGHTED Consensus Denoised Loss (Equal 1.0x across all 12 classes)[/bold yellow]")

    criterion = ConsensusDenoisedBCEWithLogitsLoss(
        gold_weight=args.gold_weight,
        pos_thresh=args.pos_thresh,
        neg_thresh=args.neg_thresh,
        class_weights=class_weights_tensor,
    )

    # 4. Optimizer & Scheduler
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-2,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    # 5. Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    best_gold_auc = 0.0
    best_val_macro = 0.0
    best_preds = None
    checkpoint_path = f"checkpoints/{args.run_name}_best.pth"
    val_pred_path = f"checkpoints/{args.run_name}_val_preds.npy"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(args.device)
            mask = batch["mask"].to(args.device)
            hard = batch["hard_targets"].to(args.device)
            soft = batch["soft_targets"].to(args.device)
            is_gold = batch["is_gold"].to(args.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, mask=mask)
                loss = criterion(out["logits"], hard, soft, is_gold)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scale_before <= scaler.get_scale():
                    scheduler.step()

        train_loss /= len(train_loader)
        epoch_time = time.time() - t0

        # Evaluate
        val_macro, val_per_label, gold_macro, gold_per_label, preds = evaluate(model, val_loader, args.device)

        mcl_auc = val_per_label.get("MCL", 0.0)
        gold_mcl = gold_per_label.get("MCL", 0.0)
        is_best = gold_macro > best_gold_auc or (gold_macro == best_gold_auc and val_macro > best_val_macro)

        star = "⭐ BEST" if is_best else ""
        console.print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.0f}s] | "
            f"Loss: {train_loss:.4f} | "
            f"Val Macro AUC: {val_macro:.4f} | "
            f"Gold AUC: {gold_macro:.4f} {star}"
        )
        console.print(
            f"   -> MCL AUC: {mcl_auc:.4f} (Gold MCL: {gold_mcl:.4f}) | "
            f"MedOA: {val_per_label.get('Medial OA', 0):.3f} | "
            f"LatOA: {val_per_label.get('Lateral OA', 0):.3f} | "
            f"PFOA: {val_per_label.get('PF OA', 0):.3f} | "
            f"Fracture: {val_per_label.get('Fracture', 0):.3f}"
        )

        if is_best:
            best_gold_auc = gold_macro
            best_val_macro = val_macro
            best_preds = preds
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_auc": val_macro,
                    "gold_macro_auc": gold_macro,
                    "val_per_label": val_per_label,
                    "gold_per_label": gold_per_label,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            np.save(val_pred_path, best_preds)

    console.print(f"\n[bold green]✓ Training Complete ({args.run_name})![/bold green]")
    console.print(f"Peak Val Macro AUC: [bold]{best_val_macro:.4f}[/bold] | Gold AUC: [bold]{best_gold_auc:.4f}[/bold]")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Predictions: {val_pred_path}")


if __name__ == "__main__":
    main()
