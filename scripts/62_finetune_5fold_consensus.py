"""
RSNA Knee Abnormality Detection - Phase 18: 5-Fold Consensus Finetuning.
Finetunes the converged 5-fold ConvNeXt-Small (384px) checkpoints using new multi-LLM consensus labels:
1. Safe Weight Preservation: Loads existing Phase 16 checkpoints without overwriting them.
   Saves new checkpoints to 'checkpoints/consensus_ft/' or custom --output_dir.
2. Label Smoothing: Applies symmetric/asymmetric label smoothing to soft consensus targets
   to mitigate noise, while preserving strict human annotations on the gold standard set.
3. Anatomy-Preserving Augmentations: Disables HorizontalFlip to prevent Medial vs Lateral
   and Anterior vs Posterior label inversion/corruption.
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.data.triplanar_dataset import TriPlanarKneeDataset, triplanar_collate
from src.models.plane_decoupled_mil import PlaneDecoupledTriPlanarMILModel
from timm.utils import ModelEmaV2


# Optional class bottleneck weighting (can be disabled via --use_bottleneck_weights 0)
BOTTLENECK_WEIGHT_MAP = {
    "MCL": 2.0,
    "Lateral OA": 1.5,
    "Synovitis": 1.5,
    "Fracture": 1.4,
    "Lateral Meniscus": 1.3,
    "PF OA": 1.2,
    "ACL": 1.0,
    "Medial Meniscus": 1.0,
    "Effusion": 1.0,
    "Medial OA": 0.8,
    "Baker's": 0.8,
    "Contusion": 0.8,
}


class LabelSmoothedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Label Smoothing & Gold Upweighting.

    For soft consensus pseudo-labels (silver):
        y_smoothed = y * (1 - label_smoothing) + 0.5 * label_smoothing

    For verified human gold annotations (N=58):
        Preserves crisp ground truth (or minimal smoothing) with upweighting (gold_weight).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        gold_weight: float = 2.0,
        smooth_gold: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.gold_weight = gold_weight
        self.smooth_gold = smooth_gold
        self.class_weights = class_weights
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_gold: Optional[torch.Tensor] = None,
        hard_gold_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (B, num_classes)
            targets: Continuous consensus soft targets in [0, 1] of shape (B, num_classes)
            is_gold: Boolean tensor of shape (B,) indicating human gold cases
            hard_gold_targets: Optional binary ground truth for gold cases
        """
        # Apply label smoothing to consensus targets:
        # y_smooth = y * (1 - eps) + 0.5 * eps
        if self.label_smoothing > 0.0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets.clone()

        weights = torch.ones_like(logits)

        # Handle Gold Standard cases (human radiologist verified)
        if is_gold is not None:
            is_gold_mask = is_gold.view(-1, 1).expand_as(logits).bool()
            # Upweight gold cases
            weights = torch.where(is_gold_mask, torch.full_like(weights, self.gold_weight), weights)

            # If hard gold targets provided, use verified binary labels for gold cases
            if hard_gold_targets is not None:
                has_hard_gold = is_gold_mask & ~torch.isnan(hard_gold_targets)
                if self.smooth_gold and self.label_smoothing > 0.0:
                    gold_clean = hard_gold_targets * (1.0 - self.label_smoothing * 0.5) + (0.5 * self.label_smoothing * 0.5)
                else:
                    gold_clean = hard_gold_targets
                smoothed_targets = torch.where(has_hard_gold, gold_clean, smoothed_targets)

        # Compute per-element BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction="none")
        weighted_loss = weights * bce

        # Optional pathology class weighting
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device).view(1, -1)
            weighted_loss = weighted_loss * cw
            denom = (weights * cw).sum().clamp(min=self.eps)
        else:
            denom = weights.sum().clamp(min=self.eps)

        return weighted_loss.sum() / denom


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, mask=mask)
                probs = torch.sigmoid(out["logits"]).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(batch["soft_targets"].numpy())
            all_is_gold.append(batch["is_gold"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
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
    parser = argparse.ArgumentParser(description="Phase 18 Consensus Finetuning (ConvNeXt-Small 384px)")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0 to 4)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device string (e.g. cuda:0)")
    parser.add_argument("--backbone", type=str, default="convnext_small")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--slices_per_plane", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6, help="Finetuning epochs (default: 6)")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Classifier LR (default: 1e-4)")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1, help="Backbone LR multiplier (default: 0.1 -> 1e-5)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="BCE label smoothing factor (default: 0.1)")
    parser.add_argument("--gold_weight", type=float, default=2.0, help="Upweight factor for gold cases (default: 2.0)")
    parser.add_argument("--smooth_gold", action="store_true", help="Apply smoothing to gold labels (default: False)")
    parser.add_argument("--use_bottleneck_weights", type=int, default=0, choices=[0, 1], help="Enable pathology balancing")
    parser.add_argument("--allow_horizontal_flip", action="store_true", help="Enable horizontal flip (WARNING: inverts medial/lateral)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_checkpoint_dir", type=str, default="checkpoints", help="Directory of existing Phase 16 weights")
    parser.add_argument("--output_dir", type=str, default="checkpoints/consensus_ft", help="Directory to save new finetuned weights")
    parser.add_argument("--consensus_path", type=str, default="../Datasets/rsna-knee-abnormality-detection/consensus_labels.csv")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name identifier")
    # Phase 19 Upgrades:
    parser.add_argument("--use_ema", type=int, default=0, choices=[0, 1], help="Enable ModelEmaV2 shadow model (default: 0)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate (default: 0.999)")
    parser.add_argument("--slices_sag", type=int, default=None, help="Sagittal slice count (default: None -> slices_per_plane)")
    parser.add_argument("--slices_cor", type=int, default=None, help="Coronal slice count (default: None -> slices_per_plane)")
    parser.add_argument("--slices_ax", type=int, default=None, help="Axial slice count (default: None -> slices_per_plane)")
    parser.add_argument("--init_from_consensus", type=int, default=0, choices=[0, 1], help="Warm-start on top of existing Phase 18 consensus weights (default: 0)")
    parser.add_argument("--custom_base_ckpt", type=str, default="", help="Explicit custom base checkpoint file path")

    args = parser.parse_args()

    if not args.run_name:
        tag = "phase19"
        if args.init_from_consensus == 1:
            tag += "_from_consensus"
        if args.slices_sag is not None or args.slices_cor is not None or args.slices_ax is not None:
            s_s = args.slices_sag or args.slices_per_plane
            s_c = args.slices_cor or args.slices_per_plane
            s_a = args.slices_ax or args.slices_per_plane
            tag += f"_asym_{s_s}_{s_c}_{s_a}"
        if args.use_ema == 1:
            tag += f"_ema{args.ema_decay}"
        args.run_name = f"{tag}_{args.backbone}_384px_f{args.fold}"

    console = Console()
    console.print(f"[bold cyan]╔══════════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║  Phase 18: 5-Fold Consensus Finetuning ({args.run_name})               ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print(f"Device: {args.device} | Backbone: {args.backbone} | Fold: {args.fold}")
    console.print(f"Resolution: {args.image_size}x{args.image_size} | Slices: {args.slices_per_plane}x3 = {args.slices_per_plane * 3}")
    console.print(f"Effective Batch: {args.batch_size * args.grad_accum_steps} (Batch {args.batch_size} x Accum {args.grad_accum_steps})")
    console.print(f"Label Smoothing: {args.label_smoothing} | Gold Weight: {args.gold_weight}x | Horizontal Flip: {args.allow_horizontal_flip}")

    # 1. Output directory & Existing Checkpoint Safety
    os.makedirs(args.output_dir, exist_ok=True)
    if args.custom_base_ckpt and os.path.exists(args.custom_base_ckpt):
        base_ckpt_path = args.custom_base_ckpt
    elif args.init_from_consensus == 1:
        # Warm-start on top of existing Phase 18 consensus weights (Folds 0, 1, 3, etc.)
        consensus_dir = args.base_checkpoint_dir if args.base_checkpoint_dir != "checkpoints" else "checkpoints/consensus_ft"
        base_ckpt_path = os.path.join(consensus_dir, f"phase18_consensus_{args.backbone}_384px_f{args.fold}_best.pth")
        if not os.path.exists(base_ckpt_path):
            alt_path = os.path.join("checkpoints/consensus_ft", f"phase18_consensus_{args.backbone}_384px_f{args.fold}_best.pth")
            if os.path.exists(alt_path):
                base_ckpt_path = alt_path
    else:
        base_ckpt_path = os.path.join(args.base_checkpoint_dir, f"phase16_decoupled_{args.backbone}_384px_f{args.fold}_best.pth")
        if not os.path.exists(base_ckpt_path):
            alt_path = os.path.join(args.base_checkpoint_dir, f"phase16_decoupled_{args.backbone}_f{args.fold}_best.pth")
            if os.path.exists(alt_path):
                base_ckpt_path = alt_path

    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}. Verify checkpoint paths.")

    new_checkpoint_path = os.path.join(args.output_dir, f"{args.run_name}_best.pth")
    new_val_pred_path = os.path.join(args.output_dir, f"{args.run_name}_val_preds.npy")

    console.print(f"[bold green]✓ Base Checkpoint Loaded From:[/bold green] {base_ckpt_path}")
    console.print(f"[bold green]✓ New Finetuned Output Saving To:[/bold green] {new_checkpoint_path} (Existing weights protected!)")

    # 2. Data Loading & Merging Consensus Labels
    splits_path = "data/splits_5fold.parquet"
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits parquet not found at {splits_path}")

    df_splits = pd.read_parquet(splits_path)
    df_consensus = pd.read_csv(args.consensus_path)

    # Human gold standard indicator (58 cases)
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_splits["is_gold"] = labeled_mask

    # Rename gold hard labels
    gold_renamed = df_splits[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"hard_{c}" for c in TARGET_COLUMNS})

    # Rename consensus soft labels
    consensus_renamed = df_consensus[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})

    # Assemble master DataFrame
    df_all = df_splits[["StudyInstanceUID", "fold", "is_gold"]].copy()
    df_all = df_all.merge(gold_renamed, on="StudyInstanceUID", how="left")
    df_all = df_all.merge(consensus_renamed, on="StudyInstanceUID", how="left")

    train_df = df_all[df_all["fold"] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == args.fold].reset_index(drop=True)

    console.print(f"Train Studies: {len(train_df)} (Gold: {train_df['is_gold'].sum()})")
    console.print(f"Val Studies:   {len(val_df)} (Gold: {val_df['is_gold'].sum()})")

    # Transforms (HorizontalFlip disabled by default for knee anatomy protection)
    train_transforms = get_training_transforms(
        (args.image_size, args.image_size),
        allow_horizontal_flip=args.allow_horizontal_flip,
    )
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_ds = TriPlanarKneeDataset(
        train_df,
        slices_per_plane=args.slices_per_plane,
        slices_sag=args.slices_sag,
        slices_cor=args.slices_cor,
        slices_ax=args.slices_ax,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = TriPlanarKneeDataset(
        val_df,
        slices_per_plane=args.slices_per_plane,
        slices_sag=args.slices_sag,
        slices_cor=args.slices_cor,
        slices_ax=args.slices_ax,
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

    # 3. Model Initialization & Weight Transfer
    model = PlaneDecoupledTriPlanarMILModel(
        backbone_name=args.backbone,
        pretrained=False,  # Weights loaded directly from checkpoint
        num_classes=len(TARGET_COLUMNS),
        slices_per_plane=args.slices_per_plane,
        slices_sag=args.slices_sag,
        slices_cor=args.slices_cor,
        slices_ax=args.slices_ax,
        chunk_size=16,
        classifier_type="linear",
        head_hidden_dim=256,
    ).to(args.device)

    # Load pre-trained Phase 16 weights
    ckpt_data = torch.load(base_ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    base_val_auc = ckpt_data.get("val_macro_auc", 0.0)
    base_gold_auc = ckpt_data.get("gold_macro_auc", 0.0)
    console.print(f"[bold green]✓ Pre-trained State Loaded Successfully![/bold green] (Base Val AUC: {base_val_auc:.4f}, Base Gold: {base_gold_auc:.4f})")

    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)

    # 4. Loss Function
    class_weights_tensor = None
    if args.use_bottleneck_weights == 1:
        weights_list = [BOTTLENECK_WEIGHT_MAP.get(c, 1.0) for c in TARGET_COLUMNS]
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        console.print(f"Bottleneck Weights: MCL={BOTTLENECK_WEIGHT_MAP['MCL']}x | LatOA=1.5x")
    else:
        console.print("[yellow]Running UNWEIGHTED Label-Smoothed BCE Loss (Equal 1.0x across all 12 classes)[/yellow]")

    criterion = LabelSmoothedBCEWithLogitsLoss(
        label_smoothing=args.label_smoothing,
        gold_weight=args.gold_weight,
        smooth_gold=args.smooth_gold,
        class_weights=class_weights_tensor,
    )

    # 5. Optimizer & Scheduler
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

    # Evaluate baseline before finetuning
    console.print("\n[bold]Evaluating initial baseline performance on validation split...[/bold]")
    init_val_auc, init_per_label, init_gold_auc, _, _ = evaluate(model, val_loader, args.device)
    console.print(f"Initial Checkpoint Val Macro AUC: [cyan]{init_val_auc:.4f}[/cyan] | Initial Gold AUC: [cyan]{init_gold_auc:.4f}[/cyan]\n")

    ema_model = None
    if args.use_ema == 1:
        ema_model = ModelEmaV2(model, decay=args.ema_decay, device=args.device)
        console.print(f"[bold green]✓ Initialized ModelEmaV2 shadow model with decay={args.ema_decay}[/bold green]\n")

    best_val_macro = init_val_auc
    best_gold_auc = init_gold_auc
    best_preds = None

    # 6. Finetuning Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(args.device)
            mask = batch["mask"].to(args.device)
            soft = batch["soft_targets"].to(args.device)
            is_gold = batch["is_gold"].to(args.device)
            hard = batch["hard_targets"].to(args.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, mask=mask)
                loss = criterion(out["logits"], targets=soft, is_gold=is_gold, hard_gold_targets=hard)
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
                if ema_model is not None:
                    ema_model.update(model)

        train_loss /= len(train_loader)
        epoch_time = time.time() - t0

        # Evaluate Raw Model
        val_macro, val_per_label, gold_macro, gold_per_label, preds = evaluate(model, val_loader, args.device)

        mcl_auc = val_per_label.get("MCL", 0.0)
        gold_mcl = gold_per_label.get("MCL", 0.0)
        is_best_raw = val_macro >= best_val_macro

        star = "⭐ BEST (RAW)" if is_best_raw else ""
        console.print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.0f}s] | "
            f"Loss: {train_loss:.4f} | "
            f"Val Macro AUC: {val_macro:.4f} | "
            f"Gold AUC: {gold_macro:.4f} {star}"
        )
        console.print(
            f"   -> MCL: {mcl_auc:.4f} | "
            f"MedOA: {val_per_label.get('Medial OA', 0):.3f} | "
            f"LatOA: {val_per_label.get('Lateral OA', 0):.3f} | "
            f"PFOA: {val_per_label.get('PF OA', 0):.3f} | "
            f"Fracture: {val_per_label.get('Fracture', 0):.3f}"
        )

        # Evaluate EMA Model (if enabled)
        ema_val_macro, ema_gold_macro, ema_preds = 0.0, 0.0, None
        is_best_ema = False
        if ema_model is not None:
            ema_val_macro, ema_val_per_label, ema_gold_macro, ema_gold_per_label, ema_preds = evaluate(
                ema_model.module, val_loader, args.device
            )
            is_best_ema = ema_val_macro >= best_val_macro
            ema_star = "🌟 BEST (EMA)" if is_best_ema else ""
            console.print(
                f"   -> [bold cyan]EMA Shadow Model[/bold cyan] | "
                f"Val Macro AUC: [bold]{ema_val_macro:.4f}[/bold] | "
                f"Gold AUC: [bold]{ema_gold_macro:.4f}[/bold] {ema_star}"
            )
            console.print(
                f"      MCL: {ema_val_per_label.get('MCL', 0):.4f} | "
                f"MedOA: {ema_val_per_label.get('Medial OA', 0):.3f} | "
                f"LatOA: {ema_val_per_label.get('Lateral OA', 0):.3f} | "
                f"PFOA: {ema_val_per_label.get('PF OA', 0):.3f} | "
                f"Fracture: {ema_val_per_label.get('Fracture', 0):.3f}"
            )

        # Save Best Checkpoint (choosing superior between raw and EMA)
        use_ema_weights = (ema_model is not None and ema_val_macro >= val_macro)
        effective_best_val = max(val_macro, ema_val_macro) if ema_model is not None else val_macro
        effective_best_gold = ema_gold_macro if use_ema_weights else gold_macro
        effective_best_preds = ema_preds if use_ema_weights else preds
        effective_state_dict = ema_model.module.state_dict() if use_ema_weights else model.state_dict()

        if effective_best_val >= best_val_macro:
            best_val_macro = effective_best_val
            best_gold_auc = effective_best_gold
            best_preds = effective_best_preds
            weight_source = "EMA" if use_ema_weights else "Raw"
            console.print(f"   [bold yellow]>>> New Peak Model Saved! ({weight_source} Val Macro AUC: {best_val_macro:.4f})[/bold yellow]")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": effective_state_dict,
                    "is_ema": use_ema_weights,
                    "val_macro_auc": best_val_macro,
                    "gold_macro_auc": best_gold_auc,
                    "val_per_label": ema_val_per_label if use_ema_weights else val_per_label,
                    "gold_per_label": ema_gold_per_label if use_ema_weights else gold_per_label,
                    "args": vars(args),
                    "base_checkpoint": base_ckpt_path,
                },
                new_checkpoint_path,
            )
            np.save(new_val_pred_path, best_preds)

    console.print(f"\n[bold green]✓ Finetuning Complete for Fold {args.fold} ({args.run_name})![/bold green]")
    console.print(f"Initial Val Macro AUC: {init_val_auc:.4f} -> Peak Val Macro AUC: [bold]{best_val_macro:.4f}[/bold]")
    console.print(f"Initial Gold AUC:      {init_gold_auc:.4f} -> Peak Gold AUC:      [bold]{best_gold_auc:.4f}[/bold]")
    console.print(f"Checkpoint Saved:     {new_checkpoint_path}")
    console.print(f"Predictions Saved:    {new_val_pred_path}")


if __name__ == "__main__":
    main()
