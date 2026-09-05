"""
RSNA Knee Abnormality Detection - 5-Fold Cross-Validation 2.5D MIL Training Script.
Trains and evaluates 2.5D ConvNeXt + Gated Attention MIL models across the 5 cross-validation splits.
Supports optional fine-tuning from weak multimodal pretrained weights.
"""

import os
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.data.dataset import KneeMRIDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.mil_backbone import KneeMILModel
from src.training.losses import AsymmetricLoss
from src.training.trainer import KneeTrainer
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


def mil_padded_collate(batch):
    """
    Batches variable-slice series into a padded tensor with a boolean mask.
    """
    depths = [item["images"].shape[0] for item in batch]
    max_d = max(depths)
    b_size = len(batch)
    _, channels, h, w = batch[0]["images"].shape

    padded_images = torch.zeros(b_size, max_d, channels, h, w, dtype=torch.float32)
    masks = torch.zeros(b_size, max_d, dtype=torch.bool)
    targets_stack = torch.stack([item["targets"] for item in batch])

    for i, item in enumerate(batch):
        d = item["images"].shape[0]
        padded_images[i, :d] = item["images"]
        masks[i, :d] = True

    return {
        "images": padded_images,
        "mask": masks,
        "targets": targets_stack,
        "study_uids": [item["study_uid"] for item in batch],
    }


def train_single_fold(
    fold: int,
    args: argparse.Namespace,
    df_splits: pd.DataFrame,
    df_index: pd.DataFrame,
    console: Console,
) -> Dict[str, Any]:
    console.print(f"\n[bold blue]━━━━━━━━━━━━━━━━━━━━ FOLD {fold} / 4 ━━━━━━━━━━━━━━━━━━━━[/bold blue]")

    # Partition train vs validation for this fold
    df_labeled = df_splits[df_splits[TARGET_COLUMNS].notnull().any(axis=1)].copy()
    train_df = df_labeled[df_labeled["fold"] != fold].reset_index(drop=True)
    val_df = df_labeled[df_labeled["fold"] == fold].reset_index(drop=True)

    console.print(f"Train Studies: [green]{len(train_df)}[/green] | Validation Studies: [yellow]{len(val_df)}[/yellow]")

    # Create Datasets & Transforms
    train_transforms = get_training_transforms(image_size=args.image_size)
    val_transforms = get_validation_transforms(image_size=args.image_size)

    train_ds = KneeMRIDataset(
        df=train_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        preferred_plane=args.plane,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = KneeMRIDataset(
        df=val_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        preferred_plane=args.plane,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_padded_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mil_padded_collate,
        pin_memory=True,
    )

    # Initialize Model
    model = KneeMILModel(
        backbone_name=args.backbone,
        pretrained=(args.pretrained_weights is None),
        num_classes=12,
        mil_hidden_dim=args.mil_hidden_dim,
        dropout=args.dropout,
    )

    # Load Weak Multimodal Pretrained Weights if specified
    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
        console.print(f"[bold yellow]Loading Weak Multimodal Pretrained Weights from: {args.pretrained_weights}[/bold yellow]")
        ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        weights = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(weights, strict=False)
        console.print(f"✓ Loaded weights. Missing keys: {len(missing)} (e.g. classifier), Unexpected: {len(unexpected)}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)

    exp_tag = f"{args.backbone}_{args.plane.lower()}"
    if args.pretrained_weights is not None:
        exp_tag += "_weak_ft"

    # Trainer Engine
    trainer = KneeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        use_amp=args.amp,
        use_ema=args.ema,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=exp_tag,
    )

    fit_results = trainer.fit(num_epochs=args.epochs, fold=fold)

    # Load best checkpoint and evaluate out-of-fold predictions
    ckpt_path = os.path.join(args.checkpoint_dir, f"{exp_tag}_fold{fold}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model = model.to(trainer.device)

    oof_eval = trainer.evaluate(use_ema_for_eval=False)
    oof_eval["fold"] = fold
    oof_eval["best_macro_auc"] = fit_results["best_macro_auc"]
    return oof_eval


def run_5fold_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_path", type=str, default="data/splits_5fold.parquet")
    parser.add_argument("--index_path", type=str, default="data/preprocessed_index.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/preprocessed_256")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to weakly pretrained checkpoint")
    parser.add_argument("--fold", type=int, default=-1, help="0-4 for single fold, or -1 for all 5 folds")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--plane", type=str, default="Sagittal")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    console = Console()

    console.print("[bold green]=== KneeVision-AI: 2.5D MIL Cross-Validation Training ===[/bold green]")
    console.print(f"Backbone: [cyan]{args.backbone}[/cyan] | Plane: [cyan]{args.plane}[/cyan] | Resolution: [cyan]{args.image_size}x{args.image_size}[/cyan]")
    if args.pretrained_weights:
        console.print(f"Pretrained Weights: [bold yellow]{args.pretrained_weights}[/bold yellow]")

    # Load splits & manifest
    df_splits = pd.read_parquet(args.splits_path) if args.splits_path.endswith(".parquet") else pd.read_csv(args.splits_path)
    df_index = pd.read_parquet(args.index_path) if args.index_path.endswith(".parquet") else pd.read_csv(args.index_path)

    folds_to_run = [args.fold] if args.fold >= 0 else list(range(5))
    oof_results = []

    for f in folds_to_run:
        res = train_single_fold(f, args, df_splits, df_index, console)
        oof_results.append(res)

    # Summary Table
    table = Table(title=f"5-Fold Cross Validation Results ({args.backbone} - {args.plane})")
    table.add_column("Fold", style="cyan")
    table.add_column("Best Val Macro AUC", justify="right", style="magenta")

    macro_aucs = []
    for res in oof_results:
        table.add_row(f"Fold {res['fold']}", f"{res['best_macro_auc']:.4f}")
        macro_aucs.append(res["best_macro_auc"])

    mean_auc = np.mean(macro_aucs)
    table.add_section()
    table.add_row("[bold]Mean 5-Fold Macro AUC[/bold]", f"[bold green]{mean_auc:.4f}[/bold green]")
    console.print(table)


if __name__ == "__main__":
    run_5fold_training()
