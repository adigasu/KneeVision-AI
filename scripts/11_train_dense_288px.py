"""
RSNA Knee Abnormality Detection - High-Resolution (288x288) Dense Supervised Retraining.
Trains 2.5D MIL backbones with Confidence-Weighted Soft BCE Loss (w = 2 * |p - 0.5|),
100% gradient efficiency, differential learning rates, Cosine Annealing, and TTA.
"""

import os
import argparse
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.data.dataset import KneeMRIDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.mil_backbone import KneeMILModel
from src.training.losses import ConfidenceWeightedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


def mil_padded_collate(batch):
    """Batches variable-depth series volumes into a padded tensor with a boolean mask."""
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


def evaluate_model_dense(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    use_amp: bool = True,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Evaluates model predictions with optional Test-Time Augmentation (Original + Horizontal Flip)."""
    model.eval()
    all_preds = []
    all_targets = []
    all_uids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            targets = batch["targets"].cpu().numpy()
            uids = batch["study_uids"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images, mask=masks)
                probs = torch.sigmoid(out["logits"])

                if use_tta:
                    images_flipped = torch.flip(images, dims=[-1])
                    out_flipped = model(images_flipped, mask=masks)
                    probs_flipped = torch.sigmoid(out_flipped["logits"])
                    probs = 0.5 * (probs + probs_flipped)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets)
            all_uids.extend(uids)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0), all_uids


def train_single_fold_dense(
    fold: int,
    args: argparse.Namespace,
    df_dense: pd.DataFrame,
    df_index: pd.DataFrame,
    gold_uids: set,
    console: Console,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    console.print(f"\n[bold cyan]══════════════ Starting Dense Retraining: Fold {fold} ══════════════[/bold cyan]")
    start_time = time.time()

    train_df = df_dense[df_dense["fold"] != fold].reset_index(drop=True)
    val_df = df_dense[df_dense["fold"] == fold].reset_index(drop=True)

    if args.dry_run:
        train_df = train_df.iloc[:24]
        val_df = val_df.iloc[:16]

    console.print(f"Train Studies: [green]{len(train_df):,}[/green] | Val Studies: [yellow]{len(val_df):,}[/yellow]")

    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_dataset = KneeMRIDataset(
        df=train_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        target_slices=args.target_slices,
        preferred_plane=args.plane,
        transforms=train_transforms,
        is_training=True,
    )
    val_dataset = KneeMRIDataset(
        df=val_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        target_slices=args.target_slices,
        preferred_plane=args.plane,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mil_padded_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mil_padded_collate,
    )

    model = KneeMILModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=args.mil_hidden_dim,
        dropout=args.dropout,
        in_chans=3,
    ).to(device)

    # Confidence-Weighted Soft BCE Loss (w = 2 * |p - 0.5|)
    criterion = ConfidenceWeightedBCEWithLogitsLoss()

    # Differential Learning Rates: 0.2x for backbone features, 1.0x for attention & head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name or "feature_extractor" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # ── Learning rates (differential or default) ────────────────────────────
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr * 0.2
    head_lr     = args.head_lr     if args.head_lr     is not None else args.lr

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": args.weight_decay},
            {"params": head_params,    "lr": head_lr,     "weight_decay": args.weight_decay},
        ]
    )

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 1
    if getattr(args, 'resume', None) and os.path.exists(args.resume):
        console.print(f"[yellow]Resuming from checkpoint: {args.resume}[/yellow]")
        resume_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        console.print(f"  Resumed at epoch {start_epoch - 1} → continuing from epoch {start_epoch}")

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = len(train_loader) * getattr(args, 'warmup_epochs', 0) // args.grad_accum_steps

    def _lr_lambda(current_step: int):
        """Linear warmup then cosine decay."""
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    # Fast-forward scheduler if resuming
    if start_epoch > 1:
        steps_done = len(train_loader) * (start_epoch - 1) // args.grad_accum_steps
        for _ in range(steps_done):
            scheduler.step()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_val_auc = 0.0
    best_gold_auc = 0.0
    best_metrics = {}
    best_checkpoint_path = os.path.join(
        args.checkpoint_dir, f"dense_{args.image_size}px_{args.backbone}_{args.plane.lower()}_fold{fold}_best.pth"
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(images, mask=masks)
                loss = criterion(out["logits"], targets) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.grad_accum_steps
            step_count += 1

        train_loss = running_loss / max(1, step_count)

        # Validation with TTA
        val_preds, val_targets, val_uids = evaluate_model_dense(model, val_loader, device=device, use_amp=args.amp, use_tta=True)

        # 1. Full Cohort OOF Macro AUC
        full_metrics = compute_macro_auc(y_true=val_targets, y_pred=val_preds)
        full_auc = full_metrics["macro_auc"]

        # 2. Gold Benchmark Human-Annotated Subset AUC
        gold_indices = [i for i, uid in enumerate(val_uids) if uid in gold_uids]
        if len(gold_indices) >= 4:
            gold_preds = val_preds[gold_indices]
            gold_targets = val_targets[gold_indices]
            gold_metrics = compute_macro_auc(y_true=gold_targets, y_pred=gold_preds)
            gold_auc = gold_metrics["macro_auc"]
        else:
            gold_auc = float("nan")

        console.print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: [magenta]{train_loss:.4f}[/magenta] | "
            f"Val Macro AUC: [bold blue]{full_auc:.4f}[/bold blue] | "
            f"Gold Subset AUC: [bold yellow]{gold_auc:.4f}[/bold yellow] | "
            f"LR: [dim]{optimizer.param_groups[1]['lr']:.2e}[/dim]"
        )

        eval_score = full_auc if np.isnan(gold_auc) else (0.4 * full_auc + 0.6 * gold_auc)
        if eval_score > best_val_auc or epoch == 1:
            best_val_auc = eval_score
            best_gold_auc = gold_auc
            best_metrics = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "full_macro_auc": full_auc,
                "gold_macro_auc": gold_auc,
                "per_class_auc": full_metrics.get("per_class_auc", {}),
                "checkpoint_path": best_checkpoint_path,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metrics": best_metrics,
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "plane": args.plane,
                },
                best_checkpoint_path,
            )

    elapsed = time.time() - start_time
    console.print(
        f"[bold green]✓ Fold {fold} Complete in {elapsed/60:.1f}m - Best Full AUC: {best_metrics.get('full_macro_auc', 0.0):.4f} | "
        f"Gold AUC: {best_metrics.get('gold_macro_auc', float('nan')):.4f}[/bold green]"
    )
    return best_metrics


def run_dense_training():
    parser = argparse.ArgumentParser(description="KneeVision-AI Dense 288px Supervised Training")
    parser.add_argument("--labels_path", type=str, default="./data/dense_labels_master.parquet")
    parser.add_argument("--index_path", type=str, default="./data/preprocessed_index.parquet")
    parser.add_argument("--splits_path", type=str, default="./data/splits_5fold.parquet")
    parser.add_argument("--cache_dir", type=str, default="./data/preprocessed_256")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--fold", type=int, default=-1, help="0-4 for single fold, or -1 for all folds")
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--plane", type=str, default="Sagittal")
    parser.add_argument("--image_size", type=int, default=288)
    parser.add_argument("--target_slices", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Linear LR warmup epochs before cosine annealing")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="Override backbone LR (default: lr * 0.2)")
    parser.add_argument("--head_lr", type=float, default=None,
                        help="Override head LR (default: lr)")
    parser.add_argument("--use_tta", action="store_true", default=False,
                        help="Enable horizontal-flip TTA during validation")
    parser.add_argument("--checkpoint_prefix", type=str, default=None,
                        help="Override checkpoint filename prefix")
    args = parser.parse_args()

    seed_everything(args.seed)
    console = Console()

    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

    console.print("[bold green]=== KneeVision-AI: Phase 6 Dense Supervised Retraining (288px) ===[/bold green]")
    console.print(
        f"Backbone: [cyan]{args.backbone}[/cyan] | Resolution: [cyan]{args.image_size}x{args.image_size}[/cyan] | "
        f"Plane: [cyan]{args.plane}[/cyan] | Epochs: [cyan]{args.epochs}[/cyan] | Batch Size: [cyan]{args.batch_size}[/cyan] | "
        f"Loss: [cyan]ConfidenceWeightedBCE (w = 2|p-0.5|)[/cyan] | Device: [cyan]{device}[/cyan]"
    )

    df_dense = pd.read_parquet(args.labels_path) if args.labels_path.endswith(".parquet") else pd.read_csv(args.labels_path)
    df_index = pd.read_parquet(args.index_path) if args.index_path.endswith(".parquet") else pd.read_csv(args.index_path)
    df_splits = pd.read_parquet(args.splits_path) if args.splits_path.endswith(".parquet") else pd.read_csv(args.splits_path)

    gold_uids = set(df_splits[df_splits[TARGET_COLUMNS].notnull().any(axis=1)]["StudyInstanceUID"])
    console.print(f"Loaded [bold green]{len(df_dense):,}[/bold green] Dense Studies (Gold Human Studies: {len(gold_uids)})")

    folds_to_run = [args.fold] if args.fold >= 0 else list(range(5))
    all_fold_results = []

    for f in folds_to_run:
        res = train_single_fold_dense(f, args, df_dense, df_index, gold_uids, console, device=device)
        all_fold_results.append(res)

    table = Table(title=f"Phase 6 Dense 288px Cross-Validation ({args.backbone} - {args.plane})")
    table.add_column("Fold", style="cyan")
    table.add_column("Epoch", justify="right", style="dim")
    table.add_column("Full Cohort Macro AUC", justify="right", style="magenta")
    table.add_column("Gold Benchmark AUC", justify="right", style="yellow")

    full_aucs = []
    gold_aucs = []
    for res in all_fold_results:
        f_auc = res.get("full_macro_auc", 0.0)
        g_auc = res.get("gold_macro_auc", float("nan"))
        table.add_row(
            f"Fold {res.get('fold', 0)}",
            f"{res.get('epoch', 0)}",
            f"{f_auc:.4f}",
            f"{g_auc:.4f}" if not np.isnan(g_auc) else "N/A",
        )
        full_aucs.append(f_auc)
        if not np.isnan(g_auc):
            gold_aucs.append(g_auc)

    table.add_section()
    table.add_row(
        "[bold]Mean 5-Fold Result[/bold]",
        "-",
        f"[bold magenta]{np.mean(full_aucs):.4f}[/bold magenta]",
        f"[bold yellow]{np.mean(gold_aucs):.4f}[/bold yellow]" if gold_aucs else "N/A",
    )
    console.print(table)


if __name__ == "__main__":
    run_dense_training()
