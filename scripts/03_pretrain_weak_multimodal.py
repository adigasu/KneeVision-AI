"""
RSNA Knee Abnormality Detection - Weak Multimodal Alignment Pretraining Script.
Performs self-supervised vision-language contrastive alignment (BETO + 2.5D ConvNeXt)
combined with multi-task 12-target pseudo-supervision on 4,349 unannotated studies.
"""

import os
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.data.multimodal_dataset import MultimodalKneeDataset
from src.data.transforms import get_training_transforms
from src.models.weak_multimodal import DualStreamKneeModel
from src.training.contrastive_loss import MultimodalJointLoss
from src.utils.common import seed_everything


def multimodal_collate_fn(batch):
    """Batches images, tokenized reports, and pseudo labels."""
    images = torch.stack([item["images"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    pseudo_targets = torch.stack([item["pseudo_targets"] for item in batch])
    study_uids = [item["study_uid"] for item in batch]

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pseudo_targets": pseudo_targets,
        "study_uids": study_uids,
    }


def run_pretraining():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_path", type=str, default="data/splits_5fold.parquet")
    parser.add_argument("--pseudo_path", type=str, default="data/pseudo_labels_master.parquet")
    parser.add_argument("--index_path", type=str, default="data/preprocessed_index.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/preprocessed_256")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--text_model", type=str, default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--plane", type=str, default="Sagittal")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--target_slices", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda_pseudo", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    console = Console()

    console.print("[bold green]=== KneeVision-AI: Weak Multimodal Vision-Language Pretraining ===[/bold green]")
    console.print(f"Vision Backbone: [cyan]{args.backbone}[/cyan] | Text Encoder: [cyan]{args.text_model}[/cyan]")
    console.print(f"Plane: [cyan]{args.plane}[/cyan] | Target Slices: [cyan]{args.target_slices}[/cyan] | Epochs: [cyan]{args.epochs}[/cyan]")

    # 1. Load splits and filter to available cached studies
    df_splits = pd.read_parquet(args.splits_path) if args.splits_path.endswith(".parquet") else pd.read_csv(args.splits_path)
    
    # Check cached studies in cache_dir
    cached_study_uids = set(os.listdir(args.cache_dir)) if os.path.exists(args.cache_dir) else set()
    df_available = df_splits[df_splits["StudyInstanceUID"].isin(cached_study_uids)].reset_index(drop=True)
    console.print(f"Available Cached Studies for Pretraining: [bold yellow]{len(df_available):,}[/bold yellow] / {len(df_splits):,}")

    if len(df_available) < 10:
        console.print("[bold red]Error: Less than 10 preprocessed studies found in cache. Ensure preprocessing is complete.[/bold red]")
        return

    # 2. Setup Dataset & Dataloader
    train_transforms = get_training_transforms(image_size=args.image_size)
    dataset = MultimodalKneeDataset(
        df=df_available,
        pseudo_df=args.pseudo_path,
        cache_dir=args.cache_dir,
        series_df=args.index_path if os.path.exists(args.index_path) else None,
        tokenizer_name=args.text_model,
        target_slices=args.target_slices,
        preferred_plane=args.plane,
        transforms=train_transforms,
        is_training=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # 3. Initialize Model
    model = DualStreamKneeModel(
        backbone_name=args.backbone,
        text_model_name=args.text_model,
        embed_dim=args.embed_dim,
        num_classes=12,
        mil_hidden_dim=128,
        use_grad_checkpointing=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Differential learning rate (smaller for pretrained text transformer, higher for vision/heads)
    optimizer = torch.optim.AdamW([
        {"params": model.vision_backbone.parameters(), "lr": args.lr},
        {"params": model.mil_pool.parameters(), "lr": args.lr},
        {"params": model.vision_projection.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
        {"params": model.text_encoder.parameters(), "lr": args.lr * 0.2},
        {"params": [model.logit_scale], "lr": args.lr * 0.1},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    criterion = MultimodalJointLoss(lambda_pseudo=args.lambda_pseudo)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    # 4. Pretraining Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_clip = 0.0
        total_asl = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Pretrain]")
        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pseudo_targets = batch["pseudo_targets"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images, input_ids, attention_mask)
                loss_dict = criterion(outputs, pseudo_targets)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_clip += loss_dict["clip_loss"].item()
            total_asl += loss_dict["asl_loss"].item()

            pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "CLIP": f"{loss_dict['clip_loss'].item():.4f}",
                "ASL": f"{loss_dict['asl_loss'].item():.4f}",
                "Tau": f"{outputs['logit_scale'].item():.2f}",
            })

        scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_clip = total_clip / len(loader)
        avg_asl = total_asl / len(loader)

        console.print(f"Epoch {epoch:02d}/{args.epochs:02d} | Avg Total Loss: [bold cyan]{avg_loss:.4f}[/bold cyan] | CLIP InfoNCE: [green]{avg_clip:.4f}[/green] | Pseudo ASL: [magenta]{avg_asl:.4f}[/magenta]")

        # Save latest & best pretrained weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save complete dual-stream checkpoint
            full_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_multimodal_full.pth")
            torch.save(model.state_dict(), full_ckpt_path)

            # Export vision-only weights compatible with KneeMILModel
            vision_weights_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_weak_pretrained.pth")
            vision_state = model.get_vision_backbone_state_dict()
            torch.save({"model_state_dict": vision_state, "pretrain_epoch": epoch, "pretrain_loss": avg_loss}, vision_weights_path)
            console.print(f"✓ Saved best pretrained vision backbone to [bold green]{vision_weights_path}[/bold green]")

    console.print("\n[bold green]✓ Weak Multimodal Pretraining Completed Successfully![/bold green]")


if __name__ == "__main__":
    run_pretraining()
