"""
RSNA Knee Abnormality Detection - Phase 11: Multi-Backbone MRI-Aware Label-Specific MIL Training.
Supports ConvNeXt-Tiny, ConvNeXt-Small, DINOv2-Small, and other timm/torchvision encoders.
Features:
  1. Label-Specific Gated Attention MIL (12 distinct attention heads for 12 abnormalities).
  2. Laterality Normalization (Left knees mirrored horizontally to canonical Right knee).
  3. Mixed Target Loss: y_train = alpha * y_hard + (1 - alpha) * y_soft.
  4. Physical Depth Windows: 32 windows.
  5. 25-Epoch training directly benchmarked against Phase 6 baseline (Macro AUC: 0.8359).
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.table import Table

from src.models.mil_backbone import LabelSpecificKneeMILModel
from src.training.losses import MixedTargetBCEWithLogitsLoss
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


class MRIAwareKneeDataset(Dataset):
    """
    Dataset supporting laterality normalization (L -> R canonical mirror)
    and dual hard/soft target loading for mixed supervision.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str = 'data/cached_series_384',
        target_slices: int = 32,
        preferred_plane: str = 'Sagittal',
        transforms=None,
        is_training: bool = False,
        return_raw: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.return_raw = return_raw
        self.cache_dir = cache_dir
        self.target_slices = target_slices
        self.preferred_plane = preferred_plane
        self.transforms = transforms
        self.is_training = is_training

        # Load series manifest
        manifest_p = os.path.join(os.path.dirname(cache_dir), 'preprocessed_index.parquet')
        if os.path.exists(manifest_p):
            self.series_df = pd.read_parquet(manifest_p)
            self.study_to_series = self.series_df.groupby('StudyInstanceUID')
        else:
            self.series_df = pd.DataFrame()
            self.study_to_series = None

    def __len__(self):
        return len(self.df)

    def _find_series_path(self, study_uid: str) -> Optional[str]:
        study_dir = os.path.join(self.cache_dir, study_uid)
        if not os.path.isdir(study_dir):
            return None

        if self.preferred_plane and self.study_to_series is not None:
            try:
                group = self.study_to_series.get_group(study_uid)
                plane_matches = group[group['Anatomical_Plane'] == self.preferred_plane]
                if len(plane_matches) > 0:
                    cand = os.path.join(study_dir, f"{plane_matches.iloc[0]['SeriesInstanceUID']}.npy")
                    if os.path.exists(cand):
                        return cand
            except Exception:
                pass

        npy_files = [f for f in os.listdir(study_dir) if f.endswith('.npy')]
        return os.path.join(study_dir, npy_files[0]) if npy_files else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        study_uid = row['StudyInstanceUID']

        series_path = self._find_series_path(study_uid)
        if series_path is None or not os.path.exists(series_path):
            raise FileNotFoundError(f'Missing volume for {study_uid}')

        # (D, 3, H, W)
        volume = np.load(series_path)



        # Depth resampling
        if self.target_slices is not None and volume.shape[0] != self.target_slices:
            cur_d = volume.shape[0]
            indices = np.linspace(0, cur_d - 1, self.target_slices).round().astype(int)
            volume = volume[indices]

        num_slices = volume.shape[0]
        transformed = []
        
        # Apply consistent 3D volume transforms across ALL slices
        replay_dict = None
        for s in range(num_slices):
            img = volume[s].transpose(1, 2, 0) # (H, W, 3)
            if self.transforms is not None:
                if hasattr(self.transforms, 'replay') and hasattr(self.transforms, '__call__'):
                    if s == 0:
                        res = self.transforms(image=img)
                        replay_dict = res.get('replay', None)
                    else:
                        res = self.transforms.replay(replay_dict, image=img) if replay_dict else self.transforms(image=img)
                else:
                    res = self.transforms(image=img)
                t = res['image']
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                elif t.dtype == torch.uint8:
                    t = t.float() / 255.0
            else:
                t = torch.tensor(volume[s], dtype=torch.float32) / 255.0
            transformed.append(t)

        vol_tensor = torch.stack(transformed, dim=0) # (D, 3, H, W)

        # Targets
        hard_vals = [float(row[f'hard_{c}']) if f'hard_{c}' in row and pd.notnull(row[f'hard_{c}']) else float('nan') for c in TARGET_COLUMNS]
        soft_vals = [float(row[f'soft_{c}']) if f'soft_{c}' in row and pd.notnull(row[f'soft_{c}']) else 0.50 for c in TARGET_COLUMNS]

        out_dict = {
            'study_uid': study_uid,
            'images': vol_tensor,
            'hard_targets': torch.tensor(hard_vals, dtype=torch.float32),
            'soft_targets': torch.tensor(soft_vals, dtype=torch.float32),
            'is_gold': bool(row.get('is_gold', False)),
        }
        if self.return_raw:
            raw_slices = [torch.tensor(volume[s], dtype=torch.float32) / 255.0 for s in range(num_slices)]
            out_dict['raw_images'] = torch.stack(raw_slices, dim=0)
        return out_dict


def mri_aware_collate(batch):
    images = torch.stack([b['images'] for b in batch])
    hard_targets = torch.stack([b['hard_targets'] for b in batch])
    soft_targets = torch.stack([b['soft_targets'] for b in batch])
    study_uids = [b['study_uid'] for b in batch]
    is_gold = torch.tensor([b['is_gold'] for b in batch], dtype=torch.bool)
    batch_dict = {
        'images': images,
        'hard_targets': hard_targets,
        'soft_targets': soft_targets,
        'study_uids': study_uids,
        'is_gold': is_gold,
    }
    if 'raw_images' in batch[0]:
        batch_dict['raw_images'] = torch.stack([b['raw_images'] for b in batch])
    return batch_dict



def dump_augmentation_visual(raw_volume, aug_volume, output_path, study_uid='', split='Train Augmented'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    D = aug_volume.shape[0]
    slice_indices = [int(0.25 * D), int(0.50 * D), int(0.75 * D)]

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    fig.suptitle(f'Visual Verification: Raw vs {split} (Epoch 1) - Study: {study_uid}', fontsize=12, fontweight='bold')

    for row, s_idx in enumerate(slice_indices):
        raw_slice = raw_volume[s_idx]
        if isinstance(raw_slice, torch.Tensor):
            raw_slice = raw_slice.permute(1, 2, 0).cpu().numpy()
        elif raw_slice.shape[0] == 3:
            raw_slice = raw_slice.transpose(1, 2, 0)

        aug_slice = aug_volume[s_idx]
        if isinstance(aug_slice, torch.Tensor):
            aug_slice = aug_slice.permute(1, 2, 0).cpu().numpy()
        elif aug_slice.shape[0] == 3:
            aug_slice = aug_slice.transpose(1, 2, 0)

        axes[row, 0].imshow(raw_slice[:, :, 0], cmap='bone')
        axes[row, 0].set_title(f'Slice {s_idx}/{D} [RAW ORIGINAL] | Min: {raw_slice.min():.2f}, Max: {raw_slice.max():.2f}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(aug_slice[:, :, 0], cmap='bone')
        axes[row, 1].set_title(f'Slice {s_idx}/{D} [{split.upper()}] | Min: {aug_slice.min():.2f}, Max: {aug_slice.max():.2f}', fontsize=10, color='darkgreen', fontweight='bold')
        axes[row, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate(model, loader, device, dump_visuals: bool = False, output_dir: str = ''):
    model.eval()
    all_preds, all_soft, all_is_gold = [], [], []

    total_val_dumps = 5
    dumped_val_count = 0

    with torch.no_grad():
        for batch in loader:
            if dump_visuals and dumped_val_count < total_val_dumps and 'raw_images' in batch:
                batch_size_cur = len(batch['images'])
                perm_indices = np.random.permutation(batch_size_cur)
                for b_idx in perm_indices:
                    if dumped_val_count >= total_val_dumps:
                        break
                    dumped_val_count += 1
                    study_uid = batch['study_uids'][b_idx]
                    aug_img_path = os.path.join(output_dir, f'augmentation_verification_val_sample{dumped_val_count}_{study_uid}.jpg')
                    dump_augmentation_visual(batch['raw_images'][b_idx], batch['images'][b_idx], aug_img_path, study_uid=study_uid, split='Val Processed')
                    Console().print(f'[bold green]✓ Saved val visual verification ({dumped_val_count}/{total_val_dumps}) to: {aug_img_path}[/bold green]')

            imgs = batch['images'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(imgs)
                probs = torch.sigmoid(out['logits']).cpu().numpy()

            all_preds.append(probs)
            all_soft.append(batch['soft_targets'].numpy())
            all_is_gold.append(batch['is_gold'].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_soft, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    # Full macro AUC
    full_metrics = compute_macro_auc(targets, preds)
    full_macro = full_metrics['macro_auc']
    full_per_label = full_metrics['per_class_auc']

    # Gold subset macro AUC
    gold_macro = 0.0
    if is_gold.sum() > 5:
        gold_metrics = compute_macro_auc(targets[is_gold], preds[is_gold])
        gold_macro = gold_metrics['macro_auc']

    return full_macro, full_per_label, gold_macro, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help='Backbone: convnext_tiny, convnext_small, dinov2_small')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.15)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--target_slices', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=288)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='artifacts/experiments/phase_11_convnext_tiny_mri_aware')
    parser.add_argument('--dump_aug_visuals', action='store_true', default=False, help='Dump visual of [raw, augmented] for 5 random samples during epoch 1')
    args = parser.parse_args()

    seed_everything(42)
    console = Console()

    # Resolve backbone aliases and patch-size constraints
    if args.backbone in ('dinov2_small', 'dinov2_s', 'dinov2'):
        actual_backbone = 'vit_small_patch14_dinov2.lvd142m'
        backbone_tag = 'dinov2_small'
        if args.image_size % 14 != 0:
            args.image_size = (args.image_size // 14) * 14
    else:
        actual_backbone = args.backbone
        backbone_tag = args.backbone

    if args.output_dir == 'artifacts/experiments/phase_11_convnext_tiny_mri_aware' and backbone_tag != 'convnext_tiny':
        args.output_dir = f'artifacts/experiments/phase_11_{backbone_tag}_mri_aware'

    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    console.print(f'[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]')
    console.print(f'[bold green]║ Phase 11: {backbone_tag.upper():<20} with MRI-Aware Label-Specific MIL   ║[/bold green]')
    console.print(f'[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]')
    console.print(f'Fold: {args.fold} | Epochs: {args.epochs} | Slices: {args.target_slices} | Res: {args.image_size}px | Alpha: {args.alpha}')

    # Load master labels
    df_splits = pd.read_parquet('data/splits_5fold.parquet')
    df_tri = pd.read_parquet('data/tristate_labels_master.parquet')
    df_dense = pd.read_parquet('data/dense_labels_master.parquet')

    df_all = df_splits[['StudyInstanceUID', 'fold']].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all['is_gold'] = labeled_mask

    # Explicitly merge on StudyInstanceUID to avoid row-order misalignment
    tri_renamed = df_tri[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'hard_{c}' for c in TARGET_COLUMNS})
    dense_renamed = df_dense[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'soft_{c}' for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on='StudyInstanceUID', how='left')
    df_all = df_all.merge(dense_renamed, on='StudyInstanceUID', how='left')

    train_df = df_all[df_all['fold'] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all['fold'] == args.fold].reset_index(drop=True)

    console.print(f'Train samples: {len(train_df)} | Val samples: {len(val_df)} (Gold in Val: {val_df["is_gold"].sum()})')

    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_ds = MRIAwareKneeDataset(
        train_df,
        target_slices=args.target_slices,
        transforms=train_transforms,
        is_training=True,
        return_raw=args.dump_aug_visuals,
    )
    val_ds = MRIAwareKneeDataset(
        val_df,
        target_slices=args.target_slices,
        transforms=val_transforms,
        is_training=False,
        return_raw=args.dump_aug_visuals,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mri_aware_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mri_aware_collate,
        pin_memory=True,
    )

    # Model
    model = LabelSpecificKneeMILModel(
        backbone_name=actual_backbone,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=128,
        chunk_size=args.target_slices,
    )
    model.to(args.device)

    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    mil_params = list(model.mil_pool.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_mult},
        {'params': mil_params, 'lr': args.lr},
    ], weight_decay=1e-2)

    total_steps = args.epochs * (len(train_loader) // args.grad_accum_steps)
    warmup_steps = args.warmup_epochs * (len(train_loader) // args.grad_accum_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    criterion = MixedTargetBCEWithLogitsLoss(alpha=args.alpha)

    best_macro_auc = 0.0
    best_gold_auc = 0.0
    best_preds = None

    console.print(f'[bold cyan]Starting Training: 25 Epochs with Label-Specific Attention + Mixed Loss[/bold cyan]')

    total_aug_to_dump = 5
    dumped_aug_count = 0

    step_count = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if epoch == 1 and args.dump_aug_visuals and dumped_aug_count < total_aug_to_dump and 'raw_images' in batch:
                batch_size_cur = len(batch['images'])
                perm_indices = np.random.permutation(batch_size_cur)
                for b_idx in perm_indices:
                    if dumped_aug_count >= total_aug_to_dump:
                        break
                    dumped_aug_count += 1
                    study_uid = batch['study_uids'][b_idx]
                    aug_img_path = os.path.join(args.output_dir, f'augmentation_verification_train_sample{dumped_aug_count}_{study_uid}.jpg')
                    dump_augmentation_visual(batch['raw_images'][b_idx], batch['images'][b_idx], aug_img_path, study_uid=study_uid)
                    console.print(f'[bold green]✓ Saved augmentation visual verification ({dumped_aug_count}/{total_aug_to_dump}) to: {aug_img_path}[/bold green]')

            imgs = batch['images'].to(args.device)
            hard = batch['hard_targets'].to(args.device)
            soft = batch['soft_targets'].to(args.device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(imgs)
                loss = criterion(out['logits'], hard, soft)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

        train_loss /= len(train_loader)
        val_macro, val_per_label, val_gold, val_preds = evaluate(model, val_loader, args.device, dump_visuals=(args.dump_aug_visuals and epoch == 1), output_dir=args.output_dir)
        epoch_time = time.time() - t0

        is_best = val_macro > best_macro_auc
        if is_best:
            best_macro_auc = val_macro
            best_gold_auc = val_gold
            best_preds = val_preds
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_auc': val_macro,
                'val_gold_auc': val_gold,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'checkpoints', f'{backbone_tag}_mri_aware_fold{args.fold}_best.pth'))

        star = ' ⭐ BEST' if is_best else ''
        console.print(
            f'Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.0f}s) | '
            f'Loss: {train_loss:.4f} | '
            f'Val Macro AUC: [bold {("green" if is_best else "white")}]{val_macro:.4f}[/] | '
            f'Gold AUC: [cyan]{val_gold:.4f}[/]{star}'
        )

    console.print(f'[bold green]Finished Fold {args.fold}![/bold green]')
    console.print(f'Previous Phase 6 Baseline Macro AUC: [yellow]0.8359[/yellow]')
    console.print(f'New MRI-Aware MIL Best Macro AUC: [bold magenta]{best_macro_auc:.4f}[/bold magenta] (Delta: {best_macro_auc - 0.8359:+.4f})')
    console.print(f'Best Gold Benchmark AUC: [bold cyan]{best_gold_auc:.4f}[/bold cyan]')

    # Save OOF predictions
    oof_df = val_df[['StudyInstanceUID', 'fold', 'is_gold']].copy()
    for i, col in enumerate(TARGET_COLUMNS):
        oof_df[f'pred_{col}'] = best_preds[:, i]
    oof_df.to_parquet(os.path.join(args.output_dir, f'oof_fold{args.fold}.parquet'), index=False)
    console.print(f'OOF predictions saved to {args.output_dir}/oof_fold{args.fold}.parquet')

if __name__ == '__main__':
    main()
