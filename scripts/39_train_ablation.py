"""
RSNA Knee Abnormality Detection - Phase 14 Independent Ablation Study.
Tests 4 independent arms against the Phase 13 ConvNeXt-Tiny Fold 0 Baseline (Val Macro AUC: 0.8684 | Gold: 0.9117):
1. AFCL Loss alone (--ablation_mode afcl)
2. High-Res Multi-Planar Pipeline alone (--ablation_mode highres)
3. 3D Intra-Plane Sequence Transformer alone (--ablation_mode intra_transformer)
4. Tri-Planar Co-Attention alone (--ablation_mode coattention)
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "net", "KneeVision-AI")))

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.data.triplanar_dataset import TriPlanarKneeDataset, triplanar_collate
from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss
import timm


# ==============================================================================
# Model Components for Ablations
# ==============================================================================

class SinusoidalPositionEncoding1D(nn.Module):
    """1D Sinusoidal Position Encoding for slice depth."""
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class IntraPlaneSequenceTransformer(nn.Module):
    """Bidirectional Transformer Encoder to model 3D anatomical continuity across slices."""
    def __init__(self, d_model: int = 768, nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = SinusoidalPositionEncoding1D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.norm(x)


class TriPlanarCoAttention(nn.Module):
    """Multi-View Cross-Attention module enabling Coronal, Sagittal, and Axial planes to cross-attend."""
    def __init__(self, d_model: int = 768, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.sag_to_others = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cor_to_others = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.axi_to_others = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_sag = nn.LayerNorm(d_model)
        self.norm_cor = nn.LayerNorm(d_model)
        self.norm_axi = nn.LayerNorm(d_model)

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, axi: torch.Tensor):
        cor_others = torch.cat([sag, axi], dim=1)
        cor_out, _ = self.cor_to_others(cor, cor_others, cor_others)
        cor_refined = self.norm_cor(cor + cor_out)

        sag_others = torch.cat([cor, axi], dim=1)
        sag_out, _ = self.sag_to_others(sag, sag_others, sag_others)
        sag_refined = self.norm_sag(sag + sag_out)

        axi_others = torch.cat([sag, cor], dim=1)
        axi_out, _ = self.axi_to_others(axi, axi_others, axi_others)
        axi_refined = self.norm_axi(axi + axi_out)

        return torch.cat([sag_refined, cor_refined, axi_refined], dim=1)


class AblationTriPlanarMILModel(nn.Module):
    """
    Modular Tri-Planar MIL Model configurable for independent ablations.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.3,
        chunk_size: int = 32,
        slices_per_plane: int = 24,
        use_intra_transformer: bool = False,
        use_coattention: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.slices_per_plane = slices_per_plane
        self.use_intra_transformer = use_intra_transformer
        self.use_coattention = use_coattention

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

        # Learned anatomical plane embeddings: 0=Sagittal, 1=Coronal, 2=Axial
        self.plane_embedding = nn.Embedding(3, self.num_features)
        nn.init.normal_(self.plane_embedding.weight, std=0.02)

        # Arm 3: 3D Intra-Plane Sequence Transformer
        if self.use_intra_transformer:
            nhead = 8 if self.num_features % 8 == 0 else 6
            self.intra_sag = IntraPlaneSequenceTransformer(d_model=self.num_features, nhead=nhead, num_layers=2)
            self.intra_cor = IntraPlaneSequenceTransformer(d_model=self.num_features, nhead=nhead, num_layers=2)
            self.intra_axi = IntraPlaneSequenceTransformer(d_model=self.num_features, nhead=nhead, num_layers=2)

        # Arm 4: Tri-Planar Co-Attention
        if self.use_coattention:
            nhead = 8 if self.num_features % 8 == 0 else 6
            self.co_attention = TriPlanarCoAttention(d_model=self.num_features, nhead=nhead)

        # 12-branch label-specific gated attention pooling
        self.mil_pool = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

    def extract_slice_features(self, images: torch.Tensor) -> torch.Tensor:
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        total = B * D

        if total <= self.chunk_size:
            feats_flat = self.backbone(x_flat)
        else:
            feats_list = []
            for i in range(0, total, self.chunk_size):
                chunk = x_flat[i : i + self.chunk_size]
                feats_list.append(self.backbone(chunk))
            feats_flat = torch.cat(feats_list, dim=0)

        return feats_flat.view(B, D, self.num_features)

    def forward(
        self,
        images: torch.Tensor,
        plane_ids: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        B, D, C, H, W = images.shape
        S = self.slices_per_plane

        # 1. Extract slice features
        slice_feats = self.extract_slice_features(images)

        # 2. Add learned plane embeddings
        plane_embeds = self.plane_embedding(plane_ids)
        slice_feats = slice_feats + plane_embeds

        # 3. Optional Ablation: Intra-Plane Sequence Transformer (Arm 3)
        if self.use_intra_transformer:
            sag = slice_feats[:, 0:S, :]
            cor = slice_feats[:, S:2*S, :]
            axi = slice_feats[:, 2*S:3*S, :]

            sag = self.intra_sag(sag)
            cor = self.intra_cor(cor)
            axi = self.intra_axi(axi)
            slice_feats = torch.cat([sag, cor, axi], dim=1)

        # 4. Optional Ablation: Tri-Planar Co-Attention (Arm 4)
        if self.use_coattention:
            sag = slice_feats[:, 0:S, :]
            cor = slice_feats[:, S:2*S, :]
            axi = slice_feats[:, 2*S:3*S, :]
            slice_feats = self.co_attention(sag, cor, axi)

        # 5. Label-Specific Gated Attention Pooling
        logits, attn_weights, study_embed = self.mil_pool(slice_feats, mask=mask)

        return {
            "logits": logits,
            "attention_weights": attn_weights,
            "study_embedding": study_embed,
        }


# ==============================================================================
# Arm 1: Asymmetric Focal Consensus Loss (AFCL)
# ==============================================================================

PATHOLOGY_WEIGHTS = [1.0, 1.8, 1.1, 1.2, 1.1, 1.2, 1.8, 1.0, 1.0, 1.0, 1.0, 1.5]

class AsymmetricFocalConsensusLoss(nn.Module):
    """
    AFCL:
    - Never zeros out hard ground truth labels!
    - When hard target is available, it is strictly trusted as ground truth.
    - Soft targets use continuous confidence weighting tanh(3 * |p - 0.5|).
    - Asymmetric focal modulation on hard false negatives.
    - Class-adaptive multipliers: MCL (1.8x), PF OA (1.8x), Fracture (1.5x).
    """
    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.register_buffer("class_weights", torch.tensor(PATHOLOGY_WEIGHTS, dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
        is_gold: torch.Tensor = None,
    ) -> torch.Tensor:
        has_hard = ~torch.isnan(hard_targets)

        # Probabilities
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)

        # Target reconciliation:
        # If hard target exists, use hard target (0 or 1).
        # Otherwise, use soft target.
        targets = torch.where(has_hard, hard_targets, soft_targets)

        # Sample confidence weighting:
        # Hard targets get weight 1.0 (or gold_weight if is_gold)
        # Soft-only targets get weight tanh(3.0 * |p - 0.5|)
        soft_conf = torch.tanh(3.0 * torch.abs(soft_targets - 0.5))
        weights = torch.where(has_hard, torch.ones_like(logits), soft_conf)

        if is_gold is not None:
            gold_mask = is_gold.view(-1, 1).expand_as(logits).bool()
            valid_gold = gold_mask & has_hard
            weights = torch.where(valid_gold, torch.full_like(weights, 2.0), weights)

        # Asymmetric focal loss
        pos_probs = probs
        neg_probs = (probs - self.clip).clamp(min=0.0)

        pos_loss = targets * torch.pow(1.0 - pos_probs, self.gamma_pos) * torch.log(pos_probs)
        neg_loss = (1.0 - targets) * torch.pow(neg_probs, self.gamma_neg) * torch.log(1.0 - pos_probs)
        focal_loss = -(pos_loss + neg_loss)

        # Apply sample weights and class multipliers
        weighted_loss = focal_loss * weights * self.class_weights.to(logits.device).unsqueeze(0)

        return weighted_loss.sum() / weights.sum().clamp(min=self.eps)


# ==============================================================================
# Evaluation Function
# ==============================================================================

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_soft, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            plane_ids = batch["plane_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, plane_ids=plane_ids, mask=mask)
                probs = torch.sigmoid(out["logits"]).cpu().numpy()

            all_preds.append(probs)
            all_soft.append(batch["soft_targets"].numpy())
            all_is_gold.append(batch["is_gold"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_soft, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    full_metrics = compute_macro_auc(targets, preds)
    full_macro = full_metrics["macro_auc"]
    full_per_label = full_metrics["per_class_auc"]

    gold_macro = 0.0
    if is_gold.sum() > 5:
        gold_metrics = compute_macro_auc(targets[is_gold], preds[is_gold])
        gold_macro = gold_metrics["macro_auc"]

    return full_macro, full_per_label, gold_macro, preds


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 14 Independent Ablation Experiments")
    parser.add_argument("--ablation_mode", type=str, required=True,
                        choices=["afcl", "highres", "intra_transformer", "coattention"],
                        help="Which independent ablation arm to run")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12, help="12 epochs reaches peak validation")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    # Configure parameters per ablation mode
    if args.ablation_mode == "highres":
        image_size = 384
        slices_per_plane = 32
        batch_size = 2
        grad_accum_steps = 8
        use_intra_transformer = False
        use_coattention = False
        loss_type = "consensus_denoised"
    elif args.ablation_mode == "afcl":
        image_size = 288
        slices_per_plane = 24
        batch_size = 2
        grad_accum_steps = 8
        use_intra_transformer = False
        use_coattention = False
        loss_type = "afcl"
    elif args.ablation_mode == "intra_transformer":
        image_size = 288
        slices_per_plane = 24
        batch_size = 2
        grad_accum_steps = 8
        use_intra_transformer = True
        use_coattention = False
        loss_type = "consensus_denoised"
    elif args.ablation_mode == "coattention":
        image_size = 288
        slices_per_plane = 24
        batch_size = 2
        grad_accum_steps = 8
        use_intra_transformer = False
        use_coattention = True
        loss_type = "consensus_denoised"

    total_slices = slices_per_plane * 3
    if not args.run_name:
        args.run_name = f"phase14_ablation_{args.ablation_mode}_convnext_tiny_f{args.fold}"

    console = Console()
    console.print(f"[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]")
    console.print(f"[bold green]║  Phase 14 Ablation: {args.ablation_mode.upper()} (ConvNeXt-Tiny Fold {args.fold})          ║[/bold green]")
    console.print(f"[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]")
    console.print(f"Mode: {args.ablation_mode} | Image Size: {image_size}x{image_size} | Slices: {total_slices} ({slices_per_plane}/plane)")
    console.print(f"Device: {args.device} | Batch Size: {batch_size} (Eff: {batch_size * grad_accum_steps}) | Epochs: {args.epochs}")
    console.print(f"Loss: {loss_type} | Intra-Transformer: {use_intra_transformer} | Co-Attention: {use_coattention}")

    # Load data
    df_splits = pd.read_parquet("data/splits_5fold.parquet")
    df_tri = pd.read_parquet("data/tristate_labels_master.parquet")
    df_dense = pd.read_parquet("data/dense_labels_master.parquet")

    df_all = df_splits[["StudyInstanceUID", "fold"]].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all["is_gold"] = labeled_mask

    tri_renamed = df_tri[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"hard_{c}" for c in TARGET_COLUMNS})
    dense_renamed = df_dense[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on="StudyInstanceUID", how="left")
    df_all = df_all.merge(dense_renamed, on="StudyInstanceUID", how="left")

    train_df = df_all[df_all["fold"] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == args.fold].reset_index(drop=True)

    console.print(f"Train studies: {len(train_df)} | Val studies: {len(val_df)} (Gold in Val: {val_df['is_gold'].sum()})")

    train_transforms = get_training_transforms((image_size, image_size))
    val_transforms = get_validation_transforms((image_size, image_size))

    train_ds = TriPlanarKneeDataset(
        train_df,
        slices_per_plane=slices_per_plane,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = TriPlanarKneeDataset(
        val_df,
        slices_per_plane=slices_per_plane,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=triplanar_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=triplanar_collate,
    )

    # Initialize model
    model = AblationTriPlanarMILModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=128,
        dropout=0.3,
        chunk_size=32,
        slices_per_plane=slices_per_plane,
        use_intra_transformer=use_intra_transformer,
        use_coattention=use_coattention,
    )
    model.to(args.device)

    # Parameter groups for optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-2)

    total_steps = args.epochs * (len(train_loader) // grad_accum_steps)
    warmup_steps = args.warmup_epochs * (len(train_loader) // grad_accum_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    if loss_type == "afcl":
        criterion = AsymmetricFocalConsensusLoss().to(args.device)
    else:
        criterion = ConsensusDenoisedBCEWithLogitsLoss(gold_weight=2.0)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{args.run_name}_best.pth")

    best_macro_auc = 0.0
    best_gold_auc = 0.0

    console.print(f"Starting Training: {args.epochs} epochs planned...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(args.device)
            plane_ids = batch["plane_ids"].to(args.device)
            mask = batch["mask"].to(args.device)
            hard = batch["hard_targets"].to(args.device)
            soft = batch["soft_targets"].to(args.device)
            is_gold = batch["is_gold"].to(args.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, plane_ids=plane_ids, mask=mask)
                loss = criterion(out["logits"], hard, soft, is_gold=is_gold)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        train_loss /= len(train_loader)
        epoch_sec = time.time() - t0

        # Evaluation
        val_macro, per_class, val_gold, _ = evaluate(model, val_loader, args.device)

        is_best = val_macro > best_macro_auc
        if is_best:
            best_macro_auc = val_macro
            best_gold_auc = val_gold
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_auc": val_macro,
                "gold_auc": val_gold,
                "per_class_auc": per_class,
                "args": vars(args),
            }, ckpt_path)

        star = " ⭐ BEST" if is_best else ""
        console.print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_sec:.0f}s] | "
            f"Loss: {train_loss:.4f} | Val Macro AUC: {val_macro:.4f} | Gold AUC: {val_gold:.4f}{star}"
        )

        # Print breakdown of key anatomy
        med_oa = per_class.get("Medial OA", 0.0)
        lat_oa = per_class.get("Lateral OA", 0.0)
        pf_oa = per_class.get("PF OA", 0.0)
        mcl = per_class.get("MCL", 0.0)
        fracture = per_class.get("Fracture", 0.0)
        console.print(
            f"   -> Key Pathologies: MedOA: {med_oa:.3f} | LatOA: {lat_oa:.3f} | PFOA: {pf_oa:.3f} | MCL: {mcl:.3f} | Fracture: {fracture:.3f}"
        )

    console.print(f"\n[bold green]✓ Ablation '{args.ablation_mode}' Complete![/bold green]")
    console.print(f"Peak Val Macro AUC: [bold yellow]{best_macro_auc:.4f}[/bold yellow] | Gold AUC: [bold yellow]{best_gold_auc:.4f}[/bold yellow]")
    console.print(f"Baseline Comparison (Phase 13 Baseline: 0.8684 Val | 0.9117 Gold)")
    delta_val = best_macro_auc - 0.8684
    delta_gold = best_gold_auc - 0.9117
    console.print(f"Delta Val: {'+' if delta_val >= 0 else ''}{delta_val:.4f} | Delta Gold: {'+' if delta_gold >= 0 else ''}{delta_gold:.4f}")
    console.print(f"Best checkpoint saved to: {ckpt_path}\n")


if __name__ == "__main__":
    main()
