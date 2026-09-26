"""
KneeVision-AI: Loss Function & Hyperparameter Fast Grid Search
==============================================================
Runs fast 4-epoch comparative trials on Fold 0 to evaluate:
  1. Loss functions:
     - 'soft_bce': Linear confidence weighted BCE: w = 2*|p-0.5|
     - 'squared_bce': Quadratic confidence weighted BCE: w = (2*|p-0.5|)^2 (amplifies pure labels)
     - 'asl': Asymmetric Focal Loss (gamma_neg=2.0, gamma_pos=0.0)
     - 'sharpened_bce': Temperature sharpened soft BCE (T=0.7)
  2. Learning rates:
     - differential LRs (backbone: 5e-5, 1e-4, 2e-4 | head: 2e-4, 5e-4, 1e-3)
"""

import os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

class AsymmetricLossCustom(nn.Module):
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = 2.0 * torch.abs(targets - 0.5)
        probs = torch.sigmoid(logits)
        p_pos = probs
        p_neg = (1.0 - probs + self.clip).clamp(max=1.0)
        
        los_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt = p_pos * targets + p_neg * (1.0 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal_weight = torch.pow(1.0 - pt, one_sided_gamma)
        loss = -loss * focal_weight * weights
        return loss.sum() / (weights.sum() + 1e-6)


class SquaredConfidenceBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.pow(2.0 * torch.abs(targets - 0.5), 2.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (weights * bce).sum() / (weights.sum() + self.eps)


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Loss & Hyperparameters")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--image_size", type=int, default=288)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    experiments = [
        {"name": "Exp1: Soft BCE (Baseline)", "loss": "soft_bce", "backbone_lr": 8e-5, "head_lr": 4e-4},
        {"name": "Exp2: Squared Conf BCE (Gold Focus)", "loss": "squared_bce", "backbone_lr": 8e-5, "head_lr": 4e-4},
        {"name": "Exp3: Asymmetric Focal Loss", "loss": "asl", "backbone_lr": 8e-5, "head_lr": 4e-4},
        {"name": "Exp4: Higher Backbone LR", "loss": "squared_bce", "backbone_lr": 1.5e-4, "head_lr": 5e-4},
    ]

    table = Table(title=f"KneeVision-AI: Loss & HParam Trials (Fold {args.fold})")
    table.add_column("Trial", style="cyan")
    table.add_column("Loss Function", style="green")
    table.add_column("Backbone LR", style="magenta")
    table.add_column("Head LR", style="magenta")
    table.add_column("Command", style="yellow")

    for exp in experiments:
        cmd = (
            f"python scripts/11_train_dense_288px.py --fold {args.fold} "
            f"--epochs {args.epochs} --backbone {args.backbone} "
            f"--backbone_lr {exp['backbone_lr']} --head_lr {exp['head_lr']} "
            f"--image_size {args.image_size} --batch_size {args.batch_size}"
        )
        table.add_row(exp["name"], exp["loss"], f"{exp['backbone_lr']:.1e}", f"{exp['head_lr']:.1e}", cmd)

    console.print(table)

if __name__ == "__main__":
    main()
