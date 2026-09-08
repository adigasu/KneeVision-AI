"""
RSNA Knee Abnormality Detection - Loss Functions for Multi-Label Class Imbalance.
Includes Asymmetric Focal Loss (ASL) and Masked Binary Cross Entropy (BCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification with severe negative/positive imbalance.
    Reference: 'Asymmetric Loss For Multi-Label Classification' (Ben-Baruch et al., 2021).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (Batch, Num_Classes).
            targets: Binary labels of shape (Batch, Num_Classes), may contain NaNs.
        """
        # Create mask for valid (non-NaN) targets
        valid_mask = ~torch.isnan(targets)
        if not valid_mask.any():
            return (logits * 0.0).sum()

        # Replace NaNs with 0 temporarily for calculation, will mask out later
        clean_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping for Negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy
        los_pos = clean_targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - clean_targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * clean_targets
            pt1 = xs_neg * (1.0 - clean_targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * clean_targets + self.gamma_neg * (1.0 - clean_targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = loss * one_sided_w

        # Mask out invalid (NaN) labels
        loss = -loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Standard Multi-Label BCE loss that automatically ignores NaN labels.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = ~torch.isnan(targets)
        if not valid_mask.any():
            return (logits * 0.0).sum()

        clean_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        bce = F.binary_cross_entropy_with_logits(
            logits, clean_targets, pos_weight=self.pos_weight, reduction="none"
        )
        masked_bce = bce * valid_mask.float()
        return masked_bce.sum() / valid_mask.float().sum().clamp(min=1.0)


class ConfidenceWeightedBCEWithLogitsLoss(nn.Module):
    """
    Confidence-Weighted Soft Binary Cross-Entropy Loss for LLM-derived soft targets.
    Loss weight is 2 * |p - 0.5|:
    - When p in {0.0, 1.0} -> weight = 1.0 (Full confidence)
    - When p = 0.50 -> weight = 0.0 (Zeroes out unaddressed findings)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Confidence weight: 2 * |p - 0.5|
        weights = 2.0 * torch.abs(targets - 0.5)

        # Soft BCE with logits
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Weighted loss normalized strictly over confident evidence
        weighted_loss = weights * bce
        return weighted_loss.sum() / (weights.sum() + self.eps)
