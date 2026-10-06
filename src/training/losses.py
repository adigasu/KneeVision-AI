from typing import Optional
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

class MixedTargetBCEWithLogitsLoss(nn.Module):
    """
    Mixed Target Binary Cross-Entropy Loss:
        y_train = alpha * y_hard + (1 - alpha) * y_soft

    Gracefully handles missing/NaN values in hard_targets and applies
    confidence weighting (2 * |p - 0.5|) to unanchored soft targets.
    """

    def __init__(self, alpha: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        has_hard = ~torch.isnan(hard_targets)
        clean_hard = torch.where(has_hard, hard_targets, torch.zeros_like(hard_targets))

        # Mixed target
        mixed_target = torch.where(
            has_hard,
            self.alpha * clean_hard + (1.0 - self.alpha) * soft_targets,
            soft_targets,
        )

        soft_weights = 2.0 * torch.abs(soft_targets - 0.5)
        sample_weights = torch.where(has_hard, torch.ones_like(hard_targets), soft_weights)

        bce = F.binary_cross_entropy_with_logits(logits, mixed_target, reduction='none')
        weighted_loss = sample_weights * bce

        return weighted_loss.sum() / sample_weights.sum().clamp(min=self.eps)


class ConsensusDenoisedBCEWithLogitsLoss(nn.Module):
    """
    4-Tier Noise-Robust Binary Cross-Entropy Loss:
    - Tier 1 (is_gold == True): Gold standard human consensus annotations (N=58).
      Highest priority (w=2.0) and strict binary ground truth.
    - Tier 2 (Active Contradictions): Heuristic tristate and LLM directly disagree
      (Tri=1 & Soft<0.30 or Tri=0 & Soft>0.70).
      Masked out with zero weight (w=0.0) to eliminate text extraction errors.
    - Tier 3 (Consensus Agreements): Heuristic tristate and LLM agree
      (Tri=1 & Soft>=0.70 or Tri=0 & Soft<=0.30).
      Standard weight (w=1.0) and gentle label smoothing (0.05 / 0.95).
    - Tier 4 (Sparse Unannotated Hard Targets): Tristate is NaN, only LLM soft target exists.
      Retained with confidence weighting w = 2 * |p - 0.5| to prevent gradient starvation!
    """

    def __init__(
        self,
        gold_weight: float = 2.0,
        pos_thresh: float = 0.70,
        neg_thresh: float = 0.30,
        eps: float = 1e-6,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gold_weight = gold_weight
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.eps = eps
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
        is_gold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        has_hard = ~torch.isnan(hard_targets)

        # 1. Contradictions (where both are present and actively disagree)
        conflict_pos = (hard_targets == 1.0) & (soft_targets < self.neg_thresh)
        conflict_neg = (hard_targets == 0.0) & (soft_targets > self.pos_thresh)
        is_conflict = has_hard & (conflict_pos | conflict_neg)

        # 2. Consensus agreements (where both are present and agree)
        agree_pos = (hard_targets == 1.0) & (soft_targets >= self.pos_thresh)
        agree_neg = (hard_targets == 0.0) & (soft_targets <= self.neg_thresh)
        is_agree = has_hard & (agree_pos | agree_neg)

        # 3. Base weights and targets:
        # Default for unanchored soft targets (hard is NaN) -> confidence weighting 2 * |p - 0.5|
        soft_weights = 2.0 * torch.abs(soft_targets - 0.5)
        weights = torch.where(has_hard, torch.zeros_like(logits), soft_weights)
        clean_targets = soft_targets.clone()

        # Apply agreement (w=1.0, smoothed 0.95 / 0.05)
        weights = torch.where(is_agree, torch.ones_like(weights), weights)
        clean_targets = torch.where(
            agree_pos,
            torch.tensor(0.95, device=logits.device, dtype=logits.dtype),
            clean_targets,
        )
        clean_targets = torch.where(
            agree_neg,
            torch.tensor(0.05, device=logits.device, dtype=logits.dtype),
            clean_targets,
        )

        # Mask out conflicts (w=0.0)
        weights = torch.where(is_conflict, torch.zeros_like(weights), weights)

        # 4. Gold overrides everything
        if is_gold is not None:
            gold_mask = is_gold.view(-1, 1).expand_as(logits).bool()
            valid_gold = gold_mask & has_hard
            weights = torch.where(valid_gold, torch.full_like(weights, self.gold_weight), weights)
            clean_targets = torch.where(valid_gold, hard_targets, clean_targets)

        # 5. Weighted binary cross-entropy with logits
        bce = F.binary_cross_entropy_with_logits(logits, clean_targets, reduction='none')
        weighted_loss = weights * bce

        # 6. Optional pathology bottleneck class weighting
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device).view(1, -1)
            weighted_loss = weighted_loss * cw
            denom = (weights * cw).sum().clamp(min=self.eps)
        else:
            denom = weights.sum().clamp(min=self.eps)

        return weighted_loss.sum() / denom



class CurriculumDenoisedBCEWithLogitsLoss(nn.Module):
    """
    4-Tier Noise-Robust BCE with dynamic late-epoch small-loss sample trimming:
    - Tier 1 (is_gold == True): Gold standard human consensus (N=58) always kept (w=2.0, never trimmed).
    - Tier 2 (Active Contradictions): Heuristic tristate and LLM directly disagree -> Masked out (w=0.0).
    - Tier 3 (Consensus Agreements): Heuristic tristate and LLM agree -> w=1.0 and smoothed (0.95/0.05).
    - Tier 4 (Sparse Unannotated Hard Targets): Confidence weighting 2 * |p - 0.5|.
    - Dynamic Curriculum (epoch >= trim_start_epoch):
      Per-study loss is computed across all 12 pathologies. For non-gold silver studies, the top
      `trim_ratio` (e.g. 10%) highest-loss studies in the batch are dynamically zeroed out,
      preventing late-stage memorization of corrupt/ambiguous text report annotations.
    """

    def __init__(
        self,
        gold_weight: float = 2.0,
        pos_thresh: float = 0.70,
        neg_thresh: float = 0.30,
        trim_ratio: float = 0.10,
        trim_start_epoch: int = 5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gold_weight = gold_weight
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.trim_ratio = trim_ratio
        self.trim_start_epoch = trim_start_epoch
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
        is_gold: Optional[torch.Tensor] = None,
        epoch: int = 1,
    ) -> torch.Tensor:
        has_hard = ~torch.isnan(hard_targets)

        # 1. Contradictions (where both are present and actively disagree)
        conflict_pos = (hard_targets == 1.0) & (soft_targets < self.neg_thresh)
        conflict_neg = (hard_targets == 0.0) & (soft_targets > self.pos_thresh)
        is_conflict = has_hard & (conflict_pos | conflict_neg)

        # 2. Consensus agreements (where both are present and agree)
        agree_pos = (hard_targets == 1.0) & (soft_targets >= self.pos_thresh)
        agree_neg = (hard_targets == 0.0) & (soft_targets <= self.neg_thresh)
        is_agree = has_hard & (agree_pos | agree_neg)

        # 3. Base weights and targets
        soft_weights = 2.0 * torch.abs(soft_targets - 0.5)
        weights = torch.where(has_hard, torch.zeros_like(logits), soft_weights)
        clean_targets = soft_targets.clone()

        # Apply agreement (w=1.0, smoothed 0.95 / 0.05)
        weights = torch.where(is_agree, torch.ones_like(weights), weights)
        clean_targets = torch.where(
            agree_pos,
            torch.tensor(0.95, device=logits.device, dtype=logits.dtype),
            clean_targets,
        )
        clean_targets = torch.where(
            agree_neg,
            torch.tensor(0.05, device=logits.device, dtype=logits.dtype),
            clean_targets,
        )

        # Mask out conflicts (w=0.0)
        weights = torch.where(is_conflict, torch.zeros_like(weights), weights)

        # 4. Gold overrides everything
        is_gold_study = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
        if is_gold is not None:
            is_gold_study = is_gold.view(-1).bool()
            gold_mask = is_gold_study.view(-1, 1).expand_as(logits)
            valid_gold = gold_mask & has_hard
            weights = torch.where(valid_gold, torch.full_like(weights, self.gold_weight), weights)
            clean_targets = torch.where(valid_gold, hard_targets, clean_targets)

        # 5. Weighted binary cross-entropy with logits
        bce = F.binary_cross_entropy_with_logits(logits, clean_targets, reduction="none")
        weighted_bce = weights * bce
        study_loss = weighted_bce.sum(dim=-1)
        study_weights = weights.sum(dim=-1)

        # 6. Dynamic Small-Loss Sample Selection for late epochs
        if epoch >= self.trim_start_epoch and self.trim_ratio > 0.0 and logits.size(0) > 1:
            silver_mask = ~is_gold_study
            num_silver = silver_mask.sum().item()
            if num_silver > 1:
                k = max(1, int(num_silver * (1.0 - self.trim_ratio)))
                silver_losses = study_loss[silver_mask].detach()
                cutoff = torch.kthvalue(silver_losses, k).values
                keep_study = is_gold_study | (study_loss <= cutoff)
                study_loss = study_loss * keep_study.float()
                study_weights = study_weights * keep_study.float()

        return study_loss.sum() / study_weights.sum().clamp(min=self.eps)
