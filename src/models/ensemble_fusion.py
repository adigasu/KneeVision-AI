"""
RSNA Knee Abnormality Detection - Ensemble & Multi-Backbone Fusion Architectures
Implements:
1. CalibratedRankAveraging / Constrained Stacking
2. BottleneckFeatureFusion (Middle Pre-Classifier Fusion with Residual Bypass)
3. ResidualGatedFusion (Context-Conditioned Instance Dynamic Gating)
4. DualAxisLabelSpecificABMIL (Hierarchical Slice-Depth & Model Attention)
5. PathologyGroupedSoftMoE (Clinical Category-Specialized Mixture of Experts)
6. PathologySlotAttention (Competitive Object-Centric Slot Normalization)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedRankAveraging(nn.Module):
    def __init__(self, num_models: int, num_classes: int = 12, initial_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        if initial_weights is None:
            initial_weights = torch.ones(num_classes, num_models) / num_models
        self.weights = nn.Parameter(initial_weights)

    def forward(self, probs_list: List[torch.Tensor], use_ranks: bool = False) -> torch.Tensor:
        stacked = torch.stack(probs_list, dim=-1)  # (Batch, 12, M)
        normalized_w = F.softmax(self.weights, dim=-1)  # (12, M)
        normalized_w = normalized_w.unsqueeze(0)  # (1, 12, M)

        if use_ranks:
            ranks = torch.argsort(torch.argsort(stacked, dim=0), dim=0).float()
            ranked_scores = ranks / (stacked.shape[0] - 1 + 1e-7)
            return (ranked_scores * normalized_w).sum(dim=-1)

        return (stacked * normalized_w).sum(dim=-1)


class BottleneckFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims: List[int] = [768, 768, 384],
        num_classes: int = 12,
        bottleneck_dim: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.total_in_dim = sum(in_dims)

        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_in_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
            for _ in range(num_classes)
        ])

        self.skip_weights = nn.Parameter(torch.ones(num_classes, len(in_dims)) / len(in_dims))

    def forward(
        self,
        model_feats_list: List[torch.Tensor],
        model_standalone_logits: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        concat_feats = torch.cat(model_feats_list, dim=-1)
        logits = []
        for k in range(self.num_classes):
            delta = self.bottlenecks[k](concat_feats[:, k, :]).squeeze(-1)
            logits.append(delta)

        fused_delta = torch.stack(logits, dim=1)

        if model_standalone_logits is not None:
            stacked_standalone = torch.stack(model_standalone_logits, dim=-1)
            norm_skips = F.softmax(self.skip_weights, dim=-1).unsqueeze(0)
            base_logits = (stacked_standalone * norm_skips).sum(dim=-1)
            return base_logits + fused_delta

        return fused_delta


class ResidualGatedFusion(nn.Module):
    def __init__(
        self,
        in_dims: List[int] = [768, 768, 384],
        num_classes: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_models = len(in_dims)
        self.total_dim = sum(in_dims)

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for dim in in_dims
        ])

        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.num_models),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_classes)
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])

    def forward(
        self, model_feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = model_feats_list[0].shape[0]
        concat_feats = torch.cat(model_feats_list, dim=-1)

        projected = [proj(feats) for proj, feats in zip(self.projections, model_feats_list)]
        stacked_proj = torch.stack(projected, dim=2)

        logits = []
        gates = []
        for k in range(self.num_classes):
            gate_w = self.gate_nets[k](concat_feats[:, k, :])
            gates.append(gate_w)

            fused_rep = (stacked_proj[:, k, :, :] * gate_w.unsqueeze(-1)).sum(dim=1)
            logit_k = self.classifiers[k](fused_rep).squeeze(-1)
            logits.append(logit_k)

        return torch.stack(logits, dim=1), torch.stack(gates, dim=1)


class DualAxisLabelSpecificABMIL(nn.Module):
    def __init__(
        self,
        in_features_dict: Dict[str, int] = {'convnext_tiny': 768, 'convnext_small': 768, 'dinov2': 384},
        num_classes: int = 12,
        mil_hidden: int = 128,
        cross_hidden: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.models = list(in_features_dict.keys())
        self.num_classes = num_classes

        self.slice_attn_V = nn.ModuleDict()
        self.slice_attn_U = nn.ModuleDict()
        self.slice_attn_w = nn.ModuleDict()
        self.model_proj = nn.ModuleDict()

        for name, dim in in_features_dict.items():
            self.slice_attn_V[name] = nn.Sequential(nn.Linear(dim, mil_hidden), nn.Tanh(), nn.Dropout(dropout))
            self.slice_attn_U[name] = nn.Sequential(nn.Linear(dim, mil_hidden), nn.Sigmoid(), nn.Dropout(dropout))
            self.slice_attn_w[name] = nn.Linear(mil_hidden, num_classes)
            self.model_proj[name] = nn.Sequential(
                nn.Linear(dim, cross_hidden),
                nn.LayerNorm(cross_hidden),
                nn.GELU(),
            )

        self.cross_query = nn.Parameter(torch.randn(num_classes, cross_hidden) * 0.02)
        self.classifier = nn.Linear(cross_hidden, 1)

    def forward(
        self,
        slice_feats_dict: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        model_study_reps = []
        slice_attentions = {}

        for name in self.models:
            feats = slice_feats_dict[name]
            V = self.slice_attn_V[name](feats)
            U = self.slice_attn_U[name](feats)
            A = self.slice_attn_w[name](V * U)

            if mask is not None:
                A = A.masked_fill(~mask.unsqueeze(-1), -1e4)

            A = torch.softmax(A, dim=1)
            slice_attentions[name] = A

            h_k = torch.einsum('bdk,bdf->bkf', A, feats)
            h_proj = self.model_proj[name](h_k)
            model_study_reps.append(h_proj)

        stacked = torch.stack(model_study_reps, dim=2)
        scores = torch.einsum('kc,bkmc->bkm', self.cross_query, stacked) / (self.cross_query.shape[-1] ** 0.5)
        beta = torch.softmax(scores, dim=-1)

        fused = torch.einsum('bkm,bkmc->bkc', beta, stacked)
        logits = self.classifier(fused).squeeze(-1)

        return logits, beta, slice_attentions


class PathologyGroupedSoftMoE(nn.Module):
    GROUPS = {
        'focal_tears': [0, 1, 2, 3],
        'osteoarthritis': [4, 5, 6],
        'fluid_bone': [7, 8, 9, 10, 11],
    }

    def __init__(
        self,
        in_dims: List[int] = [768, 768, 384],
        num_classes: int = 12,
        expert_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = len(in_dims)

        self.expert_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for dim in in_dims
        ])

        self.routers = nn.ModuleDict({
            grp_name: nn.Sequential(
                nn.Linear(expert_dim * self.num_experts, expert_dim // 2),
                nn.ReLU(),
                nn.Linear(expert_dim // 2, self.num_experts),
                nn.Softmax(dim=-1),
            )
            for grp_name in self.GROUPS
        })

        self.heads = nn.ModuleDict()
        for grp_name, indices in self.GROUPS.items():
            self.heads[grp_name] = nn.ModuleList([
                nn.Linear(expert_dim, len(indices)) for _ in range(self.num_experts)
            ])

    def forward(
        self, model_feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = model_feats_list[0].shape[0]

        study_feats = []
        for feat in model_feats_list:
            if feat.dim() == 3:
                study_feats.append(feat.mean(dim=1))
            else:
                study_feats.append(feat)

        proj_experts = [proj(f) for proj, f in zip(self.expert_projections, study_feats)]
        concat_expert_feats = torch.cat(proj_experts, dim=-1)

        logits_by_idx = [None] * self.num_classes
        routing_weights = {}

        for grp_name, indices in self.GROUPS.items():
            gate_w = self.routers[grp_name](concat_expert_feats)
            routing_weights[grp_name] = gate_w

            grp_logits = torch.zeros(B, len(indices), device=concat_expert_feats.device)
            for m in range(self.num_experts):
                expert_pred = self.heads[grp_name][m](proj_experts[m])
                grp_logits = grp_logits + expert_pred * gate_w[:, m : m + 1]

            for local_idx, global_idx in enumerate(indices):
                logits_by_idx[global_idx] = grp_logits[:, local_idx]

        return torch.stack(logits_by_idx, dim=1), routing_weights


class PathologySlotAttention(nn.Module):
    def __init__(
        self,
        in_features_list: List[int] = [768, 768, 384],
        slot_dim: int = 128,
        num_slots: int = 12,
        iters: int = 3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.input_projections = nn.ModuleList([
            nn.Linear(dim, slot_dim) for dim in in_features_list
        ])

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slots_sigma = nn.Parameter(torch.ones(1, num_slots, slot_dim) * 0.02)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

        self.classifier = nn.Linear(slot_dim, 1)

    def forward(
        self, tokens_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = tokens_list[0].shape[0]

        proj_tokens = [proj(tok) for proj, tok in zip(self.input_projections, tokens_list)]
        inputs = self.norm_inputs(torch.cat(proj_tokens, dim=1))

        slots = self.slots_mu + self.slots_sigma * torch.randn_like(self.slots_sigma).expand(B, -1, -1)

        k = self.to_k(inputs)
        v = self.to_v(inputs)

        attn = None
        for _ in range(self.iters):
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = torch.softmax(dots, dim=1) + self.eps
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bij,bjd->bid', attn_norm, v)

            slots = self.gru(
                updates.reshape(-1, updates.shape[-1]),
                slots.reshape(-1, slots.shape[-1]),
            ).reshape(B, self.num_slots, -1)

            slots = slots + self.mlp(slots)

        logits = self.classifier(slots).squeeze(-1)
        return logits, attn


class SparseTopKPathologyMoE(nn.Module):
    """
    Sparse Top-k Mixture of Experts.
    Hard-routes each clinical group to only the top-k highest scoring backbones,
    masking non-top-k logits to -inf before softmax re-normalization.
    """
    GROUPS = {
        'focal_tears': [0, 1, 2, 3],
        'osteoarthritis': [4, 5, 6],
        'fluid_bone': [7, 8, 9, 10, 11],
    }

    def __init__(
        self,
        in_dims: List[int] = [768, 768, 384],
        num_classes: int = 12,
        expert_dim: int = 128,
        k: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.k = min(k, len(in_dims))
        self.num_classes = num_classes
        self.num_experts = len(in_dims)

        self.expert_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for dim in in_dims
        ])

        self.routers = nn.ModuleDict({
            grp: nn.Linear(expert_dim * self.num_experts, self.num_experts)
            for grp in self.GROUPS
        })

        self.heads = nn.ModuleDict({
            grp: nn.ModuleList([nn.Linear(expert_dim, len(indices)) for _ in range(self.num_experts)])
            for grp, indices in self.GROUPS.items()
        })

    def forward(
        self, model_feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = model_feats_list[0].shape[0]
        study_feats = [f.mean(dim=1) if f.dim() == 3 else f for f in model_feats_list]
        proj_experts = [p(f) for p, f in zip(self.expert_projections, study_feats)]
        concat_expert_feats = torch.cat(proj_experts, dim=-1)

        logits_by_idx = [None] * self.num_classes
        routing_weights = {}

        for grp_name, indices in self.GROUPS.items():
            raw_scores = self.routers[grp_name](concat_expert_feats)  # (B, M)
            topk_vals, topk_idx = torch.topk(raw_scores, k=self.k, dim=-1)
            masked = torch.full_like(raw_scores, -float('inf'))
            masked.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            gate_w = F.softmax(masked, dim=-1)
            routing_weights[grp_name] = gate_w

            grp_logits = torch.zeros(B, len(indices), device=concat_expert_feats.device)
            for m in range(self.num_experts):
                expert_pred = self.heads[grp_name][m](proj_experts[m])
                grp_logits = grp_logits + expert_pred * gate_w[:, m : m + 1]

            for local_idx, global_idx in enumerate(indices):
                logits_by_idx[global_idx] = grp_logits[:, local_idx]

        return torch.stack(logits_by_idx, dim=1), routing_weights
