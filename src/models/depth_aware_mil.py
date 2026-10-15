"""
Anatomical Depth-Conditioned Spatial Tri-Planar MIL Model.
Injects continuous 1D Z-positional embeddings and anatomical Gaussian depth priors
to eliminate cross-compartment false positives and enforce anatomical inductive bias.
"""

from typing import Dict, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models.plane_decoupled_mil import LabelSpecificGatedAttentionMILPool


class SinusoidalDepthEmbedding(nn.Module):
    """Generates continuous positional embeddings for relative slice depth z in [0, 1]."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D) with values in [0, 1]
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=z.device) * -emb_scale)
        # Outer product
        args = z.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, D, half_dim)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, D, embed_dim)
        return self.mlp(pe)


class AnatomicalDepthMILPool(nn.Module):
    """
    Gated Attention MIL Pool with Anatomical Depth Conditioning.
    Adds learned depth embeddings and soft Gaussian spatial priors per pathology.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        plane_type: str = "sagittal",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.plane_type = plane_type

        self.depth_embed = SinusoidalDepthEmbedding(in_features)

        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_weights = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.classifier_w = nn.Parameter(torch.randn(num_classes, in_features) * (1.0 / in_features**0.5))
        self.classifier_b = nn.Parameter(torch.zeros(num_classes))

        # Anatomical Depth Centers (mu) and Spread (sigma) for each target
        # Targets: [ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture]
        if plane_type == "sagittal":
            # ACL is central (0.50); Lateral Meniscus is outer (0.18); Medial Meniscus is inner (0.82)
            mu = torch.tensor([0.50, 0.50, 0.82, 0.18, 0.75, 0.25, 0.40, 0.50, 0.50, 0.70, 0.50, 0.50])
            sigma = torch.tensor([0.18, 0.35, 0.22, 0.22, 0.30, 0.30, 0.35, 0.45, 0.45, 0.25, 0.45, 0.45])
            strength = torch.tensor([2.5, 0.5, 2.0, 2.0, 1.2, 1.2, 1.5, 0.2, 0.2, 1.5, 0.0, 0.0])
        elif plane_type == "coronal":
            # Medial structures on one side, Lateral on the other; MCL strictly medial
            mu = torch.tensor([0.50, 0.85, 0.80, 0.20, 0.80, 0.20, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50])
            sigma = torch.tensor([0.40, 0.18, 0.22, 0.22, 0.25, 0.25, 0.40, 0.40, 0.40, 0.40, 0.45, 0.45])
            strength = torch.tensor([0.2, 2.8, 1.8, 1.8, 1.8, 1.8, 0.2, 0.3, 0.3, 0.2, 0.0, 0.0])
        else: # axial
            mu = torch.full((12,), 0.50)
            sigma = torch.full((12,), 0.40)
            strength = torch.tensor([0.0, 1.0, 0.2, 0.2, 0.5, 0.5, 2.5, 1.5, 1.5, 2.0, 0.0, 0.0])

        self.register_buffer("prior_mu", mu)
        self.register_buffer("prior_sigma", sigma)
        self.register_buffer("prior_strength", strength)

    def forward(
        self,
        x: torch.Tensor,
        relative_depths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, F_dim = x.shape

        if relative_depths is None:
            relative_depths = torch.linspace(0.0, 1.0, D, device=x.device).unsqueeze(0).expand(B, -1)

        # 1. Inject Continuous Depth Embedding
        z_emb = self.depth_embed(relative_depths)  # (B, D, F_dim)
        x_depth = x + z_emb

        # 2. Gated Attention Logits
        v = self.attention_V(x_depth)
        u = self.attention_U(x_depth)
        gated = v * u
        a = self.attention_weights(gated)  # (B, D, 12)
        a = a.transpose(1, 2)              # (B, 12, D)

        # 3. Add Anatomical Gaussian Depth Bias: strength * exp(- (z - mu)^2 / 2sigma^2)
        z_diff = relative_depths.unsqueeze(1) - self.prior_mu.unsqueeze(0).unsqueeze(-1)  # (B, 12, D)
        gauss_prior = torch.exp(-0.5 * (z_diff / self.prior_sigma.unsqueeze(0).unsqueeze(-1)) ** 2)
        depth_bias = self.prior_strength.unsqueeze(0).unsqueeze(-1) * gauss_prior
        a = a + depth_bias

        if mask is not None:
            mask_exp = mask.unsqueeze(1).expand_as(a)
            a = a.masked_fill(~mask_exp, -1e4)

        attn = torch.softmax(a, dim=-1)
        h = torch.bmm(attn, x_depth)       # (B, 12, F_dim)
        h_norm = self.dropout(self.norm(h))
        logits = torch.einsum("bkf,kf->bk", h_norm, self.classifier_w) + self.classifier_b
        return logits, attn, h


class AnatomicalDepthAwareTriPlanarMILModel(nn.Module):
    """
    SOTA Anatomical Depth-Conditioned Tri-Planar MIL Model.
    Equipped with plane routing and continuous Z-depth spatial awareness.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = False,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.3,
        in_chans: int = 3,
        chunk_size: int = 32,
        slices_per_plane: int = 24,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.slices_per_plane = slices_per_plane

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )
        self.num_features = self.backbone.num_features

        # Anatomical Depth Pools
        self.pool_sag = AnatomicalDepthMILPool(self.num_features, num_classes, mil_hidden_dim, dropout, plane_type="sagittal")
        self.pool_cor = AnatomicalDepthMILPool(self.num_features, num_classes, mil_hidden_dim, dropout, plane_type="coronal")
        self.pool_ax  = AnatomicalDepthMILPool(self.num_features, num_classes, mil_hidden_dim, dropout, plane_type="axial")

        self.plane_router = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        prior_biases = torch.tensor([
            [1.0, 0.5, 0.0],
            [0.0, 2.5, 0.0],
            [1.0, 0.8, 0.0],
            [1.0, 0.8, 0.0],
            [0.5, 1.2, 0.0],
            [0.5, 1.2, 0.0],
            [0.5, 0.0, 2.0],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 1.5],
            [1.0, 0.0, 1.0],
            [0.8, 0.8, 0.8],
            [0.8, 0.8, 0.8],
        ], dtype=torch.float32)
        self.register_buffer("prior_biases", prior_biases)

        self.norm = nn.LayerNorm(self.num_features)
        self.dropout = nn.Dropout(dropout)
        self.classifier_w = nn.Parameter(torch.randn(num_classes, self.num_features) * (1.0 / self.num_features**0.5))
        self.classifier_b = nn.Parameter(torch.zeros(num_classes))

    def extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        B, D, C, H, W = x.shape
        x_flat = x.view(B * D, C, H, W)
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
        plane_ids: Optional[torch.Tensor] = None,
        relative_depths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        slice_feats = self.extract_slice_features(images)
        spl = self.slices_per_plane

        sag_feats = slice_feats[:, 0 : spl]
        cor_feats = slice_feats[:, spl : 2 * spl]
        ax_feats  = slice_feats[:, 2 * spl : 3 * spl]

        mask_sag = mask[:, 0 : spl] if mask is not None else None
        mask_cor = mask[:, spl : 2 * spl] if mask is not None else None
        mask_ax  = mask[:, 2 * spl : 3 * spl] if mask is not None else None

        z_sag = relative_depths[:, 0 : spl] if relative_depths is not None else None
        z_cor = relative_depths[:, spl : 2 * spl] if relative_depths is not None else None
        z_ax  = relative_depths[:, 2 * spl : 3 * spl] if relative_depths is not None else None

        _, attn_sag, h_sag = self.pool_sag(sag_feats, relative_depths=z_sag, mask=mask_sag)
        _, attn_cor, h_cor = self.pool_cor(cor_feats, relative_depths=z_cor, mask=mask_cor)
        _, attn_ax,  h_ax  = self.pool_ax(ax_feats,   relative_depths=z_ax, mask=mask_ax)

        h_planes = torch.stack([h_sag, h_cor, h_ax], dim=2)
        router_scores = self.plane_router(h_planes).squeeze(-1)
        router_scores = router_scores + self.prior_biases.unsqueeze(0)
        plane_weights = torch.softmax(router_scores, dim=-1)

        h_fused = torch.sum(plane_weights.unsqueeze(-1) * h_planes, dim=2)
        h_norm = self.dropout(self.norm(h_fused))
        logits = torch.einsum("bkf,kf->bk", h_norm, self.classifier_w) + self.classifier_b
        return {"logits": logits, "plane_weights": plane_weights, "attention_sag": attn_sag, "attention_cor": attn_cor}
