"""
RSNA Knee Abnormality Detection - 2.5D Multiple Instance Learning (MIL) Architecture.
Extracts slice-level feature embeddings with modern vision backbones (timm) and
aggregates them into study-level abnormality predictions using gated attention pooling.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning Pooling (Ilse et al., ICML 2018).
    Dynamically learns which MRI slices contain abnormal pathology.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, Slices, InFeatures).

        Returns:
            pooled: Study-level feature vector (Batch, InFeatures).
            attention_weights: Slice attention scores (Batch, Slices, 1).
        """
        # Gated attention computation
        v = self.attention_V(x)
        u = self.attention_U(x)
        gated = self.dropout(v * u)
        
        # Softmax over slice dimension (dim=1)
        scores = self.attention_weights(gated)  # (B, Slices, 1)
        weights = F.softmax(scores, dim=1)      # (B, Slices, 1)

        # Weighted sum of slice features
        pooled = torch.sum(x * weights, dim=1)  # (B, InFeatures)
        return pooled, weights


class KneeMILModel(nn.Module):
    """
    Complete Multi-Instance Learning Network for Knee Abnormality Detection.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        num_classes: int = 12,
        in_channels: int = 1,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Load 2D backbone with custom in_channels
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # feature extractor mode
        )
        embed_dim = self.encoder.num_features

        # MIL Aggregation layer
        self.mil_pool = GatedAttentionMIL(
            in_features=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 12-target multi-label classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, Slices, 1, H, W).

        Returns:
            logits: (B, 12) unnormalized abnormality logits.
            attention_weights: (B, Slices, 1) slice attention weights.
        """
        B, Slices, C, H, W = x.shape

        # Reshape to pass all slices through 2D backbone concurrently
        flat_x = x.view(B * Slices, C, H, W)
        slice_feats = self.encoder(flat_x)  # (B * Slices, embed_dim)

        # Reshape back to (B, Slices, embed_dim)
        slice_feats = slice_feats.view(B, Slices, -1)

        # Aggregate slices via Gated Attention MIL
        study_feats, attn_weights = self.mil_pool(slice_feats)

        # Predict 12 abnormality logits
        logits = self.classifier(study_feats)  # (B, 12)

        return logits, attn_weights
