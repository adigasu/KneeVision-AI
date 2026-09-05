"""
RSNA Knee Abnormality Detection - 2.5D Multiple Instance Learning (MIL) Backbone.
Extracts deep features per slice using pretrained 2D CNNs / Vision Transformers,
aggregates slices via Gated Attention Pooling, and outputs 12-target abnormality logits.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import timm


class GatedAttentionMILPool(nn.Module):
    """
    Gated Attention Multiple Instance Learning (MIL) Pooling Layer.
    Reference: Ilse et al., 'Attention-based Deep Multiple Instance Learning' (ICML 2018).
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
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
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Slice feature tensor of shape (Batch, Num_Slices, In_Features).
            mask: Optional boolean tensor of shape (Batch, Num_Slices), True for valid slices.

        Returns:
            pooled: Study-level representation of shape (Batch, In_Features).
            attn: Softmax attention weights of shape (Batch, Num_Slices, 1).
        """
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        raw_attn = self.attention_weights(A_V * A_U)  # (Batch, Num_Slices, 1)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (Batch, Num_Slices, 1)
            raw_attn = raw_attn.masked_fill(~mask_expanded, -1e4)

        attn = torch.softmax(raw_attn, dim=1)  # (Batch, Num_Slices, 1)
        pooled = torch.sum(attn * x, dim=1)    # (Batch, In_Features)
        return pooled, attn.squeeze(-1)


class KneeMILModel(nn.Module):
    """
    Complete 2.5D Knee MRI MIL Architecture with Gradient Checkpointing.
    Ingests variable 2.5D slice volumes, pools inter-slice features with Gated Attention,
    and outputs 12 abnormality target logits.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.3,
        in_chans: int = 3,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # 1. 2D Slice Feature Extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )
        if use_grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        self.num_features = self.backbone.num_features

        # 2. Gated Attention MIL Pooling Module
        self.mil_pool = GatedAttentionMILPool(
            in_features=self.num_features,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

        # 3. 12-Target Multi-Label Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes),
        )

    def extract_slice_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features per study to preserve memory.
        """
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        feats_flat = self.backbone(x_flat)
        return feats_flat.view(B, D, self.num_features)

    def forward(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (Batch, Max_Depth, 3, Height, Width).
            mask: Boolean tensor of shape (Batch, Max_Depth), True for valid slices.

        Returns:
            dict containing:
                'logits': Tensor of shape (Batch, 12)
                'attention_weights': Tensor of shape (Batch, Max_Depth)
                'study_embedding': Tensor of shape (Batch, Num_Features)
        """
        slice_feats = self.extract_slice_features(images)  # (B, D, Feat_Dim)
        study_embed, attn_weights = self.mil_pool(slice_feats, mask=mask)
        logits = self.classifier(study_embed)

        return {
            "logits": logits,
            "attention_weights": attn_weights,
            "study_embedding": study_embed,
        }
