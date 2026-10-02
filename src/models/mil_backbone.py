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
    Complete 2.5D Knee MRI MIL Architecture with Memory-Efficient Slice Chunking.
    Ingests variable 2.5D slice volumes, pools inter-slice features with Gated Attention,
    and outputs 12 abnormality target logits.
    """

    def __init__(
        self,
        backbone_name: str = 'convnext_tiny',
        pretrained: bool = True,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.3,
        in_chans: int = 3,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.chunk_size = chunk_size

        # 1. 2D Slice Feature Extractor (Supports CNNs, ConvNeXt, and DINOv2 ViTs)
        extra_kwargs = {}
        if 'dinov2' in backbone_name.lower():
            extra_kwargs = {'img_size': 252, 'dynamic_img_size': True}

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            **extra_kwargs,
        )

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

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, 'freeze_bn', False) and mode:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.eval()

    def extract_slice_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features per slice with memory-efficient chunking.
        """
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
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        slice_feats = self.extract_slice_features(images)  # (B, D, Feat_Dim)
        study_embed, attn_weights = self.mil_pool(slice_feats, mask=mask)
        logits = self.classifier(study_embed)

        return {
            'logits': logits,
            'attention_weights': attn_weights,
            'study_embedding': study_embed,
        }


class LabelSpecificGatedAttentionMILPool(nn.Module):
    """
    Label-Specific Gated Attention Multiple Instance Learning (MIL) Pooling Layer.
    Learns 12 independent attention distributions across slices/windows (A in R^{B x 12 x D}),
    enabling each pathology (e.g. ACL tear, Meniscus tear, Effusion) to focus on its own
    anatomically relevant slices.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, F = x.shape
        A_V = self.attention_V(x)  # (B, D, H)
        A_U = self.attention_U(x)  # (B, D, H)
        raw_attn = self.attention_weights(A_V * A_U).transpose(1, 2)  # (B, K, D)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, D)
            raw_attn = raw_attn.masked_fill(~mask_expanded, -1e4)

        attn = torch.softmax(raw_attn, dim=-1)  # (B, K, D)
        h = torch.bmm(attn, x)                  # (B, K, F)
        h_norm = self.dropout(self.norm(h))     # (B, K, F)

        logits = torch.einsum('bkf,kf->bk', h_norm, self.classifier_w) + self.classifier_b  # (B, K)
        return logits, attn, h


class LabelSpecificKneeMILModel(nn.Module):
    """
    2.5D Knee MRI MIL Architecture with Label-Specific Attention Pooling.
    Supports both ConvNeXt (convnext_tiny, convnext_small) and ViT backbones (DINOv2).
    """

    def __init__(
        self,
        backbone_name: str = 'convnext_tiny',
        pretrained: bool = True,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.3,
        in_chans: int = 3,
        chunk_size: int = 32,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.freeze_bn = freeze_bn

        extra_kwargs = {}
        if 'dinov2' in backbone_name.lower():
            extra_kwargs = {'img_size': 336, 'dynamic_img_size': True}
        elif any(k in backbone_name.lower() for k in ('dinov3', 'vit')):
            extra_kwargs = {'img_size': 288, 'dynamic_img_size': True}

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            **extra_kwargs,
        )
        if self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
        self.num_features = self.backbone.num_features

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
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        slice_feats = self.extract_slice_features(images)  # (B, D, Feat_Dim)
        logits, attn_weights, class_embeds = self.mil_pool(slice_feats, mask=mask)
        return {
            'logits': logits,
            'attention_weights': attn_weights,
            'class_embeddings': class_embeds,
        }

