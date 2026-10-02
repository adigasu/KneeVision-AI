"""
RSNA Knee Abnormality Detection - Tri-Planar Multi-View Label-Specific MIL Model (Phase 13).
Jointly ingests Sagittal, Coronal, and Axial slices (48 slices total per study),
applies learned anatomical plane embeddings (Sagittal=0, Coronal=1, Axial=2),
and uses 12 independent pathology gated-attention heads for label-specific pooling.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import timm

from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool


class TriPlanarLabelSpecificMILModel(nn.Module):
    """
    Tri-Planar 2.5D Knee MRI MIL Architecture with Learned Plane Embeddings
    and 12-Branch Label-Specific Gated Attention Pooling.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
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

        extra_kwargs = {}
        if "dinov2" in backbone_name.lower():
            extra_kwargs = {"img_size": 336, "dynamic_img_size": True}
        elif any(k in backbone_name.lower() for k in ("dinov3", "vit")):
            extra_kwargs = {"img_size": 288, "dynamic_img_size": True}

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            **extra_kwargs,
        )

        self.num_features = self.backbone.num_features

        # Learned anatomical plane embeddings: 0=Sagittal, 1=Coronal, 2=Axial
        self.plane_embedding = nn.Embedding(3, self.num_features)
        nn.init.normal_(self.plane_embedding.weight, std=0.02)

        # 12-branch label-specific gated attention pooling
        self.mil_pool = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

    def extract_slice_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts slice-level features using chunked forward passes.
        images: (B, D, C, H, W) where D is total slices (e.g. 48)
        returns: (B, D, num_features)
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
        plane_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 48, 3, H, W)
            plane_ids: Tensor of shape (B, 48) with integers in {0, 1, 2}
            mask: Optional Tensor of shape (B, 48) with bools (True=valid, False=masked)

        Returns:
            Dict containing:
                logits: (B, 12)
                attention_weights: (B, 12, 48)
                study_embedding: (B, 12, num_features)
        """
        # 1. Extract slice features: (B, 48, num_features)
        slice_feats = self.extract_slice_features(images)

        # 2. Add learned plane embeddings: (B, 48, num_features)
        plane_embeds = self.plane_embedding(plane_ids)
        slice_feats = slice_feats + plane_embeds

        # 3. Label-Specific Gated Attention Pooling across all 48 slices
        logits, attn_weights, study_embed = self.mil_pool(slice_feats, mask=mask)

        return {
            "logits": logits,
            "attention_weights": attn_weights,
            "study_embedding": study_embed,
        }
