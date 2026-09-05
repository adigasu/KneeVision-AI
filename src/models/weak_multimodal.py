"""
RSNA Knee Abnormality Detection - Dual-Stream Multimodal Vision-Language Architecture.
Aligns 2.5D MRI volumetric features with Spanish radiology report embeddings via CLIP-style InfoNCE
and multi-task pseudo-label multi-label classification.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import timm

from src.models.mil_backbone import GatedAttentionMILPool
from src.models.report_encoder import ClinicalReportEncoder


class DualStreamKneeModel(nn.Module):
    """
    Dual-Stream Vision-Language Foundation Model for Knee MRI:
    1. Vision Stream: 2.5D ConvNeXt-Tiny + Gated Attention MIL Pooling
    2. Text Stream: Spanish Clinical BERT (BETO)
    3. Multi-Task Heads: Contrastive Alignment + 12-Target Multi-Label Pseudo Supervision
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        text_model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        embed_dim: int = 256,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.25,
        init_temperature: float = 0.07,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. Vision Feature Extractor
        self.vision_backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            in_chans=3,
        )
        if use_grad_checkpointing and hasattr(self.vision_backbone, "set_grad_checkpointing"):
            try:
                self.vision_backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        self.num_vision_features = self.vision_backbone.num_features

        # 2. Gated Attention MIL Pool
        self.mil_pool = GatedAttentionMILPool(
            in_features=self.num_vision_features,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

        # 3. Vision Multimodal Projection
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(self.num_vision_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_vision_features, embed_dim),
        )

        # 4. Text Stream Encoder
        self.text_encoder = ClinicalReportEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_backbone=False,
            dropout=dropout,
        )

        # 5. Learnable Temperature (Logit Scale)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / init_temperature))

        # 6. Multi-Label Classification Head on Study Embedding
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_vision_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_vision_features, num_classes),
        )

    def extract_vision_embeddings(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (Batch, Depth, 3, H, W)
            mask: Optional boolean mask (Batch, Depth)

        Returns:
            image_embeds: L2-normalized image embeddings (Batch, Embed_Dim)
            study_feat: Raw unprojected study representation (Batch, Feat_Dim)
            attn_weights: Slice attention weights (Batch, Depth)
        """
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        slice_feats = self.vision_backbone(x_flat).view(B, D, self.num_vision_features)

        study_feat, attn_weights = self.mil_pool(slice_feats, mask=mask)
        proj_image = self.vision_projection(study_feat)
        image_embeds = nn.functional.normalize(proj_image, p=2, dim=-1)

        return image_embeds, study_feat, attn_weights

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        slice_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing image embeddings, text embeddings, and pseudo logits.
        """
        image_embeds, study_feat, attn_weights = self.extract_vision_embeddings(images, mask=slice_mask)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        pseudo_logits = self.classifier(study_feat)

        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "pseudo_logits": pseudo_logits,
            "logit_scale": self.logit_scale.exp(),
            "attention_weights": attn_weights,
        }

    def get_vision_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Extracts weights compatible with downstream KneeMILModel.
        """
        state_dict = {}
        for k, v in self.state_dict().items():
            if k.startswith("vision_backbone."):
                state_dict[k.replace("vision_backbone.", "backbone.")] = v
            elif k.startswith("mil_pool."):
                state_dict[k] = v
        return state_dict
