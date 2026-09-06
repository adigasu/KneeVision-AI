"""
RSNA Knee Abnormality Detection - Tri-Planar Multi-View Cross-Attention Architecture.
Jointly processes and fuses orthogonal MRI planes (Sagittal + Coronal + Axial) per study
using intra-plane Gated Attention MIL and inter-plane Cross-Attention Fusion.
"""

from typing import Dict, Tuple, Optional, List, Union
import torch
import torch.nn as nn
import timm

from src.models.mil_backbone import GatedAttentionMILPool


class CrossPlaneAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Layer fusing 3 plane embeddings (Sagittal, Coronal, Axial).
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.plane_pool = nn.Linear(3, 1)

    def forward(self, plane_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            plane_tokens: Tensor of shape (Batch, 3, Embed_Dim) representing [Sagittal, Coronal, Axial].

        Returns:
            fused_study: (Batch, Embed_Dim)
            attn_weights: (Batch, 3, 3)
        """
        # Self/Cross-Attention across planes
        attn_out, attn_weights = self.cross_attn(plane_tokens, plane_tokens, plane_tokens)
        x = self.norm1(plane_tokens + attn_out)
        x = self.norm2(x + self.mlp(x))  # (Batch, 3, Embed_Dim)

        # Weighted pooling across the 3 planes
        x_trans = x.transpose(1, 2)       # (Batch, Embed_Dim, 3)
        fused = self.plane_pool(x_trans).squeeze(-1)  # (Batch, Embed_Dim)

        return fused, attn_weights


class TriPlanarKneeModel(nn.Module):
    """
    Complete Tri-Planar Multi-View Knee MRI Architecture.
    Processes:
      - Sagittal series (Cruciate Ligaments, Menisci)
      - Coronal series (Collateral Ligaments, Bone Edema/OA)
      - Axial series (Patellofemoral joint, Effusion, Synovitis)
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.3,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Shared 2.5D Slice Feature Extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
        )
        if use_grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        self.num_features = self.backbone.num_features

        # Independent Gated Attention MIL Poolers per anatomical plane
        self.sag_pool = GatedAttentionMILPool(self.num_features, hidden_dim=mil_hidden_dim, dropout=dropout)
        self.cor_pool = GatedAttentionMILPool(self.num_features, hidden_dim=mil_hidden_dim, dropout=dropout)
        self.ax_pool = GatedAttentionMILPool(self.num_features, hidden_dim=mil_hidden_dim, dropout=dropout)

        # Inter-Plane Multi-Head Cross Attention
        self.fusion = CrossPlaneAttentionFusion(
            embed_dim=self.num_features,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Multi-Label Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes),
        )

    def _encode_single_plane(
        self,
        images: torch.Tensor,
        pooler: GatedAttentionMILPool,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of plane series: (B, D, 3, H, W) -> (B, Num_Features)
        """
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        slice_feats = self.backbone(x_flat).view(B, D, self.num_features)
        plane_embed, attn_w = pooler(slice_feats, mask=mask)
        return plane_embed, attn_w

    def forward(
        self,
        sagittal_images: torch.Tensor,
        coronal_images: torch.Tensor,
        axial_images: torch.Tensor,
        sag_mask: Optional[torch.Tensor] = None,
        cor_mask: Optional[torch.Tensor] = None,
        ax_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass fusing Sagittal, Coronal, and Axial volumes.
        """
        z_sag, attn_sag = self._encode_single_plane(sagittal_images, self.sag_pool, mask=sag_mask)
        z_cor, attn_cor = self._encode_single_plane(coronal_images, self.cor_pool, mask=cor_mask)
        z_ax, attn_ax = self._encode_single_plane(axial_images, self.ax_pool, mask=ax_mask)

        # Stack into 3 plane tokens: (Batch, 3, Num_Features)
        plane_tokens = torch.stack([z_sag, z_cor, z_ax], dim=1)

        # Multi-Head Cross-Plane Fusion
        fused_study, inter_plane_attn = self.fusion(plane_tokens)

        # 12 Target Logits
        logits = self.classifier(fused_study)

        return {
            "logits": logits,
            "fused_study": fused_study,
            "plane_tokens": plane_tokens,
            "inter_plane_attn": inter_plane_attn,
            "attn_sag": attn_sag,
            "attn_cor": attn_cor,
            "attn_ax": attn_ax,
        }
