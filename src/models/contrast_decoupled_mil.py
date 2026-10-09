"""
RSNA Knee Abnormality Detection - Multi-Contrast (T1 + T2 FS) Decoupled MIL Architecture.

Key Innovation:
Musculoskeletal MRI reading couples T1 Anatomical (bone margins, osteophytes, joint space narrowing,
cortical fractures, fibrocartilage) with T2 FS (intra-articular effusion, synovitis, bone marrow edema, ligament tears).

This architecture decouples features across BOTH anatomical planes (Sagittal, Coronal, Axial)
AND contrast mechanisms (T2 FS vs T1 Anatomical):
1. Independent Intra-Plane & Intra-Contrast Slice Attention Pooling:
   - Sagittal T2 FS (0:S)    -> h_sag_t2 (B, 12, F)
   - Sagittal T1    (S:2S)   -> h_sag_t1 (B, 12, F)
   - Coronal T2 FS  (2S:3S)  -> h_cor_t2 (B, 12, F)
   - Coronal T1     (3S:4S)  -> h_cor_t1 (B, 12, F)
   - Axial T2 FS    (4S:5S)  -> h_ax_t2  (B, 12, F)
   - Axial T1       (5S:6S)  -> h_ax_t1  (B, 12, F)
2. 6-Stream Pathology-Specific Contrast/Plane Router:
   - Learns pathology-specific dynamic routing across all 6 streams.
   - Enforces clinical inductive biases (e.g. routing OA & Fracture to T1; Effusion, Synovitis, Contusion to T2 FS).
3. Produces 12 calibrated pathology logits with complete plane and sequence interpretability.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import timm

from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool


class ContrastDecoupledMILModel(nn.Module):
    """
    6-Stream Contrast & Plane-Decoupled Knee MRI MIL Model.
    Slices per stream: S. Total slices = 6 * S.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        dropout: float = 0.3,
        in_chans: int = 3,
        chunk_size: int = 32,
        slices_per_plane: int = 16,
        classifier_type: str = "linear",
        head_hidden_dim: int = 256,
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

        # 1. Six independent intra-stream label-specific MIL pooling modules
        self.pool_sag_t2 = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_sag_t1 = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_cor_t2 = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_cor_t1 = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_ax_t2 = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_ax_t1 = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

        # 2. 6-Stream Contrast & Plane Router
        self.stream_router = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Anatomical & Contrast Prior Biases:
        # Stream order: [Sag_T2, Sag_T1, Cor_T2, Cor_T1, Ax_T2, Ax_T1]
        prior_biases = torch.tensor([
            [1.0, 0.4, 0.5, 0.3, 0.0, 0.0],  # ACL (Sag T2 dominant + Sag T1 anatomy)
            [0.0, 0.0, 2.5, 1.0, 0.0, 0.0],  # MCL (Cor T2 fluid + Cor T1 ligament outline)
            [0.8, 1.0, 0.6, 0.8, 0.0, 0.0],  # Medial Meniscus (High on T1 fibrocartilage)
            [0.8, 1.0, 0.6, 0.8, 0.0, 0.0],  # Lateral Meniscus (High on T1 fibrocartilage)
            [0.2, 1.5, 0.4, 1.5, 0.0, 0.0],  # Medial OA (High on T1 joint space narrowing)
            [0.2, 1.5, 0.4, 1.5, 0.0, 0.0],  # Lateral OA (High on T1 joint space narrowing)
            [0.0, 0.0, 0.0, 0.0, 2.0, 1.2],  # PF OA (Axial T2 + Axial T1 patellofemoral)
            [1.2, 0.0, 0.5, 0.0, 1.8, 0.0],  # Effusion (Pure T2 FS fluid sensitivity)
            [1.0, 0.0, 0.5, 0.0, 1.5, 0.0],  # Synovitis (Pure T2 FS fluid sensitivity)
            [1.2, 0.0, 0.0, 0.0, 1.2, 0.0],  # Baker's Cyst (Pure T2 FS fluid sensitivity)
            [1.5, 0.2, 1.5, 0.2, 1.5, 0.2],  # Contusion (T2 FS bone bruise dominant)
            [0.5, 1.5, 0.5, 1.5, 0.5, 1.5],  # Fracture (T1 cortical disruption dominant)
        ], dtype=torch.float32)

        self.register_buffer("prior_biases", prior_biases)

        # 3. Classifier
        self.classifier_type = classifier_type
        if classifier_type == "multi_head_mlp":
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.num_features, head_hidden_dim),
                    nn.LayerNorm(head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(num_classes)
            ])
        else:
            self.norm = nn.LayerNorm(self.num_features)
            self.dropout = nn.Dropout(dropout)
            self.classifier_w = nn.Parameter(torch.randn(num_classes, self.num_features) * (1.0 / self.num_features**0.5))
            self.classifier_b = nn.Parameter(torch.zeros(num_classes))

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
        plane_ids: Optional[torch.Tensor] = None,
        contrast_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        images: (B, 6 * S, 3, H, W)
        mask: (B, 6 * S)
        """
        B, D, C, H, W = images.shape
        S = self.slices_per_plane

        # 1. Backbone forward pass
        slice_feats = self.extract_slice_features(images)

        # 2. Deconstruct streams
        sag_t2_feats = slice_feats[:, 0 : S]
        sag_t1_feats = slice_feats[:, S : 2 * S]
        cor_t2_feats = slice_feats[:, 2 * S : 3 * S]
        cor_t1_feats = slice_feats[:, 3 * S : 4 * S]
        ax_t2_feats  = slice_feats[:, 4 * S : 5 * S]
        ax_t1_feats  = slice_feats[:, 5 * S : 6 * S]

        m_sag_t2 = mask[:, 0 : S] if mask is not None else None
        m_sag_t1 = mask[:, S : 2 * S] if mask is not None else None
        m_cor_t2 = mask[:, 2 * S : 3 * S] if mask is not None else None
        m_cor_t1 = mask[:, 3 * S : 4 * S] if mask is not None else None
        m_ax_t2  = mask[:, 4 * S : 5 * S] if mask is not None else None
        m_ax_t1  = mask[:, 5 * S : 6 * S] if mask is not None else None

        # 3. Independent Pooling
        _, _, h_sag_t2 = self.pool_sag_t2(sag_t2_feats, mask=m_sag_t2)
        _, _, h_sag_t1 = self.pool_sag_t1(sag_t1_feats, mask=m_sag_t1)
        _, _, h_cor_t2 = self.pool_cor_t2(cor_t2_feats, mask=m_cor_t2)
        _, _, h_cor_t1 = self.pool_cor_t1(cor_t1_feats, mask=m_cor_t1)
        _, _, h_ax_t2  = self.pool_ax_t2(ax_t2_feats,   mask=m_ax_t2)
        _, _, h_ax_t1  = self.pool_ax_t1(ax_t1_feats,   mask=m_ax_t1)

        # 4. Stack into 6 streams: (B, 12, 6, F)
        h_streams = torch.stack([h_sag_t2, h_sag_t1, h_cor_t2, h_cor_t1, h_ax_t2, h_ax_t1], dim=2)

        # 5. Routing Scores: (B, 12, 6)
        router_scores = self.stream_router(h_streams).squeeze(-1) + self.prior_biases.unsqueeze(0)

        # If a stream is completely masked out (e.g. Axial T1 in studies without T1 Axial), mask its router score
        if mask is not None:
            stream_valid = torch.stack([
                m_sag_t2.any(dim=1),
                m_sag_t1.any(dim=1),
                m_cor_t2.any(dim=1),
                m_cor_t1.any(dim=1),
                m_ax_t2.any(dim=1),
                m_ax_t1.any(dim=1),
            ], dim=1)  # (B, 6)
            router_scores = router_scores.masked_fill(~stream_valid.unsqueeze(1), -1e4)

        stream_weights = torch.softmax(router_scores, dim=-1)  # (B, 12, 6)

        # 6. Anatomical & Contrast Fusion
        h_fused = torch.sum(stream_weights.unsqueeze(-1) * h_streams, dim=2)  # (B, 12, F)

        # 7. Classification
        if self.classifier_type == "multi_head_mlp":
            logits = torch.cat([self.heads[k](h_fused[:, k, :]) for k in range(self.num_classes)], dim=1)
        else:
            h_norm = self.dropout(self.norm(h_fused))
            logits = torch.einsum("bkf,kf->bk", h_norm, self.classifier_w) + self.classifier_b

        return {
            "logits": logits,
            "stream_weights": stream_weights,
        }
