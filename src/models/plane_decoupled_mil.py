"""
RSNA Knee Abnormality Detection - Plane-Decoupled Anatomically-Routed Tri-Planar MIL Model.

Key Innovation:
Instead of a single global attention softmax across 72 mixed slices (which allows the 48
irrelevant Sagittal/Axial slices to drown out the Coronal signal for MCL, causing MCL OOF to stall at 0.661),
this architecture:
1. Decouples intra-plane slice attention:
   - Sagittal slices (24) are pooled independently -> z_sag (B, 12, F)
   - Coronal slices (24) are pooled independently -> z_cor (B, 12, F)
   - Axial slices (24) are pooled independently -> z_ax (B, 12, F)
2. Learns pathology-specific anatomical cross-plane routing gates:
   - For MCL, the gate concentrates attention on Coronal features.
   - For ACL / Meniscus, the gate blends Sagittal + Coronal.
   - For PF OA / Effusion / Synovitis, the gate routes to Axial + Sagittal.
3. Produces 12 calibrated pathology logits with complete plane-specific interpretability.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import timm

from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool


class PlaneDecoupledTriPlanarMILModel(nn.Module):
    """
    Anatomically-Decoupled Tri-Planar Knee MRI MIL Architecture.
    Slices per plane: 24 Sagittal (0:24), 24 Coronal (24:48), 24 Axial (48:72) = 72 slices total.
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
        slices_per_plane: int = 24,
        classifier_type: str = "multi_head_mlp",
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.slices_per_plane = slices_per_plane

        extra_kwargs = {}
        if backbone_name.lower() == "biomedclip":
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.backbone = clip_model.visual
            self.num_features = 512
        elif backbone_name.lower() in ("orthofoundation", "orthofoundation_vitl"):
            import timm, os
            self.backbone = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0, img_size=224, dynamic_img_size=True)
            ckpt_path = 'weights/orthofoundation/OrthoFoundation-L.pth'
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                ckpt_stripped = {k.replace('backbone.', ''): v for k, v in ckpt.items()}
                self.backbone.load_state_dict(ckpt_stripped, strict=False)
            self.num_features = 1024
        elif backbone_name.lower() in ("radimagenet_resnet50", "radimagenet"):
            from huggingface_hub import hf_hub_download
            import torchvision.models as models
            weights_path = hf_hub_download(repo_id='convergedmachine/RadImagenet', filename='resnet50.pth')
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('module.fc.')}
            res = models.resnet50()
            res.load_state_dict(state_dict, strict=False)
            class ResNetBackbone(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.features = nn.Sequential(*list(m.children())[:-1])
                    self.num_features = 2048
                def forward(self, x):
                    return self.features(x).flatten(1)
            self.backbone = ResNetBackbone(res)
            self.num_features = 2048
        else:
            if "dinov2" in backbone_name.lower():
                extra_kwargs = {"img_size": 336, "dynamic_img_size": True}
            elif any(k in backbone_name.lower() for k in ("dinov3", "vit")):
                extra_kwargs = {"img_size": 288, "dynamic_img_size": True}
            elif "swin" in backbone_name.lower():
                extra_kwargs = {"img_size": 256, "dynamic_img_size": True}

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_chans,
                **extra_kwargs,
            )
            self.num_features = self.backbone.num_features

        # 1. Three independent intra-plane label-specific MIL pooling modules
        self.pool_sag = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_cor = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.pool_ax = LabelSpecificGatedAttentionMILPool(
            in_features=self.num_features,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )

        # 2. Pathology-Specific Plane Router: projects pooled feature (B, 12, 3, F) -> scalar routing score
        self.plane_router = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Anatomical Prior Bias (Sagittal=0, Coronal=1, Axial=2) for all 12 pathologies:
        # TARGET_COLUMNS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
        #                   'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
        prior_biases = torch.tensor([
            [1.0, 0.5, 0.0],   # ACL: Sagittal primary, Coronal secondary
            [0.0, 2.5, 0.0],   # MCL: Coronal dominant (crucial bottleneck fix!)
            [1.0, 0.8, 0.0],   # Medial Meniscus: Sagittal + Coronal
            [1.0, 0.8, 0.0],   # Lateral Meniscus: Sagittal + Coronal
            [0.5, 1.2, 0.0],   # Medial OA: Coronal joint space narrowing
            [0.5, 1.2, 0.0],   # Lateral OA: Coronal joint space narrowing
            [0.5, 0.0, 2.0],   # PF OA: Axial patellofemoral compartment dominant
            [0.5, 0.5, 1.5],   # Effusion: Suprapatellar pouch on Axial & Sagittal
            [0.5, 0.5, 1.5],   # Synovitis: Hoffa's pad / capsule on Axial & Sagittal
            [1.0, 0.0, 1.0],   # Baker's: Popliteal fossa on Axial & Sagittal
            [0.8, 0.8, 0.8],   # Contusion: Multi-compartment bone marrow edema
            [0.8, 0.8, 0.8],   # Fracture: Multi-compartment cortical disruption
        ], dtype=torch.float32)  # (12, 3)

        self.register_buffer("prior_biases", prior_biases)

        # 3. Multi-Head Pathology Classifier
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
        """
        Extracts slice-level features using chunked forward passes.
        images: (B, D, C, H, W) where D is total slices (e.g. 72)
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
        plane_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 72, 3, H, W)
                    Slices 0:24 are Sagittal, 24:48 are Coronal, 48:72 are Axial.
            plane_ids: Optional plane indicator tensor (unused, ordering is fixed).
            mask: Optional Tensor of shape (B, 72) with bools (True=valid, False=masked).

        Returns:
            Dict containing:
                logits: (B, 12)
                plane_weights: (B, 12, 3) softmax routing weights over [Sag, Cor, Ax]
                attn_sag: (B, 12, 24)
                attn_cor: (B, 12, 24)
                attn_ax:  (B, 12, 24)
        """
        B, D, C, H, W = images.shape
        S = self.slices_per_plane

        # 1. Extract slice features: (B, 72, F)
        slice_feats = self.extract_slice_features(images)

        # 2. Slice decoupling
        sag_feats = slice_feats[:, :S]
        cor_feats = slice_feats[:, S : 2 * S]
        ax_feats = slice_feats[:, 2 * S : 3 * S]

        sag_mask = mask[:, :S] if mask is not None else None
        cor_mask = mask[:, S : 2 * S] if mask is not None else None
        ax_mask = mask[:, 2 * S : 3 * S] if mask is not None else None

        # 3. Independent Intra-Plane Pooling
        # Note: pool forward returns (logits_unused, attn, study_embed)
        _, attn_sag, h_sag = self.pool_sag(sag_feats, mask=sag_mask)  # h_sag: (B, 12, F), attn_sag: (B, 12, S)
        _, attn_cor, h_cor = self.pool_cor(cor_feats, mask=cor_mask)  # h_cor: (B, 12, F), attn_cor: (B, 12, S)
        _, attn_ax,  h_ax  = self.pool_ax(ax_feats,  mask=ax_mask)   # h_ax:  (B, 12, F), attn_ax:  (B, 12, S)

        # 4. Stack plane representations: (B, 12, 3, F)
        h_planes = torch.stack([h_sag, h_cor, h_ax], dim=2)

        # 5. Compute pathology-specific plane routing weights
        # Router score: (B, 12, 3, 1) -> (B, 12, 3)
        router_scores = self.plane_router(h_planes).squeeze(-1)

        # Add prior biases: (12, 3) -> broadcast to (B, 12, 3)
        router_scores = router_scores + self.prior_biases.unsqueeze(0)

        # Softmax over the 3 planes (Sagittal=0, Coronal=1, Axial=2)
        plane_weights = torch.softmax(router_scores, dim=-1)  # (B, 12, 3)

        # 6. Anatomical Fusion: weighted combination of the 3 planes per pathology
        # (B, 12, 1, 3) @ (B, 12, 3, F) -> (B, 12, 1, F) -> (B, 12, F)
        h_fused = torch.sum(plane_weights.unsqueeze(-1) * h_planes, dim=2)

        # 7. Final Classification
        if self.classifier_type == "multi_head_mlp":
            logits = torch.cat([self.heads[k](h_fused[:, k, :]) for k in range(self.num_classes)], dim=1)  # (B, 12)
        else:
            h_norm = self.dropout(self.norm(h_fused))  # (B, 12, F)
            logits = torch.einsum("bkf,kf->bk", h_norm, self.classifier_w) + self.classifier_b  # (B, 12)

        return {
            "logits": logits,
            "plane_weights": plane_weights,
            "attn_sag": attn_sag,
            "attn_cor": attn_cor,
            "attn_ax": attn_ax,
        }
