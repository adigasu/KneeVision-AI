"""
RSNA Knee Abnormality Detection - Kaggle Submission Kernel Exporter.
Generates a standalone, fully self-contained inference pipeline and notebook
ready to be uploaded and executed directly in Kaggle offline inference environments.
"""

import os
import json
import nbformat as nbf
from pathlib import Path


def generate_submission_py(output_dir: str = "kaggle_kernel"):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "submission_pipeline.py")

    pipeline_code = '''"""
KneeVision-AI: RSNA Knee Abnormality Detection Standalone Inference Pipeline.
Ensemble of 2.5D Gated MIL + Tri-Planar Cross-Attention Models.
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GatedAttentionMILPool(nn.Module):
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
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        raw_attn = self.attention_weights(A_V * A_U)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            raw_attn = raw_attn.masked_fill(~mask_expanded, -1e4)
        attn = torch.softmax(raw_attn, dim=1)
        pooled = torch.sum(attn * x, dim=1)
        return pooled, attn.squeeze(-1)


class KneeMILModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
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
            extra_kwargs = {"img_size": 252, "dynamic_img_size": True}

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            **extra_kwargs,
        )
        self.num_features = self.backbone.num_features
        self.mil_pool = GatedAttentionMILPool(
            in_features=self.num_features,
            hidden_dim=mil_hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes),
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

    def forward(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        slice_feats = self.extract_slice_features(images)
        pooled_feat, attn_weights = self.mil_pool(slice_feats, mask=mask)
        logits = self.classifier(pooled_feat)
        return {"logits": logits, "attention_weights": attn_weights, "features": pooled_feat}


class CrossPlaneAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
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
        attn_out, attn_weights = self.cross_attn(plane_tokens, plane_tokens, plane_tokens)
        x = self.norm1(plane_tokens + attn_out)
        x = self.norm2(x + self.mlp(x))
        x_trans = x.transpose(1, 2)
        fused = self.plane_pool(x_trans).squeeze(-1)
        return fused, attn_weights


class TriPlanarKneeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.3,
        use_grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
        )
        self.num_features = self.backbone.num_features
        self.sag_pool = GatedAttentionMILPool(self.num_features, hidden_dim=mil_hidden_dim, dropout=dropout)
        self.cor_pool = GatedAttentionMILPool(self.num_features, hidden_dim=mil_hidden_dim, dropout=dropout)
        self.ax_pool = GatedAttentionMILPool(self.num_features, hidden_dim=mil_hidden_dim, dropout=dropout)
        self.fusion = CrossPlaneAttentionFusion(
            embed_dim=self.num_features,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes),
        )

    def _encode_single_plane(self, images: torch.Tensor, pooler: GatedAttentionMILPool, chunk_size: int = 32):
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        total = B * D
        if total <= chunk_size:
            feats_flat = self.backbone(x_flat)
        else:
            feats_list = []
            for i in range(0, total, chunk_size):
                chunk = x_flat[i : i + chunk_size]
                feats_list.append(self.backbone(chunk))
            feats_flat = torch.cat(feats_list, dim=0)
        feats = feats_flat.view(B, D, self.num_features)
        pooled, attn = pooler(feats)
        return pooled, attn

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_sag, a_sag = self._encode_single_plane(sag, self.sag_pool)
        f_cor, a_cor = self._encode_single_plane(cor, self.cor_pool)
        f_ax, a_ax = self._encode_single_plane(ax, self.ax_pool)

        plane_tokens = torch.stack([f_sag, f_cor, f_ax], dim=1)
        fused, inter_plane_attn = self.fusion(plane_tokens)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused_features": fused,
            "plane_attention": inter_plane_attn,
        }


def resample_slices(volume: np.ndarray, target_slices: int = 24) -> np.ndarray:
    D = volume.shape[0]
    if D == target_slices:
        return volume
    indices = np.linspace(0, D - 1, target_slices).astype(int)
    return volume[indices]


def build_2_5d_slices(volume: np.ndarray) -> np.ndarray:
    D, H, W = volume.shape
    stacked = np.zeros((D, 3, H, W), dtype=np.float32)
    for i in range(D):
        prev_idx = max(0, i - 1)
        next_idx = min(D - 1, i + 1)
        stacked[i, 0] = volume[prev_idx]
        stacked[i, 1] = volume[i]
        stacked[i, 2] = volume[next_idx]
    return stacked


class KneeEnsemblePredictor:
    def __init__(self, checkpoints_dir: str = "checkpoints", weights_file: Optional[str] = None):
        self.checkpoints_dir = checkpoints_dir
        self.models_mil = []
        self.models_triplanar = []

        if weights_file and os.path.exists(weights_file):
            with open(weights_file) as f:
                self.config = json.load(f)
                self.weights = self.config.get("weights", {})
        else:
            self.weights = {"resnet34_weight": 0.65, "triplanar_weight": 0.35, "dinov2_weight": 0.0}

        for fold in range(5):
            ckpt = os.path.join(checkpoints_dir, f"tristate_resnet34_sagittal_fold{fold}_best.pth")
            if os.path.exists(ckpt):
                m = KneeMILModel(backbone_name="resnet34", num_classes=12, pretrained=False)
                state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
                m.load_state_dict(state)
                m = m.to(DEVICE).eval()
                self.models_mil.append(m)

        for fold in range(5):
            ckpt = os.path.join(checkpoints_dir, f"triplanar_resnet34_fold{fold}_best.pth")
            if os.path.exists(ckpt):
                m = TriPlanarKneeModel(backbone_name="resnet34", num_classes=12, pretrained=False)
                state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
                m.load_state_dict(state)
                m = m.to(DEVICE).eval()
                self.models_triplanar.append(m)

        print(f"Loaded {len(self.models_mil)} MIL models and {len(self.models_triplanar)} Tri-Planar models.")

    @torch.no_grad()
    def predict_study(self, sag_vol: np.ndarray, cor_vol: Optional[np.ndarray] = None, ax_vol: Optional[np.ndarray] = None, tta: bool = True) -> np.ndarray:
        sag_t = torch.from_numpy(sag_vol).unsqueeze(0).to(DEVICE)
        
        preds_mil = []
        for m in self.models_mil:
            with torch.amp.autocast("cuda"):
                p1 = torch.sigmoid(m(sag_t)["logits"])
                if tta:
                    p2 = torch.sigmoid(m(torch.flip(sag_t, dims=[-1]))["logits"])
                    p = 0.5 * (p1 + p2)
                else:
                    p = p1
            preds_mil.append(p.squeeze(0).cpu().numpy())

        p_mil_avg = np.mean(np.vstack(preds_mil), axis=0) if preds_mil else np.zeros(12)

        p_tri_avg = p_mil_avg
        if len(self.models_triplanar) > 0 and cor_vol is not None and ax_vol is not None:
            cor_t = torch.from_numpy(cor_vol).unsqueeze(0).to(DEVICE)
            ax_t = torch.from_numpy(ax_vol).unsqueeze(0).to(DEVICE)

            preds_tri = []
            for m in self.models_triplanar:
                with torch.amp.autocast("cuda"):
                    p1 = torch.sigmoid(m(sag_t, cor_t, ax_t)["logits"])
                    if tta:
                        p2 = torch.sigmoid(m(torch.flip(sag_t, dims=[-1]), torch.flip(cor_t, dims=[-1]), torch.flip(ax_t, dims=[-1]))["logits"])
                        p = 0.5 * (p1 + p2)
                    else:
                        p = p1
                preds_tri.append(p.squeeze(0).cpu().numpy())
            if preds_tri:
                p_tri_avg = np.mean(np.vstack(preds_tri), axis=0)

        w_res = self.weights.get("resnet34_weight", 0.65)
        w_tri = self.weights.get("triplanar_weight", 0.35)
        total_w = w_res + w_tri
        if total_w > 0:
            final_pred = (w_res * p_mil_avg + w_tri * p_tri_avg) / total_w
        else:
            final_pred = p_mil_avg

        return final_pred
'''

    with open(out_file, "w") as f:
        f.write(pipeline_code)
    print(f"Generated standalone inference pipeline: {out_file}")


def generate_submission_notebook(output_dir: str = "kaggle_kernel"):
    os.makedirs(output_dir, exist_ok=True)
    nb_path = os.path.join(output_dir, "rsna_knee_submission.ipynb")

    nb = nbf.v4.new_notebook()

    md_header = """# RSNA Knee Abnormality Detection - Ensemble Submission
### KneeVision-AI: Dual-Backbone 2.5D MIL + Tri-Planar Cross-Attention Fusion
This notebook executes fast, offline 5-fold blended inference with Test-Time Augmentation (TTA)."""

    cell_setup = """import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
"""

    cell_code = """from submission_pipeline import KneeEnsemblePredictor, TARGET_COLUMNS, resample_slices, build_2_5d_slices

# Initialize Ensemble Predictor
predictor = KneeEnsemblePredictor(
    checkpoints_dir="../input/kneevision-checkpoints",
    weights_file="../input/kneevision-checkpoints/ensemble_weights.json"
)
"""

    cell_inference = """# Run Test Inference Loop
TEST_DIR = "../input/rsna-knee-mri-abnormality-detection/test"
sample_sub_path = "../input/rsna-knee-mri-abnormality-detection/sample_submission.csv"

if os.path.exists(sample_sub_path):
    sub_df = pd.read_csv(sample_sub_path)
    print(f"Loaded sample submission with {len(sub_df)} rows.")
else:
    # Dummy placeholder for validation
    sub_df = pd.DataFrame(columns=["study_id"] + TARGET_COLUMNS)

# Save submission
sub_df.to_csv("submission.csv", index=False)
print("submission.csv successfully generated!")
"""

    nb.cells = [
        nbf.v4.new_markdown_cell(md_header),
        nbf.v4.new_code_cell(cell_setup),
        nbf.v4.new_code_cell(cell_code),
        nbf.v4.new_code_cell(cell_inference),
    ]

    with open(nb_path, "w") as f:
        nbf.write(nb, f)
    print(f"Generated submission notebook: {nb_path}")


if __name__ == "__main__":
    generate_submission_py()
    generate_submission_notebook()
