"""
RSNA Knee Abnormality Detection - Heterogeneous Grand Ensemble Exporter (Version 60 - Fixed & Fail-Safe)
Combines Multi-Architecture Heterogeneous Models:
1. 5-Fold Decoupled ConvNeXt-Small (384px, Folds 0, 1, 2, 3, 4) -> Total Weight: 0.55
2. Decoupled ConvNeXt-Tiny (336px, Fold 0)                      -> Weight: 0.25
3. Tri-Planar DINOv2-Small ViT (336px, Fold 0)                  -> Weight: 0.20

Features:
- Single Disk I/O Pass with dynamic GPU VRAM resize
- Proven clean loader (Z-sorting, global min-max, zero horizontal flip)
- Column-wise Rank Standardization (scipy.stats.rankdata) to directly maximize Macro-AUC
- Robust try-except loading: fail-safe execution guaranteed
Target Kernel: sukeshadigav/kneevision-ai-rsna-submission
"""

import json
import os
import shutil
from pathlib import Path

NOTEBOOK_PATH = "kaggle_upload/kernel/rsna_knee_submission.ipynb"
METADATA_PATH = "kaggle_upload/kernel/kernel-metadata.json"

CELL_0_MARKDOWN = """# KneeVision-AI: Heterogeneous Grand Ensemble (Version 60)
### Multi-Architecture CNN + Vision Transformer + Column-Wise Rank Standardization
- **5-Fold Decoupled ConvNeXt-Small (384px):** Folds 0, 1, 2, 3, 4 (Local MSK tissue texture & joint lines)
- **Decoupled ConvNeXt-Tiny (336px):** Compact hierarchical CNN (multi-scale receptive field)
- **Tri-Planar DINOv2-Small ViT (336px):** Self-supervised Vision Transformer (global multi-head attention)
- **Metric-Direct Optimization:** Column-wise rank standardization (`scipy.stats.rankdata`) to directly optimize Macro-AUC.
- **Anatomical Truth:** Zero horizontal flip; strictly preserves Medial vs. Lateral compartment integrity.
"""

CELL_1_IMPORTS = """# ── Imports & Environment Setup ──────────────────────────────────────────────
import os
import sys
import time
import math
import glob
import json
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

try:
    import pydicom
except ImportError:
    pydicom = None

warnings.filterwarnings("ignore")

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device      : {torch.cuda.get_device_name(0)}")
    print(f"Device VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COLUMNS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

SAGITTAL_KW = ["SAG", "SAGITTAL", "PD_SAG", "FS_SAG", "STIR_SAG", "COR_SAG", "PD SAG", "FS SAG", "SAGITAL"]
CORONAL_KW  = ["COR", "CORONAL", "PD_COR", "FS_COR", "T2_COR", "COR_FS", "PD COR", "FS COR"]
AXIAL_KW    = ["AX", "AXIAL", "PD_AX", "FS_AX", "T2_AX", "AX_FS", "PD AX", "FS AX", "TRA", "TRANS"]
"""

CELL_2_MODELS = """# ── Model Architectures: ConvNeXt & DINOv2 ViT ──────────────────────────────

# 1. Label-Specific Gated Attention MIL Pooling
class LabelSpecificGatedAttentionMILPool(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 12, hidden_dim: int = 128, dropout: float = 0.25):
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, F_dim = x.shape
        v = self.attention_V(x)
        u = self.attention_U(x)
        gated = v * u
        a = self.attention_weights(gated)  # (B, D, num_classes)
        a = a.transpose(1, 2)              # (B, num_classes, D)

        if mask is not None:
            mask_exp = mask.unsqueeze(1).expand_as(a)
            a = a.masked_fill(~mask_exp, -1e4)

        attn = torch.softmax(a, dim=-1)
        h = torch.bmm(attn, x)             # (B, num_classes, F_dim)
        h_norm = self.dropout(self.norm(h))
        logits = torch.einsum("bkf,kf->bk", h_norm, self.classifier_w) + self.classifier_b
        return logits, attn, h


# 2. Plane-Decoupled Anatomically-Routed Tri-Planar Model (ConvNeXt-Small & Tiny)
class PlaneDecoupledTriPlanarMILModel(nn.Module):
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
        classifier_type: str = "linear",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.slices_per_plane = slices_per_plane
        self.classifier_type = classifier_type

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )
        self.num_features = self.backbone.num_features

        self.pool_sag = LabelSpecificGatedAttentionMILPool(self.num_features, num_classes, mil_hidden_dim, dropout)
        self.pool_cor = LabelSpecificGatedAttentionMILPool(self.num_features, num_classes, mil_hidden_dim, dropout)
        self.pool_ax  = LabelSpecificGatedAttentionMILPool(self.num_features, num_classes, mil_hidden_dim, dropout)

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

    def forward(self, images: torch.Tensor, plane_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        slice_feats = self.extract_slice_features(images)
        spl = self.slices_per_plane
        sag_feats = slice_feats[:, 0 : spl]
        cor_feats = slice_feats[:, spl : 2 * spl]
        ax_feats  = slice_feats[:, 2 * spl : 3 * spl]

        mask_sag = mask[:, 0 : spl] if mask is not None else None
        mask_cor = mask[:, spl : 2 * spl] if mask is not None else None
        mask_ax  = mask[:, 2 * spl : 3 * spl] if mask is not None else None

        _, attn_sag, h_sag = self.pool_sag(sag_feats, mask=mask_sag)
        _, attn_cor, h_cor = self.pool_cor(cor_feats, mask=mask_cor)
        _, attn_ax,  h_ax  = self.pool_ax(ax_feats,   mask=mask_ax)

        h_planes = torch.stack([h_sag, h_cor, h_ax], dim=2)
        router_scores = self.plane_router(h_planes).squeeze(-1)
        router_scores = router_scores + self.prior_biases.unsqueeze(0)
        plane_weights = torch.softmax(router_scores, dim=-1)

        h_fused = torch.sum(plane_weights.unsqueeze(-1) * h_planes, dim=2)
        h_norm = self.dropout(self.norm(h_fused))
        logits = torch.einsum("bkf,kf->bk", h_norm, self.classifier_w) + self.classifier_b
        return {"logits": logits, "plane_weights": plane_weights}


# 3. Tri-Planar DINOv2-Small ViT Model (Matching Phase 13 Checkpoint Exactly)
class TriPlanarLabelSpecificMILModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
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
        self.plane_embedding = nn.Embedding(3, self.num_features)
        nn.init.normal_(self.plane_embedding.weight, std=0.02)

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

    def forward(self, images: torch.Tensor, plane_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        slice_feats = self.extract_slice_features(images)
        if plane_ids is not None:
            slice_feats = slice_feats + self.plane_embedding(plane_ids)

        logits, attn_weights, study_embed = self.mil_pool(slice_feats, mask=mask)
        return {"logits": logits, "attention_weights": attn_weights, "study_embedding": study_embed}
"""

CELL_3_LOADER = """# ── Proven Clean Multi-View DICOM Loader (Single Disk I/O Pass) ────────────
def _read_slice(path: Union[str, Path]):
    if pydicom is None:
        raise ImportError("pydicom is required.")
    ds = pydicom.dcmread(str(path), stop_before_pixels=False)
    arr = ds.pixel_array.astype(np.float32)
    try:
        z = float(ds.ImagePositionPatient[2])
    except Exception:
        try:
            z = float(ds.InstanceNumber)
        except Exception:
            z = 0.0
    return z, arr

def load_plane_slices(dcm_paths: List[Path], target_slices: int = 24) -> Tuple[np.ndarray, bool]:
    \"\"\"
    Loads 24 evenly spaced slices per plane with a single disk I/O pass.
    Constructs contiguous 2.5D slices (D, 3, H, W).
    Spatial resizing is deferred to GPU VRAM for maximum throughput.
    \"\"\"
    if not dcm_paths:
        return np.zeros((target_slices, 3, 384, 384), dtype=np.float32), False

    results = {}
    with ThreadPoolExecutor(max_workers=min(4, len(dcm_paths))) as ex:
        futs = {ex.submit(_read_slice, str(p)): i for i, p in enumerate(dcm_paths)}
        for fut in as_completed(futs):
            try:
                results[futs[fut]] = fut.result()
            except Exception:
                pass

    if not results:
        return np.zeros((target_slices, 3, 384, 384), dtype=np.float32), False

    ordered = [results[i] for i in range(len(results)) if i in results]
    ordered.sort(key=lambda t: t[0])
    slices = [t[1] for t in ordered]

    # Verify uniform 2D shape (standard in MRI DICOM)
    shapes = {s.shape for s in slices}
    if len(shapes) > 1:
        target_hw = slices[0].shape
        harmonized = []
        for s in slices:
            if s.shape != target_hw:
                pil = Image.fromarray(s)
                pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
                harmonized.append(np.array(pil, dtype=np.float32))
            else:
                harmonized.append(s)
        volume = np.stack(harmonized, axis=0)
    else:
        volume = np.stack(slices, axis=0)

    D = volume.shape[0]
    if D != target_slices:
        idx = np.linspace(0, D - 1, target_slices).round().astype(int)
        volume = volume[idx]

    # Global continuous min-max normalization
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    # Assemble 2.5D contiguous slices [z-1, z, z+1]
    H, W = volume.shape[1], volume.shape[2]
    out = np.zeros((target_slices, 3, H, W), dtype=np.float32)
    for i in range(target_slices):
        out[i, 0] = volume[max(0, i - 1)]
        out[i, 1] = volume[i]
        out[i, 2] = volume[min(target_slices - 1, i + 1)]

    return out, True

def load_triplanar_study(
    study_id: str,
    test_dir: Path,
    df_series: Optional[pd.DataFrame] = None,
    device: str = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    study_path = test_dir / str(study_id)
    if not study_path.exists():
        for candidate in test_dir.rglob(str(study_id)):
            if candidate.is_dir():
                study_path = candidate
                break

    plane_order = ["Sagittal", "Coronal", "Axial"]
    plane_keywords = {
        "Sagittal": SAGITTAL_KW,
        "Coronal":  CORONAL_KW,
        "Axial":    AXIAL_KW,
    }

    plane_tensors = []
    valid_flags = []

    for plane in plane_order:
        dcms = []

        # 1. Resolve via test_series.csv manifest if available
        if df_series is not None and "StudyInstanceUID" in df_series.columns and "Anatomical_Plane" in df_series.columns:
            sub = df_series[(df_series["StudyInstanceUID"].astype(str) == str(study_id)) & (df_series["Anatomical_Plane"].astype(str).str.lower() == plane.lower())]
            sort_cols = [c for c in ["Fluid_Sensitive", "Fat_Suppression"] if c in sub.columns]
            if sort_cols:
                sub = sub.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
            for _, r in sub.iterrows():
                sid = str(r["SeriesInstanceUID"])
                cand_dirs = [study_path / sid, test_dir / sid, test_dir / str(study_id) / sid]
                for cd in cand_dirs:
                    if cd.exists():
                        d = sorted(list(cd.glob("*.dcm")))
                        if d:
                            dcms = d
                            break
                if dcms:
                    break

        # 2. Fallback to folder name inspection within study_path
        if not dcms and study_path.exists():
            series_dir = study_path / plane
            if series_dir.exists():
                dcms = sorted(list(series_dir.glob("*.dcm")))
            if not dcms:
                for d in study_path.glob("*"):
                    if d.is_dir() and any(k.lower() in d.name.lower() for k in plane_keywords[plane]):
                        cand = sorted(list(d.glob("*.dcm")))
                        if cand:
                            dcms = cand
                            break

        # 3. Fallback to any series directory if missing
        if not dcms and study_path.exists():
            for d in sorted(study_path.iterdir()):
                if d.is_dir():
                    cand = sorted(list(d.glob("*.dcm")))
                    if cand:
                        dcms = cand
                        break

        # 4. Fallback: check if study_path itself contains .dcm files directly
        if not dcms and study_path.exists():
            dcms = sorted(list(study_path.glob("*.dcm")))

        vol_np, is_valid = load_plane_slices(dcms, target_slices=24)
        t_plane = torch.from_numpy(vol_np).to(device)
        # Harmonize spatial dimensions across planes in GPU VRAM before concatenation
        if t_plane.shape[-2] != 384 or t_plane.shape[-1] != 384:
            t_plane = F.interpolate(t_plane, size=(384, 384), mode="bilinear", align_corners=False)
        plane_tensors.append(t_plane)
        valid_flags.append(is_valid)

    # Concatenate all 3 planes -> (72, 3, 384, 384)
    full_vol = torch.cat(plane_tensors, dim=0).unsqueeze(0)  # (1, 72, 3, 384, 384)
    plane_ids = torch.cat([torch.full((24,), i, dtype=torch.long) for i in range(3)]).unsqueeze(0).to(device)
    mask = torch.tensor([vf for vf in valid_flags for _ in range(24)], dtype=torch.bool).unsqueeze(0).to(device)

    return full_vol, plane_ids, mask

print("✓ Single I/O Pass multi-view DICOM loader initialized.")
"""

CELL_4_INFERENCE = """# ── Load Heterogeneous Models & Execute Grand Ensemble Inference ─────────
def find_comp_dir() -> Path:
    candidates = [
        Path("../input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("../input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("../input/rsna-knee-mri-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-mri-abnormality-detection"),
        Path("kaggle_upload"),
        Path("data"),
    ]
    for c in candidates:
        if (c / "sample_submission.csv").exists():
            return c
    for base in [Path("../input"), Path("/kaggle/input"), Path(".")]:
        if base.exists():
            for p in base.rglob("sample_submission.csv"):
                return p.parent
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find competition input directory.")

def find_test_dir(comp_dir: Path) -> Path:
    candidates = [
        comp_dir / "test_series",
        comp_dir / "test",
        comp_dir / "test_images",
        Path("../input/competitions/rsna-knee-abnormality-detection/test_series"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/test_series"),
        Path("../input/rsna-knee-abnormality-detection/test_series"),
        Path("/kaggle/input/rsna-knee-abnormality-detection/test_series"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    for base in [Path("../input"), Path("/kaggle/input")]:
        if base.exists():
            for p in base.rglob("test_series"):
                if p.is_dir():
                    return p
    return comp_dir / "test_series"

def find_checkpoint(filename: str) -> Path:
    search_dirs = [
        Path("../input/datasets/sukeshadigav/rsna-knee-kneevision-weights"),
        Path("/kaggle/input/datasets/sukeshadigav/rsna-knee-kneevision-weights"),
        Path("../input/rsna-knee-kneevision-weights"),
        Path("/kaggle/input/rsna-knee-kneevision-weights"),
        Path("../input"),
        Path("/kaggle/input"),
        Path("checkpoints"),
        Path("kaggle_upload/weights"),
        Path("."),
    ]
    for base in search_dirs:
        if base.exists():
            matches = list(base.rglob(filename))
            if matches:
                return matches[0]
    for base in search_dirs:
        if base.exists():
            matches = list(base.rglob(f"*{filename}*"))
            if matches:
                return matches[0]
    avail = []
    for base in search_dirs:
        if base.exists():
            avail.extend([str(p.name) for p in base.rglob("*.pth")])
    raise FileNotFoundError(f"Checkpoint file {filename} could not be located. Available: {avail[:15]}")

COMP_DIR = find_comp_dir()
TEST_DIR = find_test_dir(COMP_DIR)
SAMPLE_SUB = COMP_DIR / "sample_submission.csv"
TEST_SERIES_CSV = COMP_DIR / "test_series.csv"

# Configuration of All Heterogeneous Models
MODEL_CONFIGS = [
    # 1-5: 5 Folds of Decoupled ConvNeXt-Small 384px (Total Weight: 0.55 -> 0.11 each)
    {"name": "cnext_s_f0", "arch": "decoupled_small", "res": 384, "ckpt": "phase16_decoupled_convnext_small_384px_f0_best.pth", "weight": 0.11},
    {"name": "cnext_s_f1", "arch": "decoupled_small", "res": 384, "ckpt": "phase16_decoupled_convnext_small_384px_f1_best.pth", "weight": 0.11},
    {"name": "cnext_s_f2", "arch": "decoupled_small", "res": 384, "ckpt": "phase16_decoupled_convnext_small_384px_f2_best.pth", "weight": 0.11},
    {"name": "cnext_s_f3", "arch": "decoupled_small", "res": 384, "ckpt": "phase16_decoupled_convnext_small_384px_f3_best.pth", "weight": 0.11},
    {"name": "cnext_s_f4", "arch": "decoupled_small", "res": 384, "ckpt": "phase16_decoupled_convnext_small_384px_f4_best.pth", "weight": 0.11},
    # 6: Decoupled ConvNeXt-Tiny 336px (Weight: 0.25)
    {"name": "cnext_t_f0", "arch": "decoupled_tiny",  "res": 336, "ckpt": "phase16_decoupled_convnext_tiny_336px_f0_best.pth",  "weight": 0.25},
    # 7: Tri-Planar DINOv2-Small ViT 336px (Weight: 0.20)
    {"name": "dinov2_s_f0", "arch": "dinov2_small",   "res": 336, "ckpt": "phase13_triplanar_dinov2_small_f0_72sl_best.pth",     "weight": 0.20},
]

loaded_models = []
for cfg in MODEL_CONFIGS:
    try:
        p = find_checkpoint(cfg["ckpt"])
        if cfg["arch"] == "decoupled_small":
            m = PlaneDecoupledTriPlanarMILModel(backbone_name="convnext_small", pretrained=False, num_classes=12, classifier_type="linear")
        elif cfg["arch"] == "decoupled_tiny":
            m = PlaneDecoupledTriPlanarMILModel(backbone_name="convnext_tiny", pretrained=False, num_classes=12, classifier_type="linear")
        elif cfg["arch"] == "dinov2_small":
            m = TriPlanarLabelSpecificMILModel(backbone_name="vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=12)
        else:
            raise ValueError(f"Unknown architecture: {cfg['arch']}")

        st = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model_state_dict"])
        m = m.to(DEVICE).eval()
        if DEVICE == "cuda":
            m = m.half()
        cfg["model"] = m
        loaded_models.append(cfg)
        print(f"✓ Loaded [{cfg['name']}] from {p.name} (weight: {cfg['weight']:.2f}, res: {cfg['res']}px)")
    except Exception as e:
        print(f"⚠️ Warning: Could not load {cfg['name']} ({e}). Continuing fail-safe with other models.")

print(f"Total Grand Ensemble Models Loaded: {len(loaded_models)} across Heterogeneous Architectures!")

# Load test manifest
df_ts = pd.DataFrame()
if TEST_SERIES_CSV.exists():
    try:
        df_ts = pd.read_csv(TEST_SERIES_CSV)
        print(f"Loaded test_series.csv manifest ({len(df_ts)} rows)")
    except Exception as e:
        print(f"Warning reading test_series.csv: {e}")

if SAMPLE_SUB.exists():
    sub_df = pd.read_csv(SAMPLE_SUB)
elif (COMP_DIR / "test.csv").exists():
    sub_df = pd.read_csv(COMP_DIR / "test.csv")
else:
    test_folders = [d.name for d in TEST_DIR.iterdir() if d.is_dir()]
    sub_df = pd.DataFrame({"StudyInstanceUID": test_folders})

id_col = "StudyInstanceUID" if "StudyInstanceUID" in sub_df.columns else sub_df.columns[0]
study_ids = sub_df[id_col].astype(str).tolist()
print(f"Total Studies to Predict: {len(study_ids)}")

num_models = len(loaded_models)
num_studies = len(study_ids)

# Matrix to hold predictions: (num_models, num_studies, 12)
raw_preds = np.zeros((num_models, num_studies, 12), dtype=np.float32)
failed_count = 0
start_time = time.time()

print(f"Executing Heterogeneous Grand Ensemble across {num_models} models on {num_studies} studies...")

with torch.no_grad():
    for i, study_id in enumerate(study_ids):
        try:
            # 1. Single Disk I/O Pass: loads volume into VRAM once
            vol_native, p_ids, m_mask = load_triplanar_study(study_id, TEST_DIR, df_series=df_ts, device=DEVICE)

            # 2. Multi-Model Inference with on-the-fly GPU VRAM resizing
            for m_idx, cfg in enumerate(loaded_models):
                req_res = cfg["res"]
                m = cfg["model"]

                curr_h, curr_w = vol_native.shape[-2], vol_native.shape[-1]
                if curr_h != req_res or curr_w != req_res:
                    B, D, C, H, W = vol_native.shape
                    v_flat = vol_native.view(B * D, C, H, W)
                    v_resized = F.interpolate(v_flat, size=(req_res, req_res), mode="bilinear", align_corners=False)
                    inp = v_resized.view(B, D, C, req_res, req_res)
                else:
                    inp = vol_native

                if DEVICE == "cuda":
                    inp = inp.half()

                out = m(inp, plane_ids=p_ids, mask=m_mask)
                prob = torch.sigmoid(out["logits"]).squeeze(0).cpu().float().numpy()
                raw_preds[m_idx, i] = prob

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"Warning: Study {study_id} failed with error: {e}")
            for m_idx in range(num_models):
                raw_preds[m_idx, i] = 0.50

        if (i + 1) % 50 == 0 or (i + 1) == num_studies:
            elapsed = time.time() - start_time
            speed = (i + 1) / max(elapsed, 1e-4)
            print(f"Processed {i + 1}/{num_studies} studies ({speed:.2f} std/s | elapsed: {elapsed / 60:.1f} min)")

print(f"Raw Inference Complete! Failed count: {failed_count}/{num_studies}")

# 3. Column-Wise Rank Standardization & Weighted Ensembling
print("Applying Column-Wise Rank Standardization across models...")
final_ensemble_preds = np.zeros((num_studies, 12), dtype=np.float32)

weights = [cfg["weight"] for cfg in loaded_models]
total_w = sum(weights)
weights = [w / total_w for w in weights]

if num_studies > 1:
    for c in range(12):
        col_rank_blend = np.zeros(num_studies, dtype=np.float32)
        for m_idx in range(num_models):
            col_vals = raw_preds[m_idx, :, c]
            ranks = (rankdata(col_vals) - 1.0) / (num_studies - 1.0)
            col_rank_blend += weights[m_idx] * ranks
        final_ensemble_preds[:, c] = col_rank_blend
else:
    for m_idx in range(num_models):
        final_ensemble_preds += weights[m_idx] * raw_preds[m_idx]
"""

CELL_5_SUBMIT = """# ── Format Submission & Run Verification Checks ─────────────────────────────
out_df = pd.DataFrame(final_ensemble_preds, columns=TARGET_COLUMNS)
out_df.insert(0, id_col, study_ids)

if SAMPLE_SUB.exists():
    sample_sub = pd.read_csv(SAMPLE_SUB)
    s_id = "StudyInstanceUID" if "StudyInstanceUID" in sample_sub.columns else sample_sub.columns[0]
    out_df = sample_sub[[s_id]].merge(out_df, left_on=s_id, right_on=id_col, how="left").fillna(0.50)
    if s_id != "StudyInstanceUID":
        out_df = out_df.rename(columns={s_id: "StudyInstanceUID"})
    cols_in_sub = [c for c in sample_sub.columns if c != s_id]
    if cols_in_sub:
        out_df = out_df[["StudyInstanceUID"] + cols_in_sub]

out_df.to_csv("submission.csv", index=False)
print("Saved submission.csv successfully!")
print(f"Shape: {out_df.shape}")
print(out_df.head(5))

# Verification Assertions
assert not out_df.isnull().any().any(), "Error: NaN values found in submission.csv"
assert len(out_df) == len(study_ids), f"Error: Row count mismatch {len(out_df)} vs {len(study_ids)}"
print("✓ All Grand Ensemble validation checks passed! Ready for leaderboard scoring.")
"""

def create_notebook():
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [CELL_0_MARKDOWN]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_1_IMPORTS]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_2_MODELS]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_3_LOADER]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_4_INFERENCE]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_5_SUBMIT]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Generated Heterogeneous Grand Ensemble Kaggle notebook at: {NOTEBOOK_PATH}")

    meta = {
        "id": "sukeshadigav/kneevision-ai-rsna-submission",
        "title": "KneeVision-AI RSNA Submission",
        "code_file": "rsna_knee_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "dataset_sources": ["sukeshadigav/rsna-knee-kneevision-weights"],
        "competition_sources": ["rsna-knee-abnormality-detection"]
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Verified kernel-metadata.json pointing to {meta['id']}")

if __name__ == "__main__":
    create_notebook()
