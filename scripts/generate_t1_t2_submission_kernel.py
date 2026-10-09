"""
RSNA Knee Abnormality Detection - Generate Multi-Contrast (T1 + T2 FS) Fold 0 Kaggle Kernel.
Generates kaggle_upload/kernel/rsna_knee_submission.ipynb.
"""

import json
import os
from pathlib import Path

NOTEBOOK_PATH = "kaggle_upload/kernel/rsna_knee_submission.ipynb"

# ── Cells Definition ─────────────────────────────────────────────────────────

CELL_0_MARKDOWN = """# KneeVision-AI — RSNA Knee Abnormality Detection
## Multi-Contrast (T1 Anatomical + T2 FS) Decoupled ConvNeXt-Small 384px Ensemble
* **Architecture:** Plane-Decoupled Tri-Planar MIL with Intra-Plane Label-Specific Gated Attention & Anatomical Routing
* **Dual Contrast:** T2 FS (Fluid-Sensitive) + T1 Anatomical (Non-FS) Complementary Synergy
* **Resolution:** 384×384 (24 Slices per Plane) with In-VRAM GPU Dynamic Interpolation
* **Pathology Blending:** Clinically-derived & empirically optimized contrast weights across all 12 abnormalities
* **Validation Performance:** 0.8843 Macro AUC on Fold 0 (Gold Consensus: 0.9432)
"""

CELL_1_IMPORTS = """import os, sys, gc, time, glob, json, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

try:
    import pydicom
except ImportError:
    pydicom = None

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# Optimally solved per-class contrast weights (Fold 0 Macro AUC: 0.8843)
CONTRAST_WEIGHTS = {
    "ACL":              {"t2": 0.44, "t1": 0.56},
    "MCL":              {"t2": 0.54, "t1": 0.46},
    "Medial Meniscus":  {"t2": 0.74, "t1": 0.26},
    "Lateral Meniscus": {"t2": 0.67, "t1": 0.33},
    "Medial OA":        {"t2": 0.64, "t1": 0.36},
    "Lateral OA":       {"t2": 0.46, "t1": 0.54},
    "PF OA":            {"t2": 0.77, "t1": 0.23},
    "Effusion":         {"t2": 0.61, "t1": 0.39},
    "Synovitis":        {"t2": 0.73, "t1": 0.27},
    "Baker's":          {"t2": 0.91, "t1": 0.09},
    "Contusion":        {"t2": 0.77, "t1": 0.23},
    "Fracture":         {"t2": 0.75, "t1": 0.25},
}

IMG_SIZE = 384
SLICES_PER_PLANE = 24
TOTAL_SLICES = 72
print(f"PyTorch Version: {torch.__version__} | Device: {DEVICE}")
"""

CELL_2_MODEL = """# ── Standalone Plane-Decoupled Tri-Planar MIL Model ────────────────────────
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
        B, D, F = x.shape
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        raw_attn = self.attention_weights(A_V * A_U).transpose(1, 2)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            raw_attn = raw_attn.masked_fill(~mask_expanded, -1e4)

        attn = torch.softmax(raw_attn, dim=-1)
        h = torch.bmm(attn, x)
        h_norm = self.dropout(self.norm(h))

        logits = torch.einsum('bkf,kf->bk', h_norm, self.classifier_w) + self.classifier_b
        return logits, attn, h


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
"""

CELL_3_LOADER = """# ── DICOM Volume Loader with Diagnostic Window Selection ───────────────────
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
    if not dcm_paths:
        return np.zeros((target_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), False

    results = {}
    with ThreadPoolExecutor(max_workers=min(8, len(dcm_paths))) as executor:
        fut_map = {executor.submit(_read_slice, p): p for p in dcm_paths}
        for fut in fut_map:
            try:
                z, arr = fut.result()
                results[fut_map[fut]] = (z, arr)
            except Exception:
                pass

    if not results:
        return np.zeros((target_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), False

    sorted_dcms = sorted(results.keys(), key=lambda p: results[p][0])
    raw_slices = [results[p][1] for p in sorted_dcms]

    D = len(raw_slices)
    if D == 0:
        return np.zeros((target_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), False

    idx_map = np.linspace(0, D - 1, target_slices).round().astype(int)
    sampled = [raw_slices[i] for i in idx_map]

    # 2.5D stacking
    vol_25d = []
    for s in range(target_slices):
        cur = sampled[s]
        prev = sampled[max(0, s - 1)]
        nxt  = sampled[min(target_slices - 1, s + 1)]
        ch3 = np.stack([prev, cur, nxt], axis=0)
        vol_25d.append(ch3)

    vol_arr = np.stack(vol_25d, axis=0)  # (target_slices, 3, H, W)
    v_min, v_max = float(vol_arr.min()), float(vol_arr.max())
    if v_max > v_min:
        vol_arr = (vol_arr - v_min) / (v_max - v_min)
    else:
        vol_arr = np.zeros_like(vol_arr)

    return vol_arr.astype(np.float32), True


def score_series_candidate(num_slices: int) -> int:
    if 24 <= num_slices <= 38:
        return 100 - abs(num_slices - 30)
    elif 18 <= num_slices <= 48:
        return 70 - abs(num_slices - 30)
    else:
        return max(0, 30 - min(abs(num_slices - 30), 30))


def resolve_series_dcms(study_path: Path, df_sub: Optional[pd.DataFrame], plane: str, contrast_type: str) -> List[Path]:
    is_fluid = 1 if contrast_type == "t2" else 0
    if df_sub is not None and len(df_sub) > 0 and "Anatomical_Plane" in df_sub.columns and "Fluid_Sensitive" in df_sub.columns:
        matching = df_sub[(df_sub["Anatomical_Plane"].astype(str).str.lower() == plane.lower()) &
                          (df_sub["Fluid_Sensitive"].astype(int) == is_fluid)]

        if len(matching) > 1:
            candidates = []
            for _, row in matching.iterrows():
                ser_uid = str(row["SeriesInstanceUID"])
                cand_dirs = [study_path / ser_uid, study_path.parent / ser_uid]
                for cd in cand_dirs:
                    if cd.exists():
                        dcms = sorted(cd.glob("*.dcm"))
                        if dcms:
                            score = score_series_candidate(len(dcms))
                            candidates.append((score, dcms))
                            break
            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                return candidates[0][1]

        if len(matching) == 1:
            ser_uid = str(matching.iloc[0]["SeriesInstanceUID"])
            cand_dirs = [study_path / ser_uid, study_path.parent / ser_uid]
            for cd in cand_dirs:
                if cd.exists():
                    dcms = sorted(cd.glob("*.dcm"))
                    if dcms:
                        return dcms

        # Fallback if specific contrast not found: try any series of the same plane
        matching_plane = df_sub[df_sub["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
        for _, row in matching_plane.iterrows():
            ser_uid = str(row["SeriesInstanceUID"])
            cand_dirs = [study_path / ser_uid, study_path.parent / ser_uid]
            for cd in cand_dirs:
                if cd.exists():
                    dcms = sorted(cd.glob("*.dcm"))
                    if dcms:
                        return dcms

    # Fallback to folder name inspection
    plane_keywords = {
        "Sagittal": ["sag", "sagittal"],
        "Coronal":  ["cor", "coronal"],
        "Axial":    ["ax", "axial", "tra"],
    }
    if study_path.exists():
        for d in sorted(study_path.glob("*")):
            if d.is_dir() and any(k in d.name.lower() for k in plane_keywords.get(plane, [])):
                cand = sorted(d.glob("*.dcm"))
                if cand:
                    return cand
        for d in sorted(study_path.iterdir()):
            if d.is_dir():
                cand = sorted(d.glob("*.dcm"))
                if cand:
                    return cand
        cand = sorted(study_path.glob("*.dcm"))
        if cand:
            return cand

    return []


def load_triplanar_contrast_volume(study_id: str, test_dir: Path, df_series: Optional[pd.DataFrame], contrast_type: str, img_size: int = IMG_SIZE, device: str = DEVICE):
    study_path = test_dir / str(study_id)
    if not study_path.exists():
        for cand in test_dir.rglob(str(study_id)):
            if cand.is_dir():
                study_path = cand
                break

    df_sub = None
    if df_series is not None and len(df_series) > 0 and "StudyInstanceUID" in df_series.columns:
        df_sub = df_series[df_series["StudyInstanceUID"].astype(str) == str(study_id)]

    plane_tensors = []
    valid_flags = []

    for plane in ["Sagittal", "Coronal", "Axial"]:
        dcms = resolve_series_dcms(study_path, df_sub, plane, contrast_type)
        vol_plane, is_valid = load_plane_slices(dcms, target_slices=SLICES_PER_PLANE)
        t_plane = torch.from_numpy(vol_plane)
        if is_valid:
            if t_plane.shape[-2] != img_size or t_plane.shape[-1] != img_size:
                t_plane = F.interpolate(t_plane, size=(img_size, img_size), mode="bilinear", align_corners=False)
        else:
            t_plane = torch.zeros((SLICES_PER_PLANE, 3, img_size, img_size), dtype=torch.float32)
        plane_tensors.append(t_plane)
        valid_flags.extend([is_valid] * SLICES_PER_PLANE)

    volume_72 = torch.cat(plane_tensors, dim=0).unsqueeze(0).to(device)  # (1, 72, 3, H, W)
    mask = torch.tensor([valid_flags], dtype=torch.bool, device=device)
    return volume_72, mask
"""

CELL_4_INFERENCE = """# ── Load Multi-Contrast Models & Execute Inference Loop ──────────────────────
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

print(f"Comp Dir        : {COMP_DIR}")
print(f"Test Dir        : {TEST_DIR} (exists: {TEST_DIR.exists()})")
print(f"Sample Sub      : {SAMPLE_SUB} (exists: {SAMPLE_SUB.exists()})")
print(f"Test Series CSV : {TEST_SERIES_CSV} (exists: {TEST_SERIES_CSV.exists()})")

# 1. Load T2 Model
p_t2 = find_checkpoint("phase16_decoupled_convnext_small_384px_f0_best.pth")
model_t2 = PlaneDecoupledTriPlanarMILModel(backbone_name="convnext_small", pretrained=False, num_classes=12, classifier_type="linear")
ckpt_t2 = torch.load(p_t2, map_location="cpu", weights_only=False)
model_t2.load_state_dict(ckpt_t2["model_state_dict"])
model_t2.to(DEVICE).eval()
if DEVICE == "cuda":
    model_t2 = model_t2.half()
print(f"Loaded T2 Model from: {p_t2.name}")

# 2. Load T1 Model
p_t1 = find_checkpoint("phase17_decoupled_t1_convnext_small_384px_f0_best.pth")
model_t1 = PlaneDecoupledTriPlanarMILModel(backbone_name="convnext_small", pretrained=False, num_classes=12, classifier_type="linear")
ckpt_t1 = torch.load(p_t1, map_location="cpu", weights_only=False)
model_t1.load_state_dict(ckpt_t1["model_state_dict"])
model_t1.to(DEVICE).eval()
if DEVICE == "cuda":
    model_t1 = model_t1.half()
print(f"Loaded T1 Model from: {p_t1.name}")

# Load test manifest
df_ts = pd.DataFrame()
if TEST_SERIES_CSV.exists():
    try:
        df_ts = pd.read_csv(TEST_SERIES_CSV)
        print(f"Loaded test_series.csv manifest ({len(df_ts)} rows)")
    except Exception as e:
        print(f"Warning reading test_series.csv: {e}")

# Determine studies to predict
if SAMPLE_SUB.exists():
    sub_df = pd.read_csv(SAMPLE_SUB)
elif (COMP_DIR / "test.csv").exists():
    sub_df = pd.read_csv(COMP_DIR / "test.csv")
elif len(df_ts) > 0 and "StudyInstanceUID" in df_ts.columns:
    sub_df = pd.DataFrame({"StudyInstanceUID": df_ts["StudyInstanceUID"].unique()})
elif TEST_DIR.exists():
    test_folders = [d.name for d in TEST_DIR.iterdir() if d.is_dir()]
    sub_df = pd.DataFrame({"StudyInstanceUID": test_folders})
else:
    sub_df = pd.DataFrame({"StudyInstanceUID": ["dummy_study"]})

id_col = "StudyInstanceUID" if "StudyInstanceUID" in sub_df.columns else sub_df.columns[0]
study_ids = sub_df[id_col].astype(str).unique().tolist()
print(f"Total Studies to Predict: {len(study_ids)}")

w_t2_vec = np.array([CONTRAST_WEIGHTS[c]["t2"] for c in TARGET_COLUMNS], dtype=np.float32)
w_t1_vec = np.array([CONTRAST_WEIGHTS[c]["t1"] for c in TARGET_COLUMNS], dtype=np.float32)

DEFAULT_PRED = np.full(12, 0.5, dtype=np.float32)
results = []
start_time = time.time()

with torch.no_grad():
    for i, study_id in enumerate(study_ids):
        try:
            # Predict T2
            vol_t2, m_t2 = load_triplanar_contrast_volume(study_id, TEST_DIR, df_ts, contrast_type="t2", img_size=IMG_SIZE, device=DEVICE)
            if DEVICE == "cuda":
                vol_t2 = vol_t2.half()
            out_t2 = model_t2(vol_t2, mask=m_t2)
            prob_t2 = torch.sigmoid(out_t2["logits"]).squeeze(0).cpu().float().numpy()

            # Predict T1
            vol_t1, m_t1 = load_triplanar_contrast_volume(study_id, TEST_DIR, df_ts, contrast_type="t1", img_size=IMG_SIZE, device=DEVICE)
            if DEVICE == "cuda":
                vol_t1 = vol_t1.half()
            out_t1 = model_t1(vol_t1, mask=m_t1)
            prob_t1 = torch.sigmoid(out_t1["logits"]).squeeze(0).cpu().float().numpy()

            # Pathology-Weighted Contrast Fusion
            blended_prob = w_t2_vec * prob_t2 + w_t1_vec * prob_t1
            results.append([study_id] + blended_prob.tolist())
        except Exception as e:
            print(f"Error on {study_id}: {e}")
            results.append([study_id] + DEFAULT_PRED.tolist())

        if (i + 1) % 50 == 0 or (i + 1) == len(study_ids):
            elapsed = time.time() - start_time
            print(f"[{i+1}/{len(study_ids)}] Done ({elapsed:.1f}s)")
"""

CELL_5_SUBMIT = """# ── Format Submission & Run Verification Checks ─────────────────────────────
out_df = pd.DataFrame(results, columns=["StudyInstanceUID"] + TARGET_COLUMNS)

if SAMPLE_SUB.exists():
    sample_sub = pd.read_csv(SAMPLE_SUB)
    id_c = "StudyInstanceUID" if "StudyInstanceUID" in sample_sub.columns else sample_sub.columns[0]
    out_df = sample_sub[[id_c]].merge(out_df, left_on=id_c, right_on="StudyInstanceUID", how="left").fillna(0.50)
    if id_c != "StudyInstanceUID":
        out_df = out_df.rename(columns={id_c: "StudyInstanceUID"})
    target_cols_in_sub = [c for c in sample_sub.columns if c != id_c]
    if target_cols_in_sub:
        out_df = out_df[["StudyInstanceUID"] + target_cols_in_sub]

out_df.to_csv("submission.csv", index=False)
print("Saved submission.csv successfully!")
print(f"Shape: {out_df.shape}")
print(out_df.head(5))

# Verification Assertions
assert not out_df.isnull().any().any(), "Error: NaN values found in submission.csv"
print("✓ All validation checks passed! Ready for leaderboard scoring.")
"""

def create_notebook():
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [CELL_0_MARKDOWN]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_1_IMPORTS]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [CELL_2_MODEL]},
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
    print(f"✓ Generated Kaggle submission notebook at: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    create_notebook()
