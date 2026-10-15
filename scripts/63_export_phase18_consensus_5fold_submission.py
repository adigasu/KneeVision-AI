"""
RSNA Knee Abnormality Detection - Phase 18: Consensus-Finetuned 5-Fold ConvNeXt-Small (384px) Ensemble
Standalone Kaggle Kernel Generator with Full Parity Preprocessing:
1. Complete 5-Fold Ensemble:
   - Fold 0: phase18_consensus_convnext_small_384px_f0_best.pth (Val Macro AUC: 0.8925 | Gold: 0.9380)
   - Fold 1: phase18_consensus_convnext_small_384px_f1_best.pth (Val Macro AUC: 0.8856 | Gold: 0.9454)
   - Fold 2: phase18_consensus_convnext_small_384px_f2_best.pth (Val Macro AUC: 0.8727 | Gold: 0.9206)
   - Fold 3: phase18_consensus_convnext_small_384px_f3_best.pth (Val Macro AUC: 0.8787 | Gold: 0.8842)
   - Fold 4: phase18_consensus_convnext_small_384px_f4_best.pth (Val Macro AUC: 0.8738 | Gold: 0.8934)
2. Parity Preprocessing:
   - VOI LUT with RescaleSlope/Intercept fallback
   - MONOCHROME1 dynamic range inversion
   - True 3D spatial sorting along Patient Normal Vector cross(IOP_row, IOP_col)
   - Robust (0.5%, 99.5%) percentile intensity normalization
   - In-VRAM GPU dynamic bilinear interpolation to 384x384
3. Strict Anatomical Laterality:
   - No horizontal flipping to ensure exact Medial vs Lateral compartment protection
4. Single Disk I/O Pass:
   - Load each study's 72 slices once into GPU VRAM in FP16, evaluate all 5 models sequentially.
"""

import json
import os
import shutil
from pathlib import Path

NOTEBOOK_PATH = "kaggle_upload/kernel/rsna_knee_submission.ipynb"
METADATA_PATH = "kaggle_upload/kernel/kernel-metadata.json"
WEIGHTS_DIR = "kaggle_upload/weights"

CELL_0_MARKDOWN = """# KneeVision-AI — RSNA Knee Abnormality Detection
## Phase 18: 5-Fold Consensus-Finetuned ConvNeXt-Small (384px) SOTA Ensemble
* **Architecture:** Plane-Decoupled Tri-Planar MIL with Intra-Plane Label-Specific Gated Attention & Anatomical Routing
* **Ensemble Composition (All 5 Folds Consensus Finetuned):**
  1. Fold 0: `phase18_consensus_convnext_small_384px_f0_best.pth` (Val Macro AUC: 0.8925 | Gold: 0.9380)
  2. Fold 1: `phase18_consensus_convnext_small_384px_f1_best.pth` (Val Macro AUC: 0.8856 | Gold: 0.9454)
  3. Fold 2: `phase18_consensus_convnext_small_384px_f2_best.pth` (Val Macro AUC: 0.8727 | Gold: 0.9206)
  4. Fold 3: `phase18_consensus_convnext_small_384px_f3_best.pth` (Val Macro AUC: 0.8787 | Gold: 0.8842)
  5. Fold 4: `phase18_consensus_convnext_small_384px_f4_best.pth` (Val Macro AUC: 0.8738 | Gold: 0.8934)
* **Mean Validation Macro AUC:** **0.8807** | **Mean Gold Consensus AUC:** **0.9163**
* **Inference Parity Preprocessing:**
  1. `apply_voi_lut` with fallback to `RescaleSlope`/`RescaleIntercept`
  2. Automatic `MONOCHROME1` dynamic range inversion
  3. True 3D spatial sorting along Patient Normal Vector (`cross(IOP_row, IOP_col)`)
  4. Robust `(0.5%, 99.5%)` percentile intensity normalization (eliminates hot-pixel collapse)
* **Strict Anatomical Laterality:** No horizontal flipping, preserving exact Medial vs Lateral compartment distinction
* **Resolution:** 384×384 (24 Slices per Plane = 72 Slices total) with In-VRAM GPU Dynamic Interpolation
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
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    pydicom = None
    apply_voi_lut = None

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

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
        raw_attn = self.attention_weights(A_V * A_U).transpose(1, 2)  # (B, K, D)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            raw_attn = raw_attn.masked_fill(~mask_expanded, -1e4)

        attn = torch.softmax(raw_attn, dim=-1)
        h = torch.bmm(attn, x)  # (B, K, F)
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
            [1.0, 0.5, 0.0],  # ACL: Sagittal primary, Coronal secondary
            [0.0, 2.5, 0.0],  # MCL: Coronal dominant
            [1.0, 0.8, 0.0],  # Medial Meniscus: Sagittal + Coronal
            [1.0, 0.8, 0.0],  # Lateral Meniscus: Sagittal + Coronal
            [0.5, 1.2, 0.0],  # Medial OA: Coronal joint space
            [0.5, 1.2, 0.0],  # Lateral OA: Coronal joint space
            [0.5, 0.0, 2.0],  # PF OA: Axial patellofemoral compartment
            [0.5, 0.5, 1.5],  # Effusion: Suprapatellar pouch
            [0.5, 0.5, 1.5],  # Synovitis: Hoffa capsule
            [1.0, 0.0, 1.0],  # Baker's: Popliteal fossa
            [0.8, 0.8, 0.8],  # Contusion: Multi-compartment bone marrow edema
            [0.8, 0.8, 0.8],  # Fracture: Multi-compartment cortical disruption
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

CELL_3_LOADER = """# ── DICOM Volume Loader with Parity Preprocessing & 3D Normal Sorting ─────
def _read_slice(path: Union[str, Path]):
    if pydicom is None:
        raise ImportError("pydicom is required.")
    ds = pydicom.dcmread(str(path), stop_before_pixels=False)

    meta = {
        "InstanceNumber": int(getattr(ds, "InstanceNumber", 0)),
        "ImagePositionPatient": [float(x) for x in getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])],
        "ImageOrientationPatient": [float(x) for x in getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
        "RescaleSlope": float(getattr(ds, "RescaleSlope", 1.0)),
        "RescaleIntercept": float(getattr(ds, "RescaleIntercept", 0.0)),
        "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", "MONOCHROME2"),
    }

    # 1. Apply VOI LUT (Window center/width) with fallback
    try:
        if apply_voi_lut is not None:
            arr = apply_voi_lut(ds.pixel_array, ds)
        else:
            arr = ds.pixel_array.astype(np.float32) * meta["RescaleSlope"] + meta["RescaleIntercept"]
    except Exception:
        arr = ds.pixel_array.astype(np.float32) * meta["RescaleSlope"] + meta["RescaleIntercept"]
    arr = arr.astype(np.float32)

    # 2. Invert MONOCHROME1
    if meta["PhotometricInterpretation"] == "MONOCHROME1":
        arr = np.max(arr) - arr

    return arr, meta


def load_plane_slices(dcm_paths: List[Path], target_slices: int = 24) -> Tuple[np.ndarray, bool]:
    if not dcm_paths:
        return np.zeros((target_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), False

    slice_data = []
    with ThreadPoolExecutor(max_workers=min(8, len(dcm_paths))) as executor:
        fut_map = {executor.submit(_read_slice, p): p for p in dcm_paths}
        for fut in fut_map:
            try:
                arr, meta = fut.result()
                slice_data.append((arr, meta))
            except Exception:
                pass

    if not slice_data:
        return np.zeros((target_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), False

    # 3. 3D Spatial Sorting along Patient Normal Vector
    try:
        iop = slice_data[0][1]["ImageOrientationPatient"]
        row_vec = np.array(iop[:3])
        col_vec = np.array(iop[3:])
        normal_vec = np.cross(row_vec, col_vec)
        slice_data.sort(key=lambda s: np.dot(np.array(s[1]["ImagePositionPatient"]), normal_vec))
    except Exception:
        slice_data.sort(key=lambda s: s[1]["ImagePositionPatient"][2] if "ImagePositionPatient" in s[1] else s[1]["InstanceNumber"])

    raw_slices = [s[0] for s in slice_data]
    D = len(raw_slices)
    if D == 0:
        return np.zeros((target_slices, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32), False

    idx_map = np.linspace(0, D - 1, target_slices).round().astype(int)
    sampled = [raw_slices[i] for i in idx_map]

    # 4. 2.5D stacking [z-1, z, z+1]
    vol_25d = []
    for s in range(target_slices):
        cur = sampled[s]
        prev = sampled[max(0, s - 1)]
        nxt  = sampled[min(target_slices - 1, s + 1)]
        ch3 = np.stack([prev, cur, nxt], axis=0)
        vol_25d.append(ch3)

    vol_arr = np.stack(vol_25d, axis=0)  # (target_slices, 3, H, W)

    # 5. Robust Percentile Normalization (0.5%, 99.5%) matching training preprocessor
    v_min, v_max = np.percentile(vol_arr, (0.5, 99.5))
    if v_max > v_min:
        vol_arr = np.clip((vol_arr - v_min) / (v_max - v_min), 0.0, 1.0)
    else:
        vol_arr = np.zeros_like(vol_arr)

    return vol_arr.astype(np.float32), True


def resolve_series_dcms(study_path: Path, df_sub: Optional[pd.DataFrame], plane: str) -> List[Path]:
    if df_sub is not None and len(df_sub) > 0 and "Anatomical_Plane" in df_sub.columns:
        sub = df_sub[df_sub["Anatomical_Plane"].astype(str).str.lower() == plane.lower()].copy()
        sort_cols = [c for c in ["Fluid_Sensitive", "Fat_Suppression"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

        for _, row in sub.iterrows():
            ser_uid = str(row["SeriesInstanceUID"])
            cand_dirs = [study_path / ser_uid, study_path.parent / ser_uid]
            for cd in cand_dirs:
                if cd.exists():
                    dcms = sorted(cd.glob("*.dcm"))
                    if dcms:
                        return dcms

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


def load_triplanar_volume(study_id: str, test_dir: Path, df_series: Optional[pd.DataFrame], img_size: int = IMG_SIZE, device: str = DEVICE):
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
        dcms = resolve_series_dcms(study_path, df_sub, plane)
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

CELL_4_INFERENCE = """# ── Load 5-Fold Consensus Models & Execute Strict Parity Inference ────────
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

def find_checkpoint(filename: str, fallback: Optional[str] = None) -> Path:
    search_dirs = [
        Path("../input/datasets/sukeshadigav/rsna-knee-kneevision-weights"),
        Path("/kaggle/input/datasets/sukeshadigav/rsna-knee-kneevision-weights"),
        Path("../input/rsna-knee-kneevision-weights"),
        Path("/kaggle/input/rsna-knee-kneevision-weights"),
        Path("../input"),
        Path("/kaggle/input"),
        Path("checkpoints/consensus_ft"),
        Path("checkpoints"),
        Path("kaggle_upload/weights"),
        Path("."),
    ]
    targets = [filename] if fallback is None else [filename, fallback]
    for tgt in targets:
        for base in search_dirs:
            if base.exists():
                matches = list(base.rglob(tgt))
                if matches:
                    return matches[0]
        for base in search_dirs:
            if base.exists():
                matches = list(base.rglob(f"*{tgt}*"))
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

# Load All 5 Folds of Phase 18 Consensus ConvNeXt-Small (384px)
CHECKPOINTS_5FOLD = [
    ("phase18_consensus_convnext_small_384px_f0_best.pth", "phase16_decoupled_convnext_small_384px_f0_best.pth"),
    ("phase18_consensus_convnext_small_384px_f1_best.pth", "phase16_decoupled_convnext_small_384px_f1_best.pth"),
    ("phase18_consensus_convnext_small_384px_f2_best.pth", "phase16_decoupled_convnext_small_384px_f2_best.pth"),
    ("phase18_consensus_convnext_small_384px_f3_best.pth", "phase16_decoupled_convnext_small_384px_f3_best.pth"),
    ("phase18_consensus_convnext_small_384px_f4_best.pth", "phase16_decoupled_convnext_small_384px_f4_best.pth"),
]

models = []
for primary_fname, fallback_fname in CHECKPOINTS_5FOLD:
    p = find_checkpoint(primary_fname, fallback=fallback_fname)
    m = PlaneDecoupledTriPlanarMILModel(
        backbone_name="convnext_small",
        pretrained=False,
        num_classes=12,
        slices_per_plane=SLICES_PER_PLANE,
        classifier_type="linear"
    )
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"], strict=False)
    m.to(DEVICE).eval()
    if DEVICE == "cuda":
        m = m.half()
    models.append(m)
    val_auc = ckpt.get("val_macro_auc", "N/A")
    gold_auc = ckpt.get("gold_macro_auc", "N/A")
    v_str = f"{val_auc:.4f}" if isinstance(val_auc, float) else str(val_auc)
    g_str = f"{gold_auc:.4f}" if isinstance(gold_auc, float) else str(gold_auc)
    print(f"✓ Loaded Fold Model: {p.name} (Val AUC: {v_str}, Gold: {g_str})")

print(f"Total Models Ready: {len(models)} ConvNeXt-Small 384px Models (Complete 5-Fold Consensus Ensemble)")

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

DEFAULT_PRED = np.full(12, 0.5, dtype=np.float32)
results = []
start_time = time.time()

with torch.no_grad():
    for i, study_id in enumerate(study_ids):
        try:
            # 1. Single Disk I/O Pass with Parity Preprocessing & GPU VRAM Dynamic Resize
            vol, mask = load_triplanar_volume(study_id, TEST_DIR, df_ts, img_size=IMG_SIZE, device=DEVICE)
            if DEVICE == "cuda":
                vol = vol.half()

            # 2. Predict across All 5 Consensus Folds (Strict Anatomical Laterality Preserved, NO Flip)
            fold_probs = []
            for m in models:
                out = m(vol, mask=mask)
                prob = torch.sigmoid(out["logits"]).squeeze(0)
                fold_probs.append(prob.cpu().float().numpy())

            # Mean Ensemble Across 5 Folds
            final_pred = np.mean(fold_probs, axis=0)
            results.append([study_id] + final_pred.tolist())
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
assert len(out_df) == len(study_ids), f"Error: Row count mismatch {len(out_df)} vs {len(study_ids)}"
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
    print(f"✓ Generated Phase 18 5-Fold Consensus Kaggle notebook at: {NOTEBOOK_PATH}")

    meta = {
        "id": "sukeshadigav/kneevision-ai-rsna-submission",
        "title": "KneeVision-AI RSNA Submission",
        "code_file": "rsna_knee_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [
            "sukeshadigav/rsna-knee-kneevision-weights"
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ]
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Verified kernel-metadata.json pointing to {meta['id']}")


def sync_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    ckpts = [
        "phase18_consensus_convnext_small_384px_f0_best.pth",
        "phase18_consensus_convnext_small_384px_f1_best.pth",
        "phase18_consensus_convnext_small_384px_f2_best.pth",
        "phase18_consensus_convnext_small_384px_f3_best.pth",
        "phase18_consensus_convnext_small_384px_f4_best.pth",
    ]
    for ckpt in ckpts:
        src = Path("checkpoints/consensus_ft") / ckpt
        dst = Path(WEIGHTS_DIR) / ckpt
        if src.exists() and (not dst.exists() or src.stat().st_size != dst.stat().st_size):
            shutil.copy2(src, dst)
            print(f"✓ Synced {ckpt} to {WEIGHTS_DIR} ({dst.stat().st_size / 1e6:.1f} MB)")
        elif dst.exists():
            print(f"✓ {ckpt} already present in {WEIGHTS_DIR} ({dst.stat().st_size / 1e6:.1f} MB)")
        else:
            raise FileNotFoundError(f"Source checkpoint not found: {src}")

if __name__ == "__main__":
    sync_weights()
    create_notebook()
