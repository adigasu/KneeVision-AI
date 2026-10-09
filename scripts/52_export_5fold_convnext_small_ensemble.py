"""
RSNA Knee Abnormality Detection - 5-Fold Decoupled ConvNeXt-Small (384px) Ensemble Exporter
Exports a standalone, self-contained kernel with:
- Full 5-Fold ConvNeXt-Small 384px (Folds 0, 1, 2, 3, 4)
- Single I/O Pass (Load Once from Disk, Resize in GPU VRAM via torch.nn.functional.interpolate)
- Pure FP16 execution
Target Kernel: sukeshadigav/kneevision-ai-rsna-submission
"""

import os
import json
import shutil
from pathlib import Path
import nbformat as nbf

def create_5fold_ensemble_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()

    # Papermill / Kaggle metadata
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    }

    # Markdown Header
    header_md = """# KneeVision-AI — RSNA Knee Abnormality Detection
## 5-Fold Decoupled ConvNeXt-Small (384px) Complete Ensemble
* **Architecture:** Plane-Decoupled Tri-Planar MIL Model (`convnext_small`)
* **Resolution:** 384 × 384 (24 Sagittal, 24 Coronal, 24 Axial = 72 slices per study)
* **Classifier:** Linear Gated Attention MIL (`classifier_type="linear"`)
* **Ensemble Composition (All 5 Folds):**
  1. Fold 0 Checkpoint: `phase16_decoupled_convnext_small_384px_f0_best.pth` (Val Macro AUC: 0.8752)
  2. Fold 1 Checkpoint: `phase16_decoupled_convnext_small_384px_f1_best.pth` (Val Macro AUC: 0.8775)
  3. Fold 2 Checkpoint: `phase16_decoupled_convnext_small_384px_f2_best.pth` (Val Macro AUC: 0.8605)
  4. Fold 3 Checkpoint: `phase16_decoupled_convnext_small_384px_f3_best.pth` (Val Macro AUC: 0.8630)
  5. Fold 4 Checkpoint: `phase16_decoupled_convnext_small_384px_f4_best.pth` (Val Macro AUC: 0.8640)
* **Aggregation:** Unweighted Mean Probability Ensemble: `(P_f0 + P_f1 + P_f2 + P_f3 + P_f4) / 5.0`
* **Single I/O Pass (Load Once, Dynamic GPU VRAM Resize):**
  - DICOM slices read from disk exactly once per study on CPU.
  - Slices stacked, normalized, and transferred directly to GPU VRAM.
  - Spatial resizing to 384px performed in GPU VRAM via `torch.nn.functional.interpolate` (< 5 ms).
  - All 5 models evaluate the exact same in-VRAM tensor sequentially with zero disk re-reads.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))

    # Cell 1: Environment & Directory Resolution
    cell_env = """import os, sys, gc, time, glob, json, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
IMG_SIZE = 384
BACKBONE_NAME = "convnext_small"

CKPT_NAMES = [
    "phase16_decoupled_convnext_small_384px_f0_best.pth",
    "phase16_decoupled_convnext_small_384px_f1_best.pth",
    "phase16_decoupled_convnext_small_384px_f2_best.pth",
    "phase16_decoupled_convnext_small_384px_f3_best.pth",
    "phase16_decoupled_convnext_small_384px_f4_best.pth",
]

TARGET_COLUMNS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

SAGITTAL_KW = ["SAG", "SAGITTAL", "PD_SAG", "FS_SAG", "STIR_SAG", "COR_SAG", "PD SAG", "FS SAG", "SAGITAL"]
CORONAL_KW  = ["COR", "CORONAL", "PD_COR", "FS_COR", "T2_COR", "COR_FS", "PD COR", "FS COR"]
AXIAL_KW    = ["AX", "AXIAL", "PD_AX", "FS_AX", "T2_AX", "AX_FS", "PD AX", "FS AX", "TRA", "TRANS"]

# ── Robust Competition Directory Discovery ──────────────────────────────────
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

CKPT_PATHS = []
for name in CKPT_NAMES:
    p = find_checkpoint(name)
    CKPT_PATHS.append(p)
    print(f"Loaded Checkpoint: {name} -> {p}")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_env))

    # Cell 2: Model Architecture
    cell_arch = """# ── Standalone Plane-Decoupled Tri-Planar MIL Model ────────────────────────
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

# Initialize and load all 5 fold models
models = []
for idx, ckpt_p in enumerate(CKPT_PATHS):
    print(f"Loading Fold {idx} ({BACKBONE_NAME}) from {ckpt_p.name}...")
    m = PlaneDecoupledTriPlanarMILModel(
        backbone_name=BACKBONE_NAME,
        pretrained=False,
        num_classes=12,
        classifier_type="linear"
    )
    checkpoint = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    m.load_state_dict(checkpoint["model_state_dict"])
    m = m.to(DEVICE).eval()
    if DEVICE == "cuda":
        m = m.half()
    models.append(m)
    print(f"  ✓ Fold {idx} loaded successfully onto {DEVICE} (FP16: {DEVICE == 'cuda'})!")

print(f"✅ All {len(models)} Fold models loaded and ready for 5-Fold ensembling!")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_arch))

    # Cell 3: Single I/O Pass DICOM Loader & GPU VRAM Dynamic Resize
    cell_dicom = """# ── Single I/O Pass Multi-View DICOM Loader (GPU VRAM Dynamic Resize) ────────
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
    with ThreadPoolExecutor(max_workers=4) as ex:
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
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    \"\"\"
    Single I/O Pass study loader:
    1. Reads Sagittal, Coronal, and Axial slices once from disk.
    2. Transfers 2.5D slices directly to GPU VRAM.
    3. Dynamically resizes to img_size using torch.nn.functional.interpolate (< 5 ms).
    4. Returns full 72-slice volume (1, 72, 3, img_size, img_size) ready for all 5 models.
    \"\"\"
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

        # Dynamic GPU VRAM spatial resize (< 5 ms for 24 slices)
        if t_plane.shape[-2] != img_size or t_plane.shape[-1] != img_size:
            t_plane = F.interpolate(t_plane, size=(img_size, img_size), mode="bilinear", align_corners=False)

        plane_tensors.append(t_plane)
        valid_flags.append(is_valid)

    # Concatenate all 3 planes -> (72, 3, img_size, img_size)
    full_vol = torch.cat(plane_tensors, dim=0).unsqueeze(0)  # (1, 72, 3, img_size, img_size)
    plane_ids = torch.cat([torch.full((24,), i, dtype=torch.long) for i in range(3)]).unsqueeze(0).to(device)
    mask = torch.tensor([vf for vf in valid_flags for _ in range(24)], dtype=torch.bool).unsqueeze(0).to(device)

    return full_vol, plane_ids, mask

print("✓ Single I/O Pass DICOM extraction pipeline (GPU VRAM resize) initialized.")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_dicom))

    # Cell 4: 5-Fold Ensemble Inference Loop
    cell_infer = """# ── 5-Fold Ensemble Inference Loop ──────────────────────────────────────────
df_ts = None
if TEST_SERIES_CSV.exists():
    try:
        df_ts = pd.read_csv(TEST_SERIES_CSV)
        print(f"Loaded test_series.csv manifest ({len(df_ts)} rows)")
    except Exception as e:
        print(f"Warning: could not read {TEST_SERIES_CSV}: {e}")

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

DEFAULT_PRED = np.full(12, 0.5, dtype=np.float32)
results = []
failed_count = 0
start_time = time.time()

num_models = len(models)
print(f"Executing 5-Fold Ensemble across {num_models} ConvNeXt-Small (384px) models...")

with torch.no_grad():
    for i, study_id in enumerate(study_ids):
        try:
            # Single I/O Pass from disk + GPU VRAM dynamic resize
            inp, p_ids, m = load_triplanar_study(study_id, TEST_DIR, df_series=df_ts, img_size=IMG_SIZE, device=DEVICE)
            if DEVICE == "cuda":
                inp = inp.half()

            # Multi-Fold predictions evaluated directly on in-VRAM tensor
            fold_probs = []
            for m_fold in models:
                out = m_fold(inp, plane_ids=p_ids, mask=m)
                p = torch.sigmoid(out["logits"]).squeeze(0).cpu().float().numpy()
                fold_probs.append(p)

            # 5-Fold unweighted mean probability ensemble
            ens_prob = np.mean(fold_probs, axis=0)
            results.append([study_id] + ens_prob.tolist())
        except Exception as e:
            failed_count += 1
            results.append([study_id] + DEFAULT_PRED.tolist())
            if failed_count <= 5:
                print(f"Warning: Study {study_id} failed with error: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(study_ids):
            elapsed = time.time() - start_time
            speed = (i + 1) / max(elapsed, 1e-4)
            print(f"Processed {i + 1}/{len(study_ids)} studies ({speed:.2f} std/s | elapsed: {elapsed / 60:.1f} min)")

print(f"Inference Complete! Failed count: {failed_count}/{len(study_ids)}")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_infer))

    # Cell 5: Submission Output & Assertions
    cell_save = """# ── Format Submission & Run Assertions ───────────────────────────────────────
out_df = pd.DataFrame(results, columns=[id_col] + TARGET_COLUMNS)
merged = sub_df[[id_col]].merge(out_df, on=id_col, how="left")

assert len(merged) == len(sub_df), f"Row count mismatch! {len(merged)} vs {len(sub_df)}"
for col in TARGET_COLUMNS:
    assert col in merged.columns, f"Missing required column: {col}"
    if merged[col].isnull().any():
        print(f"Filling NaN values in {col} with 0.5")
        merged[col] = merged[col].fillna(0.5)
    merged[col] = merged[col].clip(1e-4, 1.0 - 1e-4)

merged.to_csv("submission.csv", index=False)
print("=" * 60)
print(f"✓ submission.csv successfully saved ({len(merged)} rows, {len(merged.columns)} columns)")
print(f"File size: {os.path.getsize('submission.csv')} bytes")
print("First 3 rows:")
print(merged.head(3))
print("=" * 60)
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_save))

    return nb


def main():
    kernel_dir = Path("kaggle_upload/kernel")
    kernel_dir.mkdir(parents=True, exist_ok=True)

    nb = create_5fold_ensemble_notebook()
    nb_path = kernel_dir / "rsna_knee_submission.ipynb"

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Generated 5-Fold ConvNeXt-Small notebook: {nb_path} ({len(nb.cells)} cells)")

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
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote kernel-metadata.json pointing to {meta['id']}")

if __name__ == "__main__":
    main()
