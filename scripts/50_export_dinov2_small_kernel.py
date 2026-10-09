"""
RSNA Knee Abnormality Detection - DINOv2-Small Standalone Submission Exporter.
Exports a self-contained, fully air-gapped Kaggle notebook for:
Tri-Planar DINOv2-Small ViT (280px, Fold 0) -> kaggle_upload/kernels/phase_13_triplanar_dinov2_small
"""

import os
import json
import shutil
from pathlib import Path
import nbformat as nbf

def create_dinov2_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()

    # Required Kaggle Papermill metadata
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
## Individual Standalone Model Submission: Tri-Planar DINOv2-Small ViT (280px)
* **Architecture:** Tri-Planar 72-Slice Label-Specific Gated Attention MIL Model
* **Backbone:** `vit_small_patch14_dinov2.lvd142m` (Resolution: 280×280, 20×20 patches)
* **Checkpoint:** `phase13_triplanar_dinov2_small_f0_72sl_best.pth`
* **Local Validation Metrics:** Val Macro AUC: **0.8617** | Gold Consensus AUC: **0.8916**
* **Specialized Pathology Strengths:**
  - **Bone Contusion:** 0.882 AUC ⭐
  - **ACL Tear:** 0.899 AUC ⭐
  - **Joint Effusion:** 0.876 AUC ⭐
  - **Medial Meniscus Tear:** 0.916 AUC
* **Runtime Efficiency:** ~70 ms per study in FP16 on Tesla T4 (~3.5 min total for 3,000 test cases).
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
IMG_SIZE = 280
BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"
CKPT_NAME = "phase13_triplanar_dinov2_small_f0_72sl_best.pth"

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
CKPT_PATH = find_checkpoint(CKPT_NAME)

print(f"Comp Dir        : {COMP_DIR}")
print(f"Test Dir        : {TEST_DIR} (exists: {TEST_DIR.exists()})")
print(f"Sample Sub      : {SAMPLE_SUB} (exists: {SAMPLE_SUB.exists()})")
print(f"Test Series CSV : {TEST_SERIES_CSV} (exists: {TEST_SERIES_CSV.exists()})")
print(f"Checkpoint Path : {CKPT_PATH} (exists: {CKPT_PATH.exists()})")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_env))

    # Cell 2: Architecture Definition
    cell_arch = """# ── Tri-Planar DINOv2-Small Label-Specific MIL Model ─────────────────────────
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
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        raw_attn = self.attention_weights(A_V * A_U).transpose(1, 2)  # (B, K, D)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            raw_attn = raw_attn.masked_fill(~mask_expanded, -1e4)

        attn = torch.softmax(raw_attn, dim=-1)  # (B, K, D)
        h = torch.bmm(attn, x)                  # (B, K, F)
        h_norm = self.dropout(self.norm(h))     # (B, K, F)

        logits = torch.einsum('bkf,kf->bk', h_norm, self.classifier_w) + self.classifier_b
        return logits, attn, h


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


# Initialize and load model
print(f"Loading {BACKBONE_NAME} model from {CKPT_PATH.name}...")
model = TriPlanarLabelSpecificMILModel(
    backbone_name=BACKBONE_NAME,
    pretrained=False,
    num_classes=12,
    chunk_size=32,
)
checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE).eval()
if DEVICE == "cuda":
    model = model.half()
print(f"✅ {BACKBONE_NAME} successfully loaded onto {DEVICE}!")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_arch))

    # Cell 3: DICOM Loader
    cell_dicom = """# ── Robust Multi-View DICOM Loader Pipeline ──────────────────────────────────
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

def load_plane_slices(dcm_paths: List[Path], target_slices: int = 24, img_size: int = IMG_SIZE) -> Tuple[np.ndarray, bool]:
    if not dcm_paths:
        return np.zeros((target_slices, 3, img_size, img_size), dtype=np.float32), False

    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_read_slice, str(p)): i for i, p in enumerate(dcm_paths)}
        for fut in as_completed(futs):
            try:
                z, arr = fut.result()
                results[futs[fut]] = (z, arr)
            except Exception:
                pass

    if not results:
        return np.zeros((target_slices, 3, img_size, img_size), dtype=np.float32), False

    ordered = [results[i] for i in range(len(results)) if i in results]
    ordered.sort(key=lambda t: t[0])
    slices = [t[1] for t in ordered]

    volume = np.stack(slices, axis=0)

    D = volume.shape[0]
    if D != target_slices:
        idx = np.linspace(0, D - 1, target_slices).round().astype(int)
        volume = volume[idx]

    # Native min-max intensity normalization to [0.0, 1.0] exactly matching training pipeline
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    resized = []
    for sl in volume:
        pil = Image.fromarray((sl * 255.0).clip(0, 255).astype(np.uint8))
        if pil.size != (img_size, img_size):
            pil = pil.resize((img_size, img_size), Image.BILINEAR)
        resized.append(np.array(pil, dtype=np.float32) / 255.0)
    volume = np.stack(resized, axis=0)

    # Build 2.5D channel stacks (slice-1, slice, slice+1) in [0.0, 1.0]
    out = np.zeros((target_slices, 3, img_size, img_size), dtype=np.float32)
    for i in range(target_slices):
        out[i, 0] = volume[max(0, i - 1)]
        out[i, 1] = volume[i]
        out[i, 2] = volume[min(target_slices - 1, i + 1)]

    return out, True

def load_triplanar_study(study_id: str, test_dir: Path, df_series: Optional[pd.DataFrame] = None, img_size: int = IMG_SIZE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    study_path = test_dir / str(study_id)
    if not study_path.exists():
        for candidate in test_dir.rglob(str(study_id)):
            if candidate.is_dir():
                study_path = candidate
                break

    plane_keywords = {
        "Sagittal": SAGITTAL_KW,
        "Coronal":  CORONAL_KW,
        "Axial":    AXIAL_KW,
    }

    plane_vols = []
    valid_flags = []

    for plane in ["Sagittal", "Coronal", "Axial"]:
        dcms = []

        # 1. Use metadata if available
        if df_series is not None and "StudyInstanceUID" in df_series.columns:
            matches = df_series[(df_series["StudyInstanceUID"].astype(str) == str(study_id)) & (df_series.get("Plane", "") == plane)]
            for _, row in matches.iterrows():
                sid = str(row.get("SeriesInstanceUID", ""))
                for cd in [study_path / sid, test_dir / sid]:
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

        vol, is_valid = load_plane_slices(dcms, target_slices=24, img_size=img_size)
        plane_vols.append(vol)
        valid_flags.append(is_valid)

    full_vol = np.concatenate(plane_vols, axis=0)  # (72, 3, img_size, img_size)
    vol_tensor = torch.from_numpy(full_vol).unsqueeze(0)  # (1, 72, 3, img_size, img_size)
    plane_ids = torch.cat([torch.full((24,), i, dtype=torch.long) for i in range(3)]).unsqueeze(0)
    mask = torch.tensor([vf for vf in valid_flags for _ in range(24)], dtype=torch.bool).unsqueeze(0)

    return vol_tensor, plane_ids, mask

print("DICOM Multi-View extraction pipeline initialized.")
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_dicom))

    # Cell 4: Inference Execution Loop
    cell_infer = """# ── Inference Execution Loop ────────────────────────────────────────────────
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

with torch.no_grad():
    for i, study_id in enumerate(study_ids):
        try:
            vol_tensor, plane_ids, mask = load_triplanar_study(study_id, TEST_DIR, df_series=df_ts, img_size=IMG_SIZE)
            inp = vol_tensor.to(DEVICE)
            if DEVICE == "cuda":
                inp = inp.half()
            p_ids = plane_ids.to(DEVICE)
            m = mask.to(DEVICE) if mask is not None else None

            out = model(inp, plane_ids=p_ids, mask=m)
            probs = torch.sigmoid(out["logits"]).squeeze(0).cpu().float().numpy()
            results.append([study_id] + probs.tolist())
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
print(f"Saved submission.csv with shape {merged.shape} successfully!")

print("Prediction Summary Statistics:")
print(merged[TARGET_COLUMNS].describe().round(4))
print("First 5 rows of submission:")
print(merged.head(5).to_string(index=False))
"""
    nb.cells.append(nbf.v4.new_code_cell(cell_save))

    return nb

def main():
    root = Path(".")
    staging_dinov2 = root / "kaggle_upload" / "kernels" / "phase_13_triplanar_dinov2_small"
    staging_dinov2.mkdir(parents=True, exist_ok=True)
    nb_dinov2 = create_dinov2_notebook()
    nb_path = staging_dinov2 / "rsna_knee_dinov2_small_submission.ipynb"
    with open(nb_path, "w") as f:
        nbf.write(nb_dinov2, f)

    metadata = {
        "id": "sukeshadigav/kneevision-ai-dinov2-small-submission",
        "title": "KneeVision-AI DINOv2-Small Submission",
        "code_file": "rsna_knee_dinov2_small_submission.ipynb",
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
    with open(staging_dinov2 / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Successfully exported Tri-Planar DINOv2-Small (280px) to {staging_dinov2}")
    print(f"Notebook: {nb_path} ({len(nb_dinov2.cells)} cells)")

if __name__ == "__main__":
    main()
