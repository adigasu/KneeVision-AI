import os
import shutil
import json
from pathlib import Path
import nbformat as nbf

def main():
    # 1. Setup Staging Directories
    dataset_dir = Path("kaggle_upload/datasets/triplanar_convnext_tiny_fold0")
    kernel_dir = Path("kaggle_upload/kernels/triplanar_convnext_tiny_fold0")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # 2. Stage Weights & Dataset Metadata
    ckpt_src = Path("artifacts/experiments/phase_08_triplanar_fusion/checkpoints/triplanar_convnext_tiny_fold0_best.pth")
    ckpt_dst = dataset_dir / "triplanar_convnext_tiny_fold0_best.pth"
    if not ckpt_dst.exists() or ckpt_dst.stat().st_size != ckpt_src.stat().st_size:
        print(f"Copying {ckpt_src} -> {ckpt_dst}...")
        shutil.copy2(ckpt_src, ckpt_dst)

    dataset_meta = {
        "title": "RSNA Knee KneeVision AI Checkpoints v2",
        "id": "sukeshadigav/rsna-knee-kneevision-weights",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Tri-Planar ConvNeXt-Tiny 3-Plane Multi-View Fusion (Fold 0)",
        "description": "Trained 3-plane cross-attention checkpoint for KneeVision-AI on RSNA Knee MRI Challenge."
    }
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    # 3. Kernel Metadata
    kernel_meta = {
        "id": "sukeshadigav/kneevision-ai-rsna-submission",
        "title": "KneeVision-AI RSNA Submission",
        "code_file": "rsna_knee_submission_triplanar_convnext_tiny_fold0.ipynb",
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
        json.dump(kernel_meta, f, indent=2)

    # 4. Generate Submission Notebook
    nb = nbf.v4.new_notebook()

    # Header Markdown
    cell_md = """# KneeVision-AI — RSNA Knee Abnormality Detection
### Tri-Planar Multi-View Cross-Attention Model (Fold 0)
- **Modality:** Simultaneous 3-Plane Fusion (Sagittal + Coronal + Axial: 48 Slices/Study)
- **Architecture:** ConvNeXt-Tiny + Intra-Plane Gated Attention MIL + Multi-Head Cross-Plane Fusion
- **Resolution:** 256×256 | **Slices per Plane:** 16 (Total 48 Slices per study)
- **Offline / Standalone Inference:** Fully self-contained, crash-resilient DICOM loader with TTA
"""

    # Imports & Setup
    cell_imports = """import os, gc, sys, time, glob, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pydicom
from PIL import Image
import timm

warnings.filterwarnings("ignore")

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""

    # Path Resolution
    cell_paths = """# ── Robust Path Resolution ──────────────────────────────────────────────────
def find_comp_dir():
    candidates = [
        Path("../input/rsna-knee-mri-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-mri-abnormality-detection"),
        Path("../input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("../input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find competition directory!")

COMP_DIR = find_comp_dir()
TEST_DIR = COMP_DIR / "test"
SAMPLE_SUB = COMP_DIR / "sample_submission.csv"

print(f"Competition Dir : {COMP_DIR}")
print(f"Test Dir        : {TEST_DIR}")
print(f"Sample Sub      : {SAMPLE_SUB} (exists: {SAMPLE_SUB.exists()})")

# Checkpoint Resolution
def find_checkpoint():
    patterns = [
        "../input/**/triplanar_convnext_tiny_fold0_best.pth",
        "/kaggle/input/**/triplanar_convnext_tiny_fold0_best.pth",
        "../input/**/*triplanar*.pth",
        "/kaggle/input/**/*triplanar*.pth",
        "artifacts/**/triplanar_convnext_tiny_fold0_best.pth",
        "kaggle_upload/**/triplanar_convnext_tiny_fold0_best.pth",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError("Could not find tri-planar checkpoint!")

CKPT_FILE = find_checkpoint()
print(f"Found Checkpoint: {CKPT_FILE}")

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

TARGET_SLICES = 16
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""

    # Model Definition
    cell_model = """# ── Tri-Planar Architecture Definition ──────────────────────────────────────
class GatedAttentionMILPool(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        raw = self.attention_weights(A_V * A_U)
        if mask is not None:
            raw = raw.masked_fill(~mask.unsqueeze(-1), -1e4)
        attn = torch.softmax(raw, dim=1)
        return (attn * x).sum(dim=1), attn.squeeze(-1)


class CrossPlaneAttentionFusion(nn.Module):
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

    def forward(self, plane_tokens: torch.Tensor):
        attn_out, attn_weights = self.cross_attn(plane_tokens, plane_tokens, plane_tokens)
        x = self.norm1(plane_tokens + attn_out)
        x = self.norm2(x + self.mlp(x))
        x_trans = x.transpose(1, 2)
        fused = self.plane_pool(x_trans).squeeze(-1)
        return fused, attn_weights


class TriPlanarKneeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = False,
        num_classes: int = 12,
        mil_hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.3,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.chunk_size = chunk_size
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

    def _encode_single_plane(self, images: torch.Tensor, pooler: GatedAttentionMILPool):
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

        slice_feats = feats_flat.view(B, D, self.num_features)
        plane_embed, attn_w = pooler(slice_feats)
        return plane_embed, attn_w

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor):
        z_sag, _ = self._encode_single_plane(sag, self.sag_pool)
        z_cor, _ = self._encode_single_plane(cor, self.cor_pool)
        z_ax, _ = self._encode_single_plane(ax, self.ax_pool)

        plane_tokens = torch.stack([z_sag, z_cor, z_ax], dim=1)
        fused, _ = self.fusion(plane_tokens)
        logits = self.classifier(fused)
        return logits


# ── Load Model & Weights ─────────────────────────────────────────────────────
model = TriPlanarKneeModel(backbone_name="convnext_tiny", pretrained=False, num_classes=12)
ckpt = torch.load(CKPT_FILE, map_location="cpu")
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()
if DEVICE == "cuda":
    model = model.half()  # FP16 fast inference

print("✅ Tri-Planar Model & Checkpoint Successfully Loaded!")
if "best_metrics" in ckpt:
    print(f"  Trained Metrics: {ckpt['best_metrics']}")
"""

    # DICOM Multi-View Loader
    cell_dicom = """# ── DICOM Multi-View Extraction Utilities ────────────────────────────────────
SAGITTAL_KW = ["SAG", "SAGITTAL", "PD_SAG", "FS_SAG", "STIR_SAG", "COR_SAG", "PD SAG", "FS SAG", "SAGITAL"]
CORONAL_KW  = ["COR", "CORONAL", "PD_COR", "FS_COR", "T2_COR", "COR_FS", "PD COR", "FS COR"]
AXIAL_KW    = ["AX", "AXIAL", "PD_AX", "FS_AX", "T2_AX", "AX_FS", "PD AX", "FS AX", "TRA", "TRANS"]


def _read_slice(path: str):
    """Read DICOM pixel array and z-position."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=False)
    arr = ds.pixel_array.astype(np.float32)
    try:
        z = float(ds.ImagePositionPatient[2])
    except Exception:
        z = 0.0
    return z, arr


def find_plane_series(study_dir: Path, keywords: list):
    """Find the best matching series directory for a given anatomical plane."""
    series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    if not series_dirs:
        return sorted(study_dir.glob("*.dcm"))

    best, best_score = None, -1
    for sd in series_dirs:
        dcms = list(sd.glob("*.dcm"))
        if not dcms:
            continue
        name = sd.name.upper()
        kw_score = sum(2 for k in keywords if k in name)
        score = kw_score * 1000 + len(dcms)
        if score > best_score:
            best_score, best = score, sd

    if best:
        return sorted(best.glob("*.dcm"))
    # Fallback to the largest series directory
    largest = max(series_dirs, key=lambda d: len(list(d.glob("*.dcm"))), default=None)
    return sorted(largest.glob("*.dcm")) if largest else []


def load_plane_volume(dcm_paths, target_slices=TARGET_SLICES, img_size=IMG_SIZE):
    """
    Loads DICOM files, sorts by z-slice position, resamples to target_slices,
    normalizes, and constructs 2.5D RGB-channel tensors: (D, 3, H, W).
    """
    if not dcm_paths:
        return np.zeros((target_slices, 3, img_size, img_size), dtype=np.float32)

    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_read_slice, str(p)): i for i, p in enumerate(dcm_paths)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    ordered = [results[i] for i in range(len(results))]
    ordered.sort(key=lambda t: t[0])
    slices = [t[1] for t in ordered]

    volume = np.stack(slices, axis=0)  # (D, H, W)

    D = volume.shape[0]
    if D != target_slices:
        idx = np.linspace(0, D - 1, target_slices, dtype=int)
        volume = volume[idx]

    # Normalize per volume
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    # Spatial resize
    D2, H, W = volume.shape
    if H != img_size or W != img_size:
        resized = []
        for sl in volume:
            pil = Image.fromarray((sl * 255).astype(np.uint8))
            pil = pil.resize((img_size, img_size), Image.BILINEAR)
            resized.append(np.array(pil, dtype=np.float32) / 255.0)
        volume = np.stack(resized, axis=0)

    # Build 2.5D stacks [z-1, z, z+1]
    D2 = volume.shape[0]
    out = np.zeros((D2, 3, img_size, img_size), dtype=np.float32)
    for i in range(D2):
        out[i, 0] = volume[max(0, i - 1)]
        out[i, 1] = volume[i]
        out[i, 2] = volume[min(D2 - 1, i + 1)]

    return out

print("✅ DICOM Multi-View extraction pipeline initialized.")
"""

    # Inference Loop
    cell_infer = """# ── Inference Execution Loop ────────────────────────────────────────────────
if SAMPLE_SUB.exists():
    sub_df = pd.read_csv(SAMPLE_SUB)
else:
    sub_df = pd.DataFrame(columns=["StudyInstanceUID"] + TARGET_COLUMNS)

id_col = "StudyInstanceUID" if "StudyInstanceUID" in sub_df.columns else sub_df.columns[0]
study_ids = sub_df[id_col].tolist()
print(f"ID column: {id_col} | Studies to predict: {len(study_ids)}")

DEFAULT_PRED = np.full(12, 0.5, dtype=np.float32)
results = []
failed = 0
t_start = time.time()

for idx, study_id in enumerate(study_ids):
    study_dir = TEST_DIR / str(study_id)
    if not study_dir.exists():
        # Fallback if unnested
        study_dir = TEST_DIR

    try:
        # 1. Locate Sagittal, Coronal, and Axial series
        sag_dcms = find_plane_series(study_dir, SAGITTAL_KW)
        cor_dcms = find_plane_series(study_dir, CORONAL_KW)
        ax_dcms  = find_plane_series(study_dir, AXIAL_KW)

        # 2. Build 2.5D Volumetric Tensors for each plane
        sag_vol = load_plane_volume(sag_dcms, target_slices=TARGET_SLICES, img_size=IMG_SIZE)
        cor_vol = load_plane_volume(cor_dcms, target_slices=TARGET_SLICES, img_size=IMG_SIZE)
        ax_vol  = load_plane_volume(ax_dcms,  target_slices=TARGET_SLICES, img_size=IMG_SIZE)

        # 3. Convert to Torch Tensors
        sag_t = torch.from_numpy(sag_vol).unsqueeze(0).to(DEVICE)
        cor_t = torch.from_numpy(cor_vol).unsqueeze(0).to(DEVICE)
        ax_t  = torch.from_numpy(ax_vol).unsqueeze(0).to(DEVICE)

        if DEVICE == "cuda":
            sag_t = sag_t.half()
            cor_t = cor_t.half()
            ax_t  = ax_t.half()

        # 4. Predict with Test-Time Augmentation (Horizontal Flip)
        with torch.no_grad():
            # Standard pass
            logits_orig = model(sag_t, cor_t, ax_t)
            probs_orig  = torch.sigmoid(logits_orig)

            # Horizontal flip TTA pass
            sag_flip = torch.flip(sag_t, dims=[-1])
            cor_flip = torch.flip(cor_t, dims=[-1])
            ax_flip  = torch.flip(ax_t, dims=[-1])
            logits_flip = model(sag_flip, cor_flip, ax_flip)
            probs_flip  = torch.sigmoid(logits_flip)

            probs = (0.5 * (probs_orig + probs_flip)).squeeze(0).cpu().numpy()

        results.append([study_id] + probs.tolist())

    except Exception as e:
        failed += 1
        results.append([study_id] + DEFAULT_PRED.tolist())
        if failed <= 5:
            print(f"Warning: Study {study_id} failed with error: {e}")

    if (idx + 1) % 50 == 0 or (idx + 1) == len(study_ids):
        elapsed = time.time() - t_start
        rate = (idx + 1) / elapsed
        print(f"[{idx+1:04d}/{len(study_ids):04d}] Processed at {rate:.2f} studies/sec (Failed: {failed})")

print(f"\nInference Completed in {time.time() - t_start:.2f}s! (Total Failed: {failed})")
"""

    # Submission Generation
    cell_save = """# ── Generate & Verify submission.csv ────────────────────────────────────────
out_df = pd.DataFrame(results, columns=[id_col] + TARGET_COLUMNS)
merged = sub_df[[id_col]].merge(out_df, on=id_col, how="left")

for col in TARGET_COLUMNS:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0.5)
    else:
        merged[col] = 0.5

merged.to_csv("submission.csv", index=False)
print(f"✅ submission.csv successfully saved — Shape: {merged.shape}")
print("\nPredictions Summary Preview:")
print(merged[TARGET_COLUMNS].describe().round(4))
print("\nFirst 5 Predictions:")
print(merged.head())
"""

    nb.cells = [
        nbf.v4.new_markdown_cell(cell_md),
        nbf.v4.new_code_cell(cell_imports),
        nbf.v4.new_code_cell(cell_paths),
        nbf.v4.new_code_cell(cell_model),
        nbf.v4.new_code_cell(cell_dicom),
        nbf.v4.new_code_cell(cell_infer),
        nbf.v4.new_code_cell(cell_save),
    ]

    nb_path = kernel_dir / "rsna_knee_submission_triplanar_convnext_tiny_fold0.ipynb"
    with open(nb_path, "w") as f:
        nbf.write(nb, f)

    print(f"✅ Generated Tri-Planar submission notebook: {nb_path}")
    print(f"✅ Staged Dataset in: {dataset_dir}")
    print(f"✅ Staged Kernel in: {kernel_dir}")

if __name__ == "__main__":
    main()
