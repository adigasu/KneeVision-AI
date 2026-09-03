"""
Generate a complete suite of high-resolution, publication-quality visual EDA plots
for the RSNA Knee Abnormality Detection (KneeVision-AI) project.
"""

import os
import glob
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.metrics import accuracy_score, f1_score
import torch

from src.data.dicom_reader import read_dicom_slice, load_series_volume

import torch.nn as nn
class GatedAttentionMILPool(nn.Module):
    def __init__(self, in_features=128, hidden_dim=64, num_classes=12):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(in_features, num_classes)
    def forward(self, x):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        M = torch.sum(A * x, dim=1)
        logits = self.classifier(M)
        return logits, A


# Configuration
DATA_DIR = '../Datasets/rsna-knee-abnormality-detection'
TRAIN_SERIES_DIR = f'{DATA_DIR}/train_series'
FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("Loading metadata...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
train_series_df = pd.read_csv(f"{DATA_DIR}/train_series.csv")

labeled_mask = train_df[TARGET_COLUMNS].notnull().any(axis=1)
df_labeled = train_df[labeled_mask].copy()

# ==========================================
# Plot 1: Target Prevalences
# ==========================================
print("1. Generating Target Prevalences plot...")
stats = []
for col in TARGET_COLUMNS:
    pos = int((df_labeled[col] == 1.0).sum())
    neg = int((df_labeled[col] == 0.0).sum())
    total = pos + neg
    prev = (pos / total * 100) if total > 0 else 0
    stats.append({
        'Abnormality': col,
        'Positive': pos,
        'Negative': neg,
        'Prevalence (%)': prev
    })

stats_df = pd.DataFrame(stats).sort_values(by='Prevalence (%)', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("mako", len(stats_df))
bars = ax.barh(stats_df['Abnormality'], stats_df['Prevalence (%)'], color=colors, height=0.65)

ax.set_title('Clinical Knee Abnormality Prevalence (%) in Ground-Truth Cohort (N=58)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Prevalence (%)', fontsize=11, fontweight='bold')
ax.set_xlim(0, 75)

for bar, (_, row) in zip(bars, stats_df.iterrows()):
    w = bar.get_width()
    ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}% ({int(row['Positive'])}/{int(row['Positive']+row['Negative'])})",
            va='center', ha='left', fontsize=9, fontweight='bold', color='#222222')

plt.tight_layout()
p1_path = os.path.join(FIGURES_DIR, '01_target_prevalences.png')
plt.savefig(p1_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p1_path}")

# ==========================================
# Plot 2: Target Correlation Matrix
# ==========================================
print("2. Generating Target Correlation Matrix...")
corr_matrix = df_labeled[TARGET_COLUMNS].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    vmin=-0.6,
    vmax=0.6,
    mask=mask,
    square=True,
    linewidths=0.7,
    cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient (r)"},
    ax=ax
)
ax.set_title('Pathology Target Multi-Label Correlation Heatmap', fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
p2_path = os.path.join(FIGURES_DIR, '02_target_correlations.png')
plt.savefig(p2_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p2_path}")

# ==========================================
# Plot 3: Series & Anatomical Planes Distribution
# ==========================================
print("3. Generating Series & Planes Breakdown...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

series_per_study = train_series_df.groupby('StudyInstanceUID').size()
sns.countplot(x=series_per_study.values, palette='Blues_r', ax=ax1)
ax1.set_title('Series Count per Knee Study Distribution (Min:3, Med:5, Max:14)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Series per Study')
ax1.set_ylabel('Number of Studies')
for p in ax1.patches:
    h = p.get_height()
    if h > 0:
        ax1.annotate(f"{int(h)}", (p.get_x() + p.get_width()/2, h + 25),
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

plane_counts = train_series_df.groupby(['Anatomical_Plane', 'Fluid_Sensitive']).size().unstack(fill_value=0)
plane_counts.rename(columns={0: 'Non-Fluid/T1', 1: 'Fluid-Sensitive (T2/PDFS)'}).plot(
    kind='bar', stacked=True, color=['#7fcdbb', '#2c7fb8'], ax=ax2
)
ax2.set_title('Series by Anatomical Plane & Fluid Sensitivity', fontsize=12, fontweight='bold')
ax2.set_xlabel('Anatomical Plane')
ax2.set_ylabel('Total Series Count')
ax2.tick_params(axis='x', rotation=0)
ax2.legend(frameon=True)

plt.tight_layout()
p3_path = os.path.join(FIGURES_DIR, '03_series_and_planes_distribution.png')
plt.savefig(p3_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p3_path}")

# ==========================================
# Plot 4: DICOM Geometry & Volumetric Depth
# ==========================================
print("4. Extracting sample DICOM header metrics...")
study_folders = [f for f in os.listdir(TRAIN_SERIES_DIR) if os.path.isdir(os.path.join(TRAIN_SERIES_DIR, f))]

meta_records = []
for study in study_folders[:200]:
    study_path = os.path.join(TRAIN_SERIES_DIR, study)
    for series in os.listdir(study_path):
        series_path = os.path.join(study_path, series)
        if not os.path.isdir(series_path):
            continue
        dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        if not dcm_files:
            continue
        try:
            dcm = pydicom.dcmread(os.path.join(series_path, dcm_files[0]), stop_before_pixels=True)
            meta_records.append({
                'NumSlices': len(dcm_files),
                'Rows': dcm.Rows,
                'Columns': dcm.Columns,
                'PixelSpacing': float(dcm.PixelSpacing[0]) if hasattr(dcm, 'PixelSpacing') else np.nan,
                'SliceThickness': float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else np.nan,
            })
        except Exception:
            pass

df_meta = pd.DataFrame(meta_records)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Depth (Slice Counts)
sns.histplot(df_meta['NumSlices'], bins=35, kde=True, color='#2b5c8f', ax=ax1)
med_d = df_meta['NumSlices'].median()
ax1.axvline(med_d, color='crimson', linestyle='--', linewidth=2, label=f"Median Depth: {med_d:.0f} Slices")
ax1.set_title('Variable Series Depth / Slice Count (11 to 320 slices)', fontweight='bold')
ax1.set_xlabel('Number of Slices (D)')
ax1.legend()

# 2. Slice Thickness
st_counts = df_meta['SliceThickness'].value_counts().head(5)
sns.barplot(x=st_counts.index.astype(str) + " mm", y=st_counts.values, palette='crest', ax=ax2)
ax2.set_title('Slice Thickness Distribution', fontweight='bold')
ax2.set_xlabel('Slice Thickness')
ax2.set_ylabel('Series Count')
for p in ax2.patches:
    ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height() + 10),
                 ha='center', va='bottom', fontsize=9)

# 3. Resolutions
shape_series = df_meta['Rows'].astype(str) + "x" + df_meta['Columns'].astype(str)
top_shapes = shape_series.value_counts().head(6)
sns.barplot(x=top_shapes.index, y=top_shapes.values, palette='crest', ax=ax3)
ax3.set_title('Top Image Dimensions (H x W)', fontweight='bold')
ax3.set_xlabel('In-Plane Resolution')
ax3.set_ylabel('Series Count')
for p in ax3.patches:
    ax3.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height() + 8),
                 ha='center', va='bottom', fontsize=9)

# 4. In-Plane Spacing
sns.histplot(df_meta['PixelSpacing'].dropna(), bins=25, kde=True, color='#8856a7', ax=ax4)
ax4.set_title('In-Plane Pixel Spacing (mm) Distribution', fontweight='bold')
ax4.set_xlabel('Pixel Spacing (mm)')

plt.tight_layout()
p4_path = os.path.join(FIGURES_DIR, '04_dicom_geometry_and_depth.png')
plt.savefig(p4_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p4_path}")

# ==========================================
# Plot 5: Multi-Slice Volumetric Montage
# ==========================================
print("5. Generating Volumetric Montage...")
sample_study = study_folders[0]
sample_study_path = os.path.join(TRAIN_SERIES_DIR, sample_study)
sample_series = os.listdir(sample_study_path)[0]
sample_series_path = os.path.join(sample_study_path, sample_series)

vol, metas = load_series_volume(sample_series_path, target_slices=24, normalize='minmax')

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
slice_indices = np.linspace(2, vol.shape[0] - 3, 12).astype(int)

for idx, ax in zip(slice_indices, axes.flatten()):
    ax.imshow(vol[idx], cmap='bone')
    ax.set_title(f"Slice #{idx+1}/{vol.shape[0]}", fontsize=10, fontweight='bold')
    ax.axis('off')

plt.suptitle(f"Spatially Sorted 3D Knee Volume Slices ({sample_series[:20]}...)", fontsize=14, fontweight='bold')
plt.tight_layout()
p5_path = os.path.join(FIGURES_DIR, '05_multi_slice_montage.png')
plt.savefig(p5_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p5_path}")

# ==========================================
# Plot 6: Tri-Planar Study Views (Sagittal + Coronal + Axial)
# ==========================================
print("6. Generating Tri-Planar Study Views...")
study_series_df = train_series_df[train_series_df['StudyInstanceUID'] == sample_study]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
planes = ['Sagittal', 'Coronal', 'Axial']

for i, plane in enumerate(planes):
    matching = study_series_df[study_series_df['Anatomical_Plane'] == plane]
    if len(matching) > 0:
        ser_uid = matching.iloc[0]['SeriesInstanceUID']
        ser_dir = os.path.join(TRAIN_SERIES_DIR, sample_study, ser_uid)
        if os.path.exists(ser_dir):
            p_vol, _ = load_series_volume(ser_dir, target_slices=32, normalize='minmax')
            mid_idx = p_vol.shape[0] // 2
            axes[i].imshow(p_vol[mid_idx], cmap='bone')
            axes[i].set_title(f"{plane} Plane (Mid-Slice {mid_idx})", fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.suptitle(f"Tri-Planar Knee Views for Study: {sample_study[:22]}...", fontsize=14, fontweight='bold')
plt.tight_layout()
p6_path = os.path.join(FIGURES_DIR, '06_triplanar_views.png')
plt.savefig(p6_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p6_path}")

# ==========================================
# Plot 7: NLP Report Analysis & Extraction Benchmark
# ==========================================
print("7. Generating NLP Report plots...")
def extract_weak_labels(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {col: 0.0 for col in TARGET_COLUMNS}
    t = text.lower()
    labels = {}
    
    # ACL
    acl_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|avulsi[oó]n|interrupci[oó]n).*?(cruzado anterior|lca)', t) or
                  re.search(r'(lca|cruzado anterior).*?(rotura|desgarro|lesi[oó]n)', t))
    acl_neg = bool(re.search(r'(cruzado anterior|lca).*?(normal|intacto|conservad[oa]|sin signos)', t))
    labels['ACL'] = 1.0 if (acl_pos and not acl_neg) else 0.0
    
    # MCL
    mcl_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|esguince).*?(colateral medial|colateral interno|lcm|lci)', t))
    mcl_neg = bool(re.search(r'(colateral medial|colateral interno|lcm).*?(normal|intacto)', t))
    labels['MCL'] = 1.0 if (mcl_pos and not mcl_neg) else 0.0
    
    # Medial Meniscus
    mm_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|fisura|amputaci[oó]n).*?(menisco medial|menisco interno)', t) or
                 re.search(r'(menisco medial|menisco interno).*?(rotura|desgarro|lesi[oó]n)', t))
    mm_neg = bool(re.search(r'(menisco medial|menisco interno).*?(conservad[oa]|normal|sin signos)', t))
    labels['Medial Meniscus'] = 1.0 if (mm_pos and not mm_neg) else 0.0

    # Lateral Meniscus
    lm_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|fisura).*?(menisco lateral|menisco externo)', t) or
                 re.search(r'(menisco lateral|menisco externo).*?(rotura|desgarro)', t))
    lm_neg = bool(re.search(r'(menisco lateral|menisco externo).*?(conservad[oa]|normal)', t))
    labels['Lateral Meniscus'] = 1.0 if (lm_pos and not lm_neg) else 0.0

    labels['Medial OA'] = 1.0 if re.search(r'(artrosis|condropat[ií]a).*?(medial|interno)', t) else 0.0
    labels['Lateral OA'] = 1.0 if re.search(r'(artrosis|condropat[ií]a).*?(lateral|externo)', t) else 0.0
    labels['PF OA'] = 1.0 if re.search(r'(artrosis|condromalacia).*?(patelar|rotulian[oa]|tr[oó]clea|femoropatelar)', t) else 0.0
    
    eff_pos = bool(re.search(r'(derrame|l[ií]quido intraarticular|hidrartros)', t))
    eff_neg = bool(re.search(r'(sin derrame|derrame ausente)', t))
    labels['Effusion'] = 1.0 if (eff_pos and not eff_neg) else 0.0
    
    labels['Synovitis'] = 1.0 if re.search(r'(sinovitis|engrosamiento sinovial)', t) else 0.0
    labels["Baker's"] = 1.0 if re.search(r'(quiste de baker|quiste popl[ií]teo|baker)', t) else 0.0
    labels['Contusion'] = 1.0 if re.search(r'(edema [oó]seo|contusi[oó]n [oó]sea|bruise)', t) else 0.0
    labels['Fracture'] = 1.0 if re.search(r'(fractura|trazo de fractura|avulsi[oó]n)', t) else 0.0
    return labels

extracted = [extract_weak_labels(r) for r in df_labeled['Report']]
df_ext = pd.DataFrame(extracted, index=df_labeled.index)

bench_rows = []
for col in TARGET_COLUMNS:
    y_true = df_labeled[col].values
    y_pred = df_ext[col].values
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bench_rows.append({'Abnormality': col, 'Accuracy': acc, 'F1': f1})

df_bench = pd.DataFrame(bench_rows).sort_values(by='Accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("viridis", len(df_bench))
bars = ax.barh(df_bench['Abnormality'], df_bench['Accuracy'], color=colors, height=0.65)
ax.set_title('Spanish Radiology Report NLP Weak Extractor Accuracy vs Ground Truth (N=58)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_xlim(0, 110)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
            va='center', ha='left', fontsize=9, fontweight='bold', color='#222222')

plt.tight_layout()
p7_path = os.path.join(FIGURES_DIR, '07_nlp_report_metrics.png')
plt.savefig(p7_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p7_path}")

# ==========================================
# Plot 8: Gated Attention MIL Pooling Simulation
# ==========================================
print("8. Generating MIL Attention Pooling plot...")
feat_dim = 128
num_slices = 24
mil_pool = GatedAttentionMILPool(in_features=feat_dim, hidden_dim=64, num_classes=12)

# Simulate slice embeddings
features = torch.randn(1, num_slices, feat_dim)
# Add pathology peak at slice 14 & 15
features[0, 13] += 3.2
features[0, 14] += 4.5
features[0, 15] += 2.8

_, attn_weights = mil_pool(features)
attn = attn_weights.squeeze().detach().numpy()

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(range(1, num_slices + 1), attn, marker='o', color='#d95f02', linewidth=2.5, label='Attention Weight $a_i$')
ax.fill_between(range(1, num_slices + 1), attn, alpha=0.35, color='#d95f02')
ax.axvspan(13.5, 16.5, color='crimson', alpha=0.15, label='Pathology ROI Slices (Tear/Lesion)')
ax.set_title('Gated Attention MIL Pooling Weights across 24 Slices (Pathology Localization)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Spatial Slice Index ($z$)', fontsize=11, fontweight='bold')
ax.set_ylabel('Attention Weight (Softmax Probability)', fontsize=11, fontweight='bold')
ax.set_xticks(range(1, num_slices + 1))
ax.legend(frameon=True, loc='upper right')

plt.tight_layout()
p8_path = os.path.join(FIGURES_DIR, '08_mil_attention_simulation.png')
plt.savefig(p8_path, bbox_inches='tight')
plt.close()
print(f"   Saved {p8_path}")

print(f"\n✓ Successfully generated all 8 publication-quality EDA figures in: {FIGURES_DIR}/")
