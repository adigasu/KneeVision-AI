"""
Script to generate the complete suite of 4 EDA & Data Analysis Jupyter Notebooks
for RSNA Knee Abnormality Detection (KneeVision-AI).
"""

import os
import nbformat as nbf

DATA_RESOLVER = """import os
import sys

# Ensure repository root is in python path
for root_cand in ['.', '..', os.path.abspath(os.path.join(os.getcwd(), '..'))]:
    if os.path.exists(os.path.join(root_cand, 'src')):
        if os.path.abspath(root_cand) not in sys.path:
            sys.path.insert(0, os.path.abspath(root_cand))
        break

# Resolve DATA_DIR whether running from workspace root or notebooks/
DATA_DIR = None
for cand in [
    '../Datasets/rsna-knee-abnormality-detection',
    '../../Datasets/rsna-knee-abnormality-detection',
    os.path.expanduser('~/net/Datasets/rsna-knee-abnormality-detection'),
    '/home/AQ44130/net/Datasets/rsna-knee-abnormality-detection'
]:
    if os.path.exists(cand):
        DATA_DIR = os.path.abspath(cand)
        break

print(f"✓ Resolved DATA_DIR: {DATA_DIR}")
"""

def build_notebook_01():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 01: Dataset Overview & Target Labels EDA
## RSNA Knee Abnormality Detection (KneeVision-AI)

This notebook provides exploratory data analysis (EDA) of the **RSNA Knee Abnormality Detection** challenge tabular metadata:
1. **Study-level Metadata (`train.csv`, `test.csv`)**: Exploring patient studies and target labels.
2. **Label Distributions & Class Imbalance**: Evaluating positive prevalence across the 12 clinical knee abnormality targets.
3. **Multi-label Co-occurrence**: Correlation heatmaps and co-occurring pathology patterns.
4. **Series-level Metadata (`train_series.csv`)**: Analyzing series counts per study, anatomical planes (`Sagittal`, `Coronal`, `Axial`), and MRI contrasts (`Fluid_Sensitive`, `Fat_Suppression`).
5. **Leak-free 5-Fold Multilabel Stratified Split**: Generating and validating cross-validation folds.
"""))

    cells.append(nbf.v4.new_code_cell(DATA_RESOLVER + """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Load Tabular Metadata"""))

    cells.append(nbf.v4.new_code_cell("""train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
train_series_df = pd.read_csv(f"{DATA_DIR}/train_series.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
test_series_df = pd.read_csv(f"{DATA_DIR}/test_series.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"Train Studies: {len(train_df):,}")
print(f"Train Series:  {len(train_series_df):,}")
print(f"Test Studies:  {len(test_df):,}")
print(f"Test Series:   {len(test_series_df):,}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Abnormality Targets & Label Distribution Analysis"""))

    cells.append(nbf.v4.new_code_cell("""TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Separate labeled vs unlabeled (weak report only) studies
labeled_mask = train_df[TARGET_COLUMNS].notnull().any(axis=1)
df_labeled = train_df[labeled_mask].copy()
df_unlabeled = train_df[~labeled_mask].copy()

print(f"Studies with Ground Truth Binary Labels: {len(df_labeled):,} ({len(df_labeled)/len(train_df)*100:.1f}%)")
print(f"Studies with Weak Radiology Reports only:  {len(df_unlabeled):,} ({len(df_unlabeled)/len(train_df)*100:.1f}%)")
"""))

    cells.append(nbf.v4.new_code_cell("""# Calculate target frequencies and prevalence in labeled cohort
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
        'Total Labeled': total,
        'Prevalence (%)': round(prev, 1)
    })

stats_df = pd.DataFrame(stats).sort_values(by='Prevalence (%)', ascending=False)
display(stats_df)

# Plot prevalence bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("mako", len(stats_df))
bars = ax.barh(stats_df['Abnormality'], stats_df['Prevalence (%)'], color=colors, height=0.65)
ax.set_title('Pathology Target Prevalence (%) in Ground Truth Labeled Cohort (N=58)', fontsize=13, fontweight='bold')
ax.set_xlabel('Prevalence (%)')
for bar, (_, row) in zip(bars, stats_df.iterrows()):
    w = bar.get_width()
    ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}% ({int(row['Positive'])}/{int(row['Total Labeled'])})",
            ha='left', va='center', fontsize=9, color='black', fontweight='bold')
plt.xlim(0, 75)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Pathology Co-occurrence & Correlation Matrix"""))

    cells.append(nbf.v4.new_code_cell("""# Compute multi-label correlation matrix
corr_matrix = df_labeled[TARGET_COLUMNS].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.6, vmax=0.6,
            mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('Pathology Target Correlation Matrix (Ground Truth Subset)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Series-Level Breakdown (Anatomical Planes & Contrasts)"""))

    cells.append(nbf.v4.new_code_cell("""# Series per study distribution
series_per_study = train_series_df.groupby('StudyInstanceUID').size()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

# Plot 1: Series Count Histogram
sns.countplot(x=series_per_study.values, hue=series_per_study.values, palette='Blues_r', legend=False, ax=ax1)
ax1.set_title('Series Count Distribution per Knee Study', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Series in Study')
ax1.set_ylabel('Number of Studies')
for p in ax1.patches:
    h = p.get_height()
    if h > 0:
        ax1.annotate(f"{int(h)}", (p.get_x() + p.get_width()/2, h + 20),
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Anatomical Planes
plane_counts = train_series_df['Anatomical_Plane'].value_counts()
sns.barplot(x=plane_counts.index, y=plane_counts.values, hue=plane_counts.index, palette='mako', legend=False, ax=ax2)
ax2.set_title('Total Series by Anatomical Plane', fontsize=12, fontweight='bold')
ax2.set_xlabel('Anatomical Plane')
ax2.set_ylabel('Series Count')
for p in ax2.patches:
    ax2.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height() + 100),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Leak-Free 5-Fold Multilabel Stratified Split Generation"""))

    cells.append(nbf.v4.new_code_cell("""from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Ensure reproducible 5-fold partition
SEED = 42
N_SPLITS = 5

mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
X = df_labeled["StudyInstanceUID"].values
Y = df_labeled[TARGET_COLUMNS].fillna(0).values

df_labeled["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(mskf.split(X, Y)):
    df_labeled.iloc[val_idx, df_labeled.columns.get_loc("fold")] = fold_idx

# Partition unlabeled studies evenly across folds
df_unlabeled["fold"] = np.random.RandomState(SEED).randint(0, N_SPLITS, size=len(df_unlabeled))

# Master splits dataframe
splits_df = pd.concat([df_labeled, df_unlabeled], ignore_index=True)
os.makedirs("data", exist_ok=True)
splits_df.to_csv("data/splits_5fold.csv", index=False)

print(f"Master splits generated successfully with {N_SPLITS} folds.")
display(splits_df.groupby(['fold', splits_df[TARGET_COLUMNS].notnull().any(axis=1)]).size().unstack().rename(columns={False: 'Unlabeled', True: 'Labeled'}))
"""))

    nb.cells = cells
    with open("notebooks/01_dataset_and_labels_eda.ipynb", "w") as f:
        nbf.write(nb, f)
    print("✓ Created notebooks/01_dataset_and_labels_eda.ipynb")


def build_notebook_02():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 02: DICOM Volumes & 3D Spatial Geometry EDA
## RSNA Knee Abnormality Detection (KneeVision-AI)

This notebook examines the volumetric imaging characteristics of the DICOM series:
1. **DICOM Metadata Tag Extraction**: Slice thickness, pixel spacing, transfer syntaxes, and photometric interpretation.
2. **Variable Slice Depth ($D$) Analysis**: Quantifying slice counts per series (11 to 320 slices).
3. **3D Spatial Slice Ordering**: Sorting slices along the physical slice normal vector:
   $$\\vec{n} = \\vec{u}_{\\text{row}} \\times \\vec{u}_{\\text{col}}, \\quad z_{\\text{projected}} = \\vec{p}_{\\text{patient}} \\cdot \\vec{n}$$
4. **Intensity Normalization & VOI LUT Windowing**: Clipping percentiles and rescaling.
5. **Multi-Plane Volume Visualization**: Rendering Sagittal, Coronal, and Axial slice series.
"""))

    cells.append(nbf.v4.new_code_cell(DATA_RESOLVER + """
import glob
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dicom_reader import read_dicom_slice, load_series_volume

TRAIN_SERIES_DIR = f'{DATA_DIR}/train_series'
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Volumetric Header & Geometry Extraction (Sample 1500 Series)"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect sample series across studies
study_folders = [f for f in os.listdir(TRAIN_SERIES_DIR) if os.path.isdir(os.path.join(TRAIN_SERIES_DIR, f))]
print(f"Total Study Folders in train_series: {len(study_folders):,}")

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
                'StudyUID': study,
                'SeriesUID': series,
                'NumSlices': len(dcm_files),
                'Rows': dcm.Rows,
                'Columns': dcm.Columns,
                'PixelSpacing_X': float(dcm.PixelSpacing[0]) if hasattr(dcm, 'PixelSpacing') else np.nan,
                'PixelSpacing_Y': float(dcm.PixelSpacing[1]) if hasattr(dcm, 'PixelSpacing') else np.nan,
                'SliceThickness': float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else np.nan,
                'Photometric': getattr(dcm, 'PhotometricInterpretation', 'Unknown'),
            })
        except Exception:
            pass

df_meta = pd.DataFrame(meta_records)
print(f"Extracted metadata for {len(df_meta):,} series.")
display(df_meta.head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Slice Count & Spatial Resolution Distributions"""))

    cells.append(nbf.v4.new_code_cell("""fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Slice Counts
sns.histplot(df_meta['NumSlices'], bins=40, kde=True, color='#2b5c8f', ax=ax1)
ax1.set_title('Slice Count Distribution per Series (Depth)', fontweight='bold')
ax1.set_xlabel('Number of Slices (D)')
ax1.axvline(df_meta['NumSlices'].median(), color='crimson', linestyle='--', label=f"Median: {df_meta['NumSlices'].median():.0f}")
ax1.legend()

# 2. Slice Thickness
st_counts = df_meta['SliceThickness'].value_counts().head(5)
sns.barplot(x=st_counts.index.astype(str) + " mm", y=st_counts.values, hue=st_counts.index.astype(str), palette='crest', legend=False, ax=ax2)
ax2.set_title('Slice Thickness Distribution', fontweight='bold')
ax2.set_xlabel('Slice Thickness')
ax2.set_ylabel('Series Count')

# 3. Image Dimensions (Rows x Cols)
shape_series = df_meta['Rows'].astype(str) + "x" + df_meta['Columns'].astype(str)
top_shapes = shape_series.value_counts().head(6)
sns.barplot(x=top_shapes.index, y=top_shapes.values, hue=top_shapes.index, palette='mako', legend=False, ax=ax3)
ax3.set_title('Top Image Dimensions (H x W)', fontweight='bold')
ax3.set_xlabel('Resolution')
ax3.set_ylabel('Series Count')

# 4. In-Plane Pixel Spacing
sns.histplot(df_meta['PixelSpacing_X'].dropna(), bins=30, kde=True, color='#8856a7', ax=ax4)
ax4.set_title('In-plane Pixel Spacing (mm) Distribution', fontweight='bold')
ax4.set_xlabel('Pixel Spacing (mm)')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Spatial Slice Sorting along Patient Normal Vector vs Instance Number"""))

    cells.append(nbf.v4.new_code_cell("""# Select an example series to demonstrate 3D spatial sorting
sample_study = study_folders[0]
sample_study_path = os.path.join(TRAIN_SERIES_DIR, sample_study)
sample_series = os.listdir(sample_study_path)[0]
sample_series_path = os.path.join(sample_study_path, sample_series)

print(f"Loading Series: {sample_series_path}")
vol, metas = load_series_volume(sample_series_path, target_slices=32, normalize='minmax')
print(f"Loaded Volume Shape: {vol.shape} (Depth x Height x Width)")
print(f"Intensity Range: Min={vol.min():.3f}, Max={vol.max():.3f}, Mean={vol.mean():.3f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Multi-Plane Volume Montage Visualization"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize an 8-slice montage through the volume
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
indices = np.linspace(4, vol.shape[0] - 5, 8).astype(int)

for idx, ax in zip(indices, axes.flatten()):
    ax.imshow(vol[idx], cmap='bone')
    ax.set_title(f"Slice #{idx+1}/{vol.shape[0]}", fontsize=10, fontweight='bold')
    ax.axis('off')

plt.suptitle(f"Volumetric Slices through Knee MRI Series ({sample_series[:16]}...)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Tri-Planar Study Visualization (Sagittal + Coronal + Axial)"""))

    cells.append(nbf.v4.new_code_cell("""train_series_csv = pd.read_csv(f"{DATA_DIR}/train_series.csv")
study_series_df = train_series_csv[train_series_csv['StudyInstanceUID'] == sample_study]
display(study_series_df)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
            axes[i].set_title(f"{plane} View (Mid-Slice {mid_idx})", fontweight='bold')
    axes[i].axis('off')

plt.suptitle(f"Tri-Planar Knee Views for Study: {sample_study[:20]}...", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    with open("notebooks/02_dicom_volumes_and_spatial_eda.ipynb", "w") as f:
        nbf.write(nb, f)
    print("✓ Created notebooks/02_dicom_volumes_and_spatial_eda.ipynb")


def build_notebook_03():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 03: Radiology Reports NLP & Weak Supervision EDA
## RSNA Knee Abnormality Detection (KneeVision-AI)

This notebook analyzes the **4,349 weakly supervised radiology reports** (in Spanish) in `train.csv`:
1. **Corpus Linguistics & Structure**: Text length, token distributions, standard report section headers (*Técnica*, *Hallazgos*, *Impresión / Conclusión*).
2. **Clinical Terminology Extraction**: Frequency of Spanish anatomical keywords (*menisco interno/externo*, *ligamento cruzado anterior*, *artrosis*, *derrame articular*, *edema óseo*).
3. **Rule-Based & Regex Weak Label Extractor**: Building an NLP extractor for all 12 abnormality targets.
4. **Validation Benchmark**: Evaluating the weak label extractor against the 58 ground-truth human annotations (Precision, Recall, F1, Accuracy).
5. **Pseudo-Label Generation**: Exporting weakly supervised labels to augment model training.
"""))

    cells.append(nbf.v4.new_code_cell(DATA_RESOLVER + """
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Radiology Report Corpus Statistics"""))

    cells.append(nbf.v4.new_code_cell("""reports = train_df['Report'].dropna()
train_df['char_count'] = train_df['Report'].fillna('').apply(len)
train_df['word_count'] = train_df['Report'].fillna('').apply(lambda x: len(x.split()))

print(f"Total Reports: {len(reports):,}")
print(f"Word Count - Median: {train_df['word_count'].median():.0f}, Mean: {train_df['word_count'].mean():.1f}, Max: {train_df['word_count'].max()}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
sns.histplot(train_df['word_count'], bins=35, kde=True, color='navy', ax=ax1)
ax1.set_title('Radiology Report Word Count Distribution', fontweight='bold')
ax1.set_xlabel('Words per Report')

# Top section headers
has_hallazgos = train_df['Report'].str.contains('hallazgos|findings', case=False, na=False).sum()
has_impresion = train_df['Report'].str.contains('impresión|conclusion|diagnóstico', case=False, na=False).sum()
has_tecnica = train_df['Report'].str.contains('técnica|secuencias', case=False, na=False).sum()

sections_df = pd.DataFrame({
    'Section': ['Hallazgos (Findings)', 'Impresión (Impression)', 'Técnica (Technique)'],
    'Present in Reports (%)': [has_hallazgos/len(reports)*100, has_impresion/len(reports)*100, has_tecnica/len(reports)*100]
})
sns.barplot(data=sections_df, x='Present in Reports (%)', y='Section', hue='Section', palette='Blues_r', legend=False, ax=ax2)
ax2.set_title('Standard Section Structure Presence (%)', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Spanish Medical Regex & NLP Extractor Implementation"""))

    cells.append(nbf.v4.new_code_cell("""def extract_weak_labels_from_report(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {col: 0.0 for col in TARGET_COLUMNS}
    
    t = text.lower()
    labels = {}
    
    # 1. ACL
    acl_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|avulsi[oó]n|interrupci[oó]n|afectaci[oó]n).*?(cruzado anterior|lca)', t) or
                  re.search(r'(lca|cruzado anterior).*?(rotura|desgarro|lesi[oó]n|avulsi[oó]n|interrumpido)', t))
    acl_neg = bool(re.search(r'(cruzado anterior|lca).*?(normal|intacto|continuo|sin alteraciones|sin signos de rotura|conservad[oa])', t) or
                  re.search(r'(no se observan|sin).*?(rotura|lesi[oó]n).*?(cruzado anterior|lca)', t))
    labels['ACL'] = 1.0 if (acl_pos and not acl_neg) else 0.0
    
    # 2. MCL
    mcl_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|esguince|engrosamiento).*?(colateral medial|colateral interno|lcm|lci)', t))
    mcl_neg = bool(re.search(r'(colateral medial|colateral interno|lcm|lci).*?(normal|intacto|continuo|sin alteraciones|conservad[oa])', t))
    labels['MCL'] = 1.0 if (mcl_pos and not mcl_neg) else 0.0
    
    # 3. Medial Meniscus
    mm_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|fisura|amputaci[oó]n|compleja).*?(menisco medial|menisco interno|cuerpo posterior.*?interno|cuerno posterior.*?medial)', t) or
                 re.search(r'(menisco medial|menisco interno).*?(rotura|desgarro|lesi[oó]n|fisura|amputaci[oó]n)', t))
    mm_neg = bool(re.search(r'(menisco medial|menisco interno).*?(morfolog[ií]a y se[nñ]al conservada|normal|sin alteraciones|sin signos de rotura)', t))
    labels['Medial Meniscus'] = 1.0 if (mm_pos and not mm_neg) else 0.0

    # 4. Lateral Meniscus
    lm_pos = bool(re.search(r'(rotura|desgarro|lesi[oó]n|fisura|amputaci[oó]n).*?(menisco lateral|menisco externo)', t) or
                 re.search(r'(menisco lateral|menisco externo).*?(rotura|desgarro|lesi[oó]n|fisura|amputaci[oó]n)', t))
    lm_neg = bool(re.search(r'(menisco lateral|menisco externo).*?(morfolog[ií]a y se[nñ]al conservada|normal|sin alteraciones|sin signos de rotura)', t))
    labels['Lateral Meniscus'] = 1.0 if (lm_pos and not lm_neg) else 0.0

    # 5. Medial OA
    labels['Medial OA'] = 1.0 if re.search(r'(artrosis|condropat[ií]a|desgaste|pinzamiento).*?(medial|interno|femorotibial medial|femorotibial interno)', t) else 0.0
    
    # 6. Lateral OA
    labels['Lateral OA'] = 1.0 if re.search(r'(artrosis|condropat[ií]a|desgaste|pinzamiento).*?(lateral|externo|femorotibial lateral|femorotibial externo)', t) else 0.0
    
    # 7. PF OA
    labels['PF OA'] = 1.0 if re.search(r'(artrosis|condropat[ií]a|condromalacia|desgaste).*?(femoropatelar|patelar|rotulian[oa]|tr[oó]clea)', t) else 0.0

    # 8. Effusion
    eff_pos = bool(re.search(r'(derrame|l[ií]quido intraarticular|hidrartros|abundante l[ií]quido)', t))
    eff_neg = bool(re.search(r'(sin derrame|no se observa derrame|derrame ausente|m[ií]nima cuant[ií]a fisiol[oó]gica)', t))
    labels['Effusion'] = 1.0 if (eff_pos and not eff_neg) else 0.0

    # 9. Synovitis
    labels['Synovitis'] = 1.0 if re.search(r'(sinovitis|engrosamiento sinovial|hipertrofia sinovial|plica sinovial)', t) else 0.0

    # 10. Baker's
    labels["Baker's"] = 1.0 if re.search(r'(quiste de baker|quiste popl[ií]teo|baker)', t) else 0.0

    # 11. Contusion
    labels['Contusion'] = 1.0 if re.search(r'(edema [oó]seo|contusi[oó]n [oó]sea|bruise|edema trabecular)', t) else 0.0

    # 12. Fracture
    labels['Fracture'] = 1.0 if re.search(r'(fractura|trazo de fractura|avulsi[oó]n [oó]sea|hundimiento cortical)', t) else 0.0

    return labels

print("Weak label extractor initialized.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Benchmark Extractor vs Ground-Truth Annotations (N=58)"""))

    cells.append(nbf.v4.new_code_cell("""labeled_df = train_df[train_df[TARGET_COLUMNS].notnull().any(axis=1)].copy()

# Run extractor on labeled subset
extracted = [extract_weak_labels_from_report(r) for r in labeled_df['Report']]
df_extracted = pd.DataFrame(extracted, index=labeled_df.index)

benchmark_rows = []
for col in TARGET_COLUMNS:
    y_true = labeled_df[col].values
    y_pred = df_extracted[col].values
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    benchmark_rows.append({
        'Abnormality': col,
        'Accuracy (%)': round(acc * 100, 1),
        'F1-Score': round(f1, 3),
        'True Positives (GT)': int(y_true.sum()),
        'Extracted Positives': int(y_pred.sum())
    })

bench_df = pd.DataFrame(benchmark_rows)
display(bench_df)

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("crest", len(bench_df))
bars = ax.barh(bench_df['Abnormality'], bench_df['Accuracy (%)'], color=colors, height=0.65)
ax.set_title('NLP Weak Label Extractor Accuracy vs Ground Truth (N=58)', fontweight='bold')
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
            ha='left', va='center', fontsize=9, fontweight='bold')
plt.xlim(0, 110)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Generate & Save Master Pseudo-Labels for all 4,349 Unlabeled Studies"""))

    cells.append(nbf.v4.new_code_cell("""# Extract pseudo-labels across the entire dataset
all_extracted = [extract_weak_labels_from_report(r) for r in train_df['Report']]
pseudo_df = pd.DataFrame(all_extracted)
pseudo_df['StudyInstanceUID'] = train_df['StudyInstanceUID']

# Overwrite with ground-truth labels where available
for col in TARGET_COLUMNS:
    pseudo_df[col] = train_df[col].combine_first(pseudo_df[col])

pseudo_df.to_csv("data/pseudo_labels_master.csv", index=False)
print("✓ Master pseudo-labels saved to data/pseudo_labels_master.csv")
print("Total positive counts in master pseudo-dataset:")
print(pseudo_df[TARGET_COLUMNS].sum())
"""))

    nb.cells = cells
    with open("notebooks/03_radiology_reports_nlp_eda.ipynb", "w") as f:
        nbf.write(nb, f)
    print("✓ Created notebooks/03_radiology_reports_nlp_eda.ipynb")


def build_notebook_04():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 04: Multi-Instance Learning (MIL) & Multi-View Pipeline EDA
## RSNA Knee Abnormality Detection (KneeVision-AI)

This notebook demonstrates the end-to-end PyTorch deep learning data loading and model architecture pipeline:
1. **Handling Variable Slices via MIL Bags**: How variable depth ($D \\in [11, 320]$) is processed without forcing fixed interpolation.
2. **2.5D Slice Stacking ($z-1, z, z+1$)**: Capturing inter-slice volumetric context into 3-channel RGB image tensors.
3. **Albumentations Augmentation Pipeline**: Spatial flips, affine rotations, brightness/contrast jittering.
4. **Tri-Planar Study Collation (`Sagittal`, `Coronal`, `Axial`)**: Aggregating orthogonal planes into multi-view study representations.
5. **Gated Attention MIL Pooling & Attention Map Visualization**: Inspecting how attention weights pinpoint the pathological slice.
"""))

    cells.append(nbf.v4.new_code_cell(DATA_RESOLVER + """
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.dicom_reader import load_series_volume
from src.data.transforms import get_train_transforms, get_valid_transforms

TRAIN_SERIES_DIR = f'{DATA_DIR}/train_series'
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. 2.5D Slice Stacking Implementation Demonstration"""))

    cells.append(nbf.v4.new_code_cell("""def stack_25d_slices(volume_3d: np.ndarray) -> np.ndarray:
    \"\"\"
    Converts (D, H, W) volume to (D, 3, H, W) 2.5D stacked slices.
    Each slice z receives channels [z-1, z, z+1].
    \"\"\"
    D, H, W = volume_3d.shape
    stacked = np.zeros((D, 3, H, W), dtype=np.float32)
    for z in range(D):
        z_prev = max(0, z - 1)
        z_next = min(D - 1, z + 1)
        stacked[z, 0] = volume_3d[z_prev]
        stacked[z, 1] = volume_3d[z]
        stacked[z, 2] = volume_3d[z_next]
    return stacked

# Load sample series
study_id = os.listdir(TRAIN_SERIES_DIR)[0]
series_id = os.listdir(os.path.join(TRAIN_SERIES_DIR, study_id))[0]
vol, _ = load_series_volume(os.path.join(TRAIN_SERIES_DIR, study_id, series_id), target_slices=24, normalize='minmax')
vol_25d = stack_25d_slices(vol)

print(f"Original Volume: {vol.shape}")
print(f"2.5D Stacked Volume: {vol_25d.shape} (Depth, 3 Channels, Height, Width)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Albumentations Augmentations on 2.5D Slices"""))

    cells.append(nbf.v4.new_code_cell("""aug = get_train_transforms(image_size=256)

# Apply augmentations to slice 12
sample_slice = (vol_25d[12].transpose(1, 2, 0) * 255).astype(np.uint8)
augmented = aug(image=sample_slice)['image']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(sample_slice)
ax1.set_title("Original 2.5D Slice (RGB mapped)", fontweight='bold')
ax1.axis('off')

ax2.imshow(augmented.permute(1, 2, 0).numpy())
ax2.set_title("Augmented Slice (Flips, Rotation, Jitter)", fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Gated Attention MIL Pooling & Attention Heatmaps"""))

    cells.append(nbf.v4.new_code_cell("""class GatedAttentionMILPool(nn.Module):
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

feat_dim = 128
mil_pool = GatedAttentionMILPool(in_features=feat_dim, hidden_dim=64, num_classes=12)

# Simulate slice embeddings from a backbone for a series with 24 slices
num_slices = 24
slice_features = torch.randn(1, num_slices, feat_dim)
slice_features[0, 14] += 3.5

logits, attn_weights = mil_pool(slice_features)
attn = attn_weights.squeeze().detach().numpy()

print(f"Pooled Logits Shape: {logits.shape} (12 Targets)")
print(f"Attention Weights Shape: {attn.shape} (Sum = {attn.sum():.3f})")

# Plot Attention Weights across Slices
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, num_slices + 1), attn, marker='o', color='#d95f02', linewidth=2)
ax.fill_between(range(1, num_slices + 1), attn, alpha=0.3, color='#d95f02')
ax.set_title('MIL Attention Weights across 24 Slices (Pathology Localization)', fontweight='bold')
ax.set_xlabel('Slice Index')
ax.set_ylabel('Attention Weight (Softmax Probability)')
ax.axvline(15, color='crimson', linestyle='--', label='Pathological Slice (#15)')
ax.legend()
plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    with open("notebooks/04_mil_bag_and_multiview_pipeline.ipynb", "w") as f:
        nbf.write(nb, f)
    print("✓ Created notebooks/04_mil_bag_and_multiview_pipeline.ipynb")

if __name__ == '__main__':
    build_notebook_01()
    build_notebook_02()
    build_notebook_03()
    build_notebook_04()
    print("\n✓ Successfully built all 4 data analysis notebooks!")
