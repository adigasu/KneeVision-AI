# KneeVision-AI: Multimodal Foundation Modeling for RSNA Knee Abnormality Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)

An end-to-end deep learning and multimodal foundation model framework for the **RSNA Knee Abnormality Detection** challenge.

This repository is built for both **competitive performance** and **systematic skill development** in modern medical AI engineering (tracked in [`skills.md`](skills.md) and [`docs/data_analysis_and_architecture_guide.md`](docs/data_analysis_and_architecture_guide.md)).

---

## 🔬 Clinical Targets & Evaluation

The goal is to detect **12 clinically important knee abnormalities** across varied MRI sequences (`Sagittal`, `Coronal`, `Axial`) and sequence contrasts (`Fluid_Sensitive`, `Fat_Suppression`):

| # | Abnormality Target | Description / Clinical Significance | Prevalence (GT $N=58$) |
|---|---|---|---|
| 1 | **Effusion** | Joint effusion / intra-articular excess fluid | **60.3%** |
| 2 | **Synovitis** | Synovial membrane inflammation | **46.6%** |
| 3 | **Medial Meniscus** | Medial meniscus tear / degenerative degeneration | **44.8%** |
| 4 | **ACL** | Anterior Cruciate Ligament tear / injury | **41.4%** |
| 5 | **Lateral Meniscus** | Lateral meniscus tear | **39.7%** |
| 6 | **PF OA** | Patellofemoral osteoarthritis / cartilage loss | **36.2%** |
| 7 | **Contusion** | Subchondral bone marrow edema / bruise | **32.8%** |
| 8 | **Fracture** | Acute or occult bone fracture | **31.0%** |
| 9 | **Medial OA** | Osteoarthritis of medial tibiofemoral joint | **25.9%** |
| 10 | **Baker's** | Popliteal (Baker's) cyst in posterior fossa | **20.7%** |
| 11 | **Lateral OA** | Osteoarthritis of lateral tibiofemoral joint | **19.0%** |
| 12 | **MCL** | Medial Collateral Ligament tear / injury | **15.5%** |

### Evaluation Metric
Submissions are evaluated on **Macro-Averaged AUC-ROC** across all 12 targets:
$$\text{Macro AUC} = \frac{1}{12} \sum_{c=1}^{12} \text{AUC}_c$$

---

## 🏗️ System Architecture & 2.5D MIL Design

```
                                  Knee MRI Study
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
      Sagittal Series             Coronal Series               Axial Series
   (ACL, Menisci, Contusion)     (MCL, Cartilage, OA)       (PF OA, Patella)
             │                           │                           │
    [2.5D Slice Encoder]        [2.5D Slice Encoder]        [2.5D Slice Encoder]
   (ConvNeXt / Swin / EVA)     (ConvNeXt / Swin / EVA)     (ConvNeXt / Swin / EVA)
             │                           │                           │
   [Gated Attention MIL]       [Gated Attention MIL]       [Gated Attention MIL]
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                             [Cross-Plane Multi-View Fusion]
                                         │
                         [12-Target Multi-Label Head]
```

### Why 2.5D + Gated Attention MIL?
* **Preserves In-Plane Detail ($0.4\text{mm}$)**: Avoids blurring from $3\times 3\times 3$ 3D kernels on thick $3.0\text{mm}$ slices.
* **Variable Slice Counts ($11 \to 320$ slices)**: Dynamic MIL bag aggregation without geometric resampling distortion.
* **Modern Foundation Models**: Leverages ImageNet-22k and medical pretrained backbones (ConvNeXt-V2, Swin, BiomedCLIP).

---

## 📁 Repository Structure

```
KneeVision-AI/
├── configs/                  # Experiment and model configuration files
│   └── base.yaml             # Base hyperparameters & training specs
├── data/                     # Partitioned splits & manifest files
│   ├── splits_5fold.parquet  # 5-fold Multilabel Stratified Splits (Snappy)
│   ├── pseudo_labels_master.parquet # 12 weak targets across 4,349 studies
│   └── preprocessed_index.parquet   # Fast cached series manifest
├── docs/                     # Technical documentation & architecture guides
│   └── data_analysis_and_architecture_guide.md
├── notebooks/                # Pre-executed Jupyter EDA & Pipeline notebooks
│   ├── 01_dataset_and_labels_eda.ipynb
│   ├── 02_dicom_volumes_and_spatial_eda.ipynb
│   ├── 03_radiology_reports_nlp_eda.ipynb
│   └── 04_mil_bag_and_multiview_pipeline.ipynb
├── reports/figures/          # Publication-quality visual EDA figures
│   ├── 01_target_prevalences.png
│   ├── 02_target_correlations.png
│   ├── 03_series_and_planes_distribution.png
│   ├── 04_dicom_geometry_and_depth.png
│   ├── 05_multi_slice_montage.png
│   ├── 06_triplanar_views.png
│   ├── 07_nlp_report_metrics.png
│   └── 08_mil_attention_simulation.png
├── src/                      # Source package (kneevision)
│   ├── data/                 # Volume preprocessing, dataset & transforms
│   │   ├── preprocessing.py  # 3D spatial slice normal sorting & 2.5D stacking
│   │   ├── dataset.py        # PyTorch KneeMRIDataset (Parquet/NPY loader)
│   │   ├── transforms.py     # Albumentations augmentations
│   │   └── dicom_reader.py   # Low-level DICOM parsing & VOI LUT
│   ├── models/               # Neural network architectures
│   │   └── mil_backbone.py   # 2.5D backbone + Gated Attention MIL
│   ├── metrics/              # Evaluation metrics
│   │   └── auc_metrics.py    # Competition Macro ROC-AUC calculation
│   └── utils/                # Reproducibility and seeding
├── scripts/                  # CLI execution scripts
│   ├── 01_eda_and_splits.py         # EDA & 5-fold split generation
│   ├── 02_preprocess_and_cache.py   # Multi-process parallel volume caching
│   ├── generate_eda_plots.py        # High-res figure generator
│   └── build_all_notebooks.py       # Notebook builder
├── tests/                    # Unit & integration tests
│   ├── test_dataset.py       # Dataset & Parquet validation suite
│   └── test_metrics.py       # Metric validation test suite
├── requirements.txt          # Python dependencies
├── setup.py                  # Package descriptor
└── skills.md                 # Learning objectives checklist
```

---


---

## 🏆 5-Fold Cross-Validation Leaderboard

Cross-validation performance is tracked in [](docs/experimentation_log.md).

| Experiment | Architecture | Modality / Input | Pretraining | Loss Function | **Mean 5-Fold Macro AUC** | Status |
|:---|:---|:---|:---|:---|:---:|:---:|
| **EXP-001** |  + Gated MIL | 2.5D Sagittal Slices | ImageNet-1k | Asymmetric Loss (ASL) | **0.5600** | Completed |

Detailed technical documentation and mathematical formulation are available in [](docs/phase_2_walkthrough.md).

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Activate conda environment
conda activate rsna-knee

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Multi-Process Volume Preprocessing & Caching
```bash
# Preprocess the 58 ground-truth labeled studies into 256x256 2.5D tensors (~4 sec)
python scripts/02_preprocess_and_cache.py --labeled_only --num_workers 16

# Or preprocess the entire 4,407-study dataset (~4-5 mins)
python scripts/02_preprocess_and_cache.py --num_workers 16
```

### 3. Run Test Suite
```bash
pytest tests/ -W ignore::DeprecationWarning
```

### 4. Explore Interactive Visual Notebooks
```bash
jupyter lab notebooks/
```
