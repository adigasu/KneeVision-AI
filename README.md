# KneeVision-AI: Multimodal Foundation Modeling for RSNA Knee Abnormality Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning and multimodal foundation model framework for the **RSNA Knee Abnormality Detection** challenge.

This repository is built for both **competitive performance** and **systematic skill development** in modern medical AI engineering (tracked in [`skills.md`](skills.md)).

---

## 🔬 Clinical Targets & Evaluation

The goal is to detect **12 clinically important knee abnormalities** across varied MRI sequences (`Sagittal`, `Coronal`, `Axial`) and sequence contrasts (`Fluid_Sensitive`, `Fat_Suppression`):

| # | Abnormality Target | Description / Clinical Significance |
|---|---|---|
| 1 | **ACL** | Anterior Cruciate Ligament tear / injury |
| 2 | **MCL** | Medial Collateral Ligament tear / injury |
| 3 | **Medial Meniscus** | Medial meniscus tear / degenerative degeneration |
| 4 | **Lateral Meniscus** | Lateral meniscus tear |
| 5 | **Medial OA** | Osteoarthritis of medial tibiofemoral compartment |
| 6 | **Lateral OA** | Osteoarthritis of lateral tibiofemoral compartment |
| 7 | **PF OA** | Patellofemoral osteoarthritis |
| 8 | **Effusion** | Joint effusion / intra-articular excess fluid |
| 9 | **Synovitis** | Synovial membrane inflammation |
| 10 | **Baker's** | Popliteal (Baker's) cyst |
| 11 | **Contusion** | Subchondral bone marrow edema / bruise |
| 12 | **Fracture** | Acute or occult bone fracture |

### Metric
Submissions are evaluated on **Macro-Averaged AUC-ROC** across all 12 targets:
$$\text{Macro AUC} = \frac{1}{12} \sum_{c=1}^{12} \text{AUC}_c$$

---

## 🏗️ System Architecture

```
                                  Knee MRI Study
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
      Sagittal Series             Coronal Series               Axial Series
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
                                         │
                   (ACL, MCL, Meniscus, OA, Effusion, etc.)
```

### Multimodal Training Strategy
1. **Weak Supervision from Radiology Reports**: The dataset includes ~58k studies with raw multilingual radiology reports (`train.csv`), of which only a subset have ground-truth binary labels. Medical LLM extractors parse reports into pseudo-labels.
2. **Vision-Language Alignment**: Vision encoders are contrastively aligned with medical text embeddings (e.g. BiomedCLIP / Clinical-Longformer).
3. **Multi-Instance Learning (MIL)**: Slices in each series are aggregated dynamically via gated attention pooling, learning which slices carry pathology without requiring slice-level annotations.

---

## 📁 Repository Structure

```
KneeVision-AI/
├── configs/                  # Experiment and model configuration files
│   └── base.yaml             # Core dataset, model, and training hyperparams
├── src/                      # Source package (kneevision)
│   ├── data/                 # DICOM decoding, volume handling, datasets & transforms
│   │   ├── dicom_reader.py   # Spatial sorting & transfer syntax decoding
│   │   ├── dataset.py        # PyTorch KneeMRIDataset
│   │   └── transforms.py     # Albumentations augmentations
│   ├── models/               # Neural network architectures
│   │   └── mil_backbone.py   # 2.5D timm backbone + Gated Attention MIL
│   ├── metrics/              # Evaluation metrics
│   │   └── auc_metrics.py    # Competition Macro ROC-AUC calculation
│   ├── nlp/                  # Radiology text processing & weak supervision
│   └── utils/                # Reproducibility, seed, and config utilities
│       └── common.py
├── scripts/                  # Command-line training and preprocessing scripts
│   └── 01_eda_and_splits.py  # EDA & 5-fold Multilabel Stratified Split
├── tests/                    # Unit and integration tests
│   └── test_metrics.py       # Metric validation test suite
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation descriptor
└── skills.md                 # Project learning objectives checklist
```

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
```bash
# Activate conda environment
conda activate rsna-knee

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Exploratory Data Analysis & Generate 5-Fold Splits
```bash
python scripts/01_eda_and_splits.py \
    --train_csv ../rsna-knee-abnormality-detection/train.csv \
    --train_series_csv ../rsna-knee-abnormality-detection/train_series.csv \
    --output_splits ./data/splits_5fold.csv
```

### 3. Run Metric Unit Tests
```bash
pytest tests/test_metrics.py
```

---

## 🎓 Learning Curriculum ([skills.md](skills.md))
- [x] DICOM spatial slice ordering & transfer syntax decompression
- [x] Multi-label Macro ROC-AUC metric formulation & NaN masking
- [x] Attention-based Multiple Instance Learning (MIL) pooling
- [x] Multilabel Stratified K-Fold cross-validation partitioning
- [ ] Radiology Report NLP extraction & weak supervision
- [ ] Cross-view anatomical plane fusion (Sagittal + Coronal + Axial)
- [ ] Mixed-precision FP16 training & torch.compile optimization
- [ ] Knowledge distillation & Kaggle inference deployment
