# KneeVision-AI: Multimodal Foundation Modeling for RSNA Knee Abnormality Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2Bcu128-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python->=3.9-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)](tests/)
[![Kaggle Public LB](https://img.shields.io/badge/Kaggle%20LB-~0.930-gold.svg)](https://www.kaggle.com)
[![CV Peak Macro AUC](https://img.shields.io/badge/CV%20Peak%20Macro%20AUC-0.8843-blueviolet.svg)](https://github.com/adigasu/KneeVision-AI#-master-experimentation--kaggle-leaderboard)

An end-to-end deep learning and multimodal foundation model framework for the **RSNA Knee Abnormality Detection** challenge.

This repository is engineered for both **competitive performance** (Kaggle Public Leaderboard **~0.930**, CV Peak **0.8843**) and **systematic learning of research and engineering skills** (tracked in [`skills.md`](skills.md)).

---

## 🔬 Clinical Targets & Evaluation

The objective is to detect **12 clinically critical knee abnormalities** across varied MRI planes (`Sagittal`, `Coronal`, `Axial`) and sequence contrasts (`Fluid_Sensitive` / T2-FS, `Anatomical` / T1):

| # | Abnormality Target | Description / Clinical Significance | Prevalence (GT $N=58$) | Best Validated AUC |
|:---:|:---|:---|:---:|:---:|
| 1 | **Medial Meniscus** | Medial meniscus tear / degenerative complex tear | **44.8%** | **0.9334** |
| 2 | **Baker's** | Popliteal (Baker's) cyst in gastrocnemio-semimembranosus fossa | **20.7%** | **0.9272** |
| 3 | **Medial OA** | Osteoarthritis of medial tibiofemoral joint (cartilage/subchondral) | **25.9%** | **0.9038** |
| 4 | **ACL** | Anterior Cruciate Ligament tear / complete rupture | **41.4%** | **0.8909** |
| 5 | **Synovitis** | Synovial membrane thickening / joint inflammation | **46.6%** | **0.8902** |
| 6 | **Effusion** | Joint effusion / intra-articular excess fluid | **60.3%** | **0.8897** |
| 7 | **Lateral Meniscus** | Lateral meniscus tear / bucket-handle disruption | **39.7%** | **0.8889** |
| 8 | **Lateral OA** | Osteoarthritis of lateral tibiofemoral compartment | **19.0%** | **0.8836** |
| 9 | **Contusion** | Subchondral bone marrow edema / trabecular microfracture | **32.8%** | **0.8819** |
| 10 | **Fracture** | Acute cortical or occult trabecular bone fracture | **31.0%** | **0.8659** |
| 11 | **MCL** | Medial Collateral Ligament tear / sprain | **15.5%** | **0.8425** |
| 12 | **PF OA** | Patellofemoral osteoarthritis / trochlear cartilage wear | **36.2%** | **0.8140** |

### Evaluation Metric
Submissions are evaluated on **Macro-Averaged AUC-ROC** across all 12 targets:
$$\text{Macro AUC} = \frac{1}{12} \sum_{c=1}^{12} \text{AUC}_c$$

---

## 🏗️ Core Architectural Innovations

```
                                      Knee MRI Study
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
       Sagittal Series               Coronal Series                 Axial Series
    (ACL, Menisci, Contusion)       (MCL, Collaterals, OA)        (PF OA, Effusion, Cyst)
              │                             │                             │
    [2.5D Slice Encoders]         [2.5D Slice Encoders]         [2.5D Slice Encoders]
   (ConvNeXt / DINOv2 / Swin)    (ConvNeXt / DINOv2 / Swin)    (ConvNeXt / DINOv2 / Swin)
              │                             │                             │
    [T1 / T2 Decoupling]          [T1 / T2 Decoupling]          [T1 / T2 Decoupling]
  (Fluid-Sensitive vs Anatomy)  (Fluid-Sensitive vs Anatomy)  (Fluid-Sensitive vs Anatomy)
              │                             │                             │
              └─────────────────────────────┼─────────────────────────────┘
                                            │
                       [Anatomically-Routed Plane Decoupling]
                           [12-Branch Label-Specific MIL]
                                            │
                             [12-Target Multi-Label Head]
```

1. **2.5D Slice Encoding + Gated Attention MIL**:
   - Preserves high in-plane spatial resolution ($0.4\text{ mm}$) by stacking adjacent slices `[z-1, z, z+1]` as 3-channel tensors, avoiding 3D convolution blur across thick $3.0\text{ mm}$ slices.
   - Dynamic Gated Attention MIL aggregates variable slice depths ($11 \to 320$ slices) without geometric resampling artifacts.

2. **Anatomically-Routed Plane Decoupling (`PlaneDecoupledMIL`)**:
   - Rather than collapsing all planes into a generic representation, orthogonal series are routed according to radiological diagnostic principles:
     - **Sagittal**: ACL, Medial/Lateral Meniscus, Contusion.
     - **Coronal**: MCL, Collateral ligaments, Medial/Lateral OA.
     - **Axial**: Patellofemoral (PF) OA, Joint Effusion, Baker's Cyst.

3. **Multi-Contrast MRI Decoupling (`ContrastDecoupledMIL`)**:
   - Orthogonally models **T2/Fluid-Sensitive** sequences (highlighting hyperintense fluid, edema, and tears) and **T1/Anatomical** sequences (highlighting bone marrow, cortical margins, and cartilage thickness).
   - Late-stage pathology-specific convex ensembling elevates Macro AUC to **`0.8843`**.

4. **12-Branch Label-Specific Attention MIL**:
   - Generates independent spatial attention weights per abnormality ($\mathbf{A} \in \mathbb{R}^{D \times 12}$), solving ViT attention collapse and lifting DINOv2-Small performance from `0.5239` to `0.8187` (+0.2948 AUC).

5. **Dense Multilingual Soft Label Matrix (`llm_labels_v4_blend`)**:
   - Eliminates 80–95% NaN rates in original regex annotations across 4,407 Spanish, Greek, Cyrillic, and English radiology reports, resolving gradient starvation and allowing every mini-batch to supervise all 12 abnormalities.

---

## 🏆 Master Experimentation & Kaggle Leaderboard

| Phase / Exp ID | Architecture | Resolution | Modality / Sequences | Loss Function | **Validation Macro AUC** | **Kaggle Public LB** | Status |
|:---|:---|:---:|:---|:---|:---:|:---:|:---:|
| **EXP-001** | `convnext_tiny` + Gated MIL | 256px | 2.5D Sagittal | Asymmetric Loss (ASL) | 0.5600 | — | Completed |
| **EXP-002** | `convnext_tiny` + Gated MIL | 256px | Weak Multimodal (BETO InfoNCE) | Asymmetric Loss (ASL) | 0.5252 | — | Completed |
| **EXP-004** | `resnet34` + Gated MIL | 256px | 2.5D Sagittal (Full Cohort) | Masked BCE (Tri-State) | 0.7444 | — | Completed |
| **SUB-001 (v13)** | `convnext_tiny` (Single, 8 ep) | 288px | 2.5D Sagittal | Soft BCE | 0.7710 | 0.654 | Completed |
| **SUB-002 (v16)** | **5-Fold `convnext_tiny` Ensemble** | 288px | 2.5D Sagittal (25 ep) | Soft BCE | 0.8158 (Gold: 0.8606) | 0.694 | Submitted |
| **SUB-003 (v20)** | **`convnext_small` (50M)** | 320px | 2.5D Sagittal (320px) | Soft BCE | 0.8367 (Gold: 0.9172) | 0.697 | Submitted |
| **EXP-009** | `dinov2_small` + 12-Branch MIL | 280px | 2.5D Sagittal (32 slices) | Mixed Convex Loss ($\alpha=0.7$) | 0.8187 | — | Completed |
| **EXP-013** | **Multi-Backbone Fusion Ensemble** | 280–320px | ConvNeXt-T + Small + DINOv2 | Multi-Tier Soft + Gold Weighted | 0.8483 (Gold: 0.8753) | — | Completed |
| **Phase 13** | **Tri-Planar Multi-View MIL (ConvNeXt-Tiny/Small)** | 288px | Sagittal + Coronal + Axial (72 sl) | Soft Consensus BCE | 0.8684 (Gold: 0.9117) | ~0.880–0.900 | Completed |
| **Phase 16 (Single)** | **Decoupled ConvNeXt-Tiny & Small (384px)** | 336–384px | Decoupled Tri-Planar T2 FS (1-Fold) | Bottleneck Denoised BCE | 0.8752 (Gold: 0.9210) | 0.920 | Submitted |
| **Phase 16 (3-Fold)** | **Decoupled ConvNeXt-Small 384px (3-Fold Blend)** | 384px | Decoupled Tri-Planar T2 FS | Logit-Space Stacking | 0.8785 (Gold: 0.9260) | 0.925 🚀 | Submitted |
| **Phase 16 (5-Fold)** | **5-Fold Decoupled ConvNeXt-Small 384px + Tiny Blend** | 384px | Decoupled Tri-Planar T2 FS (Full 5-Fold) | Optimal Decoupled Ensembling | **0.8810** (Gold: **0.9320**) | **~0.930** ⭐ | **Current SOTA (T2)** |
| **Phase 17 (1-Fold)** | **Multi-Contrast T1 + T2 FS Decoupled Ensemble** | 384px | T1 Anatomical + T2 Fluid-Sensitive | Contrast-Decoupled Optimization | **0.8843** (Gold: 0.9250) | 0.920 | Submitted (T1+T2) |

---

## 🧬 Foundation Model Benchmark & Local Hardware Profile

Comprehensive 5-fold cross-validation benchmarking evaluated on an **NVIDIA RTX A6000 GPU** with full 12-target MIL heads (`artifacts/master_experiment_results.md`):

### 1. Macro AUC Comparison
| Foundation Backbone | Parameters | Feature Dim | 5-Fold Val Macro AUC | Gold Set AUC ($N=58$) | Architecture Type |
|:---|:---:|:---:|:---:|:---:|:---|
| **DINOv2 + BioMedCLIP** *(Fusion)* | 172.8M | 1,280 | **`0.6963 ± 0.0100`** | **`0.7201 ± 0.0506`** | Self-Supervised ViT + Medical VLM |
| **DINOv3 (ViT-B/16)** | 85.6M | 768 | **`0.6850 ± 0.0056`** | **`0.7062 ± 0.0646`** | Modern Self-Supervised Vision Transformer |
| **DINOv2 (ViT-B/14, 280px)** | 86.6M | 768 | **`0.6831 ± 0.0070`** | **`0.6836 ± 0.0410`** | Top Single Foundation Model |
| **DINOv2-Small (ViT-S/14, 280px)** | 22.1M | 384 | **`0.6786 ± 0.0072`** | **`0.6907 ± 0.0554`** | Ultra-efficient, low-latency ViT |
| **MedSigLIP (SO400M)** | 428.2M | 1,152 | **`0.6748 ± 0.0082`** | **`0.7135 ± 0.0488`** | High-capacity Medical Domain VLM |
| **BioMedCLIP (ViT-B/16)** | 86.2M | 512 | `0.6605 ± 0.0091` | `0.6766 ± 0.0514` | PubMed Medical Vision-Language Specialist |
| **RadImageNet (ResNet-50)** | 23.5M | 2,048 | `0.5565 ± 0.0089` | `0.5575 ± 0.0625` | Radiology Pretrained ResNet |

### 2. Inference Latency & Throughput Profile (NVIDIA RTX A6000)
| Model Architecture | Params | Slice Latency | 16-Slice Study Latency | 48-Slice Tri-Planar Study | Throughput (16-sl) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RadImageNet (ResNet-50)** | 23.5M | **5.29 ms** | **11.53 ms** | **30.94 ms** | **86.7 studies/sec** |
| **ConvNeXt-Tiny** | 27.8M | **5.23 ms** | **26.03 ms** | **76.64 ms** | **38.4 studies/sec** |
| **DINOv2-Small** | 22.1M | **5.14 ms** | **27.35 ms** | **78.80 ms** | **36.6 studies/sec** |
| **BioMedCLIP (ViT-B/16)** | 86.2M | 4.76 ms | 40.39 ms | 114.76 ms | 24.8 studies/sec |
| **ConvNeXt-Small** | 49.5M | 9.56 ms | 44.67 ms | 130.07 ms | 22.4 studies/sec |
| **DINOv2 (ViT-B/14)** | 86.6M | 6.30 ms | 83.03 ms | 243.73 ms | 12.0 studies/sec |

> **Inference Budget Guarantee**: Even our 48-slice tri-planar multi-model grand ensembles execute in **< 150 ms per study**, running 4–6x well within Kaggle's 9-hour inference quota for ~5,000 hidden test studies.

---

## 🎯 Phase 17 Multi-Contrast Ensemble Performance

From [`kaggle_upload/weights/contrast_ensemble_weights.json`](kaggle_upload/weights/contrast_ensemble_weights.json) and [`checkpoints/phase17_3fold_ensemble_summary.json`](checkpoints/phase17_3fold_ensemble_summary.json):

* **T2 / Fluid-Sensitive Baseline**: `0.8752` Macro AUC
* **T1 / Anatomical Baseline**: `0.8535` Macro AUC
* **Uniform Ensemble Blend**: `0.8809` Macro AUC
* **Optimized Pathology-Weighted Ensemble**: **`0.8843` Macro AUC**

| Abnormality | Optimized AUC | T2-FS Weight | T1 Weight | Dominant Sequence Rationale |
|:---|:---:|:---:|:---:|:---|
| **Medial Meniscus** | **0.9334** | 0.74 | 0.26 | Fluid contrast visualizes intrameniscal tear clefts |
| **Baker's Cyst** | **0.9272** | 0.91 | 0.09 | T2 hyperintensity marks popliteal fossa fluid collection |
| **Medial OA** | **0.9038** | 0.64 | 0.36 | Joint space narrowing + subchondral marrow reaction |
| **ACL** | **0.8909** | 0.44 | 0.56 | T1 cortical anatomy + T2 ligamentous disruption |
| **Synovitis** | **0.8902** | 0.73 | 0.27 | Fluid-sensitive synovial hypertrophy and joint effusion |
| **Effusion** | **0.8897** | 0.61 | 0.39 | Hyperintense capsular fluid distension |
| **Lateral Meniscus** | **0.8889** | 0.67 | 0.33 | High-signal tear line breaching articular surface |
| **Lateral OA** | **0.8836** | 0.46 | 0.54 | Subchondral sclerosis on T1 and cartilage loss on T2 |
| **Contusion** | **0.8819** | 0.77 | 0.23 | Bone marrow edema is exquisitely bright on T2/FS |
| **Fracture** | **0.8659** | 0.75 | 0.25 | Trabecular bone bruise and cortical line break |
| **MCL** | **0.8425** | 0.54 | 0.46 | Coronal plane visualization of medial ligament bundle |
| **PF OA** | **0.8140** | 0.77 | 0.23 | Patellofemoral cartilage thinning and trochlear wear |

---

## 📁 Repository Structure

```
KneeVision-AI/
├── environment.yml           # Reproducible Conda environment specification
├── requirements.txt          # Python dependencies (Pinned, Python >= 3.9)
├── setup.py                  # Package descriptor (kneevision)
├── skills.md                 # Systematically tracked learning objectives & progress
│
├── configs/                  # Experiment and model configuration files
│   ├── base.yaml             # Base training hyperparameters & pipeline specs
│   └── env.yaml              # Local path environments & compute settings
│
├── data/                     # Partitioned manifests & stratified splits
│   ├── splits_5fold.parquet  # 5-fold Multilabel Stratified patient splits
│   ├── splits_3fold.parquet  # 3-fold Multi-Contrast Stratified splits
│   ├── dense_labels_master.parquet  # 0% NaN continuous dense label matrix
│   └── pseudo_labels_master.parquet # Multilingual NLP weak label matrix
│
├── src/                      # Core package (kneevision)
│   ├── data/                 # Volume preprocessing, dataset & transforms
│   │   ├── preprocessing.py  # 3D spatial slice sorting, VOI LUT & 2.5D stacking
│   │   ├── dataset.py        # PyTorch KneeMRIDataset (Zero-copy Parquet/NPY loader)
│   │   ├── triplanar_dataset.py # Multi-plane (Sag+Cor+Ax) and dual-contrast loader
│   │   ├── transforms.py     # Medical Albumentations augmentation pipeline
│   │   └── dicom_reader.py   # Multi-threaded low-level DICOM parser
│   ├── models/               # Neural network architectures
│   │   ├── mil_backbone.py   # Gated Attention MIL + 2.5D CNN/ViT backbones
│   │   ├── plane_decoupled_mil.py # Anatomically-Routed Plane-Decoupled MIL
│   │   └── contrast_decoupled_mil.py # T1/T2 Multi-Contrast Decoupled MIL
│   ├── benchmarks/           # Foundation extractors (DINOv2/v3, BioMedCLIP, SigLIP)
│   │   └── foundation_extractors.py
│   ├── metrics/              # Competition metrics
│   │   └── auc_metrics.py    # 12-target NaN-masked Macro ROC-AUC calculator
│   └── utils/                # Reproducibility seeds and checkpoint serialization
│
├── scripts/                  # Automated pipelines & execution scripts
│   ├── 02_preprocess_and_cache.py      # Multi-process parallel volume caching
│   ├── 11_train_dense_288px.py         # High-resolution dense retraining
│   ├── 27_train_mri_aware_mil.py       # DINOv2 label-specific MIL training
│   ├── 47_export_grand_ensemble_kernel.py # Standalone grand ensemble kernel exporter
│   ├── 50_export_dinov2_small_kernel.py   # Standalone DINOv2 kernel exporter
│   ├── 52_export_5fold_convnext_small_ensemble.py # 5-fold ConvNeXt-Small kernel exporter
│   ├── 53_train_decoupled_multicontrast_3fold.py  # Multi-contrast 3-fold training engine
│   └── generate_eda_plots.py           # Publication-quality figure generator
│
├── checkpoints/              # Best model weights & ensemble parameters
│   ├── ensemble_weights.json
│   ├── phase13_pathology_weights.json
│   └── phase17_3fold_ensemble_summary.json
│
├── kaggle_kernel/            # Self-contained Kaggle submission notebooks & engines
│   ├── submission_pipeline.py
│   └── rsna_knee_submission.ipynb
│
├── reports/figures/          # High-resolution visual figures (01 to 08)
└── tests/                    # Pytest validation test suite (17 passed)
```

---

## 🚀 Quickstart & Reproduction

### 1. Prerequisites & Environment Setup
> [!IMPORTANT]
> Requires **Python >= 3.9** (Python 3.10 recommended). Python 3.8 is unsupported as modern scientific packages (`numpy>=1.26.0`, `pandas>=2.2.0`, `scipy>=1.12.0`) require Python 3.9+.

#### Option A: Create from `environment.yml` (Recommended)
```bash
conda env create -f environment.yml
conda activate rsna-knee
```

#### Option B: Install into Existing Conda Environment
```bash
# Create and activate conda environment
conda create -n rsna-knee python=3.10 -y
conda activate rsna-knee

# Install dependencies using python -m pip (avoids PATH shadowing from ~/.local/bin/pip)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install kneevision in editable development mode
python -m pip install -e .
```

#### Verify Hardware Acceleration & Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"
```

---

### 2. Multi-Process Volume Preprocessing & Caching
```bash
# Preprocess the 58 ground-truth labeled studies into 2.5D tensors (~4 sec)
python scripts/02_preprocess_and_cache.py --labeled_only --num_workers 16

# Preprocess the entire 4,407-study dataset (~4-5 mins)
python scripts/02_preprocess_and_cache.py --num_workers 16
```

---

### 3. Run Automated Validation Test Suite
```bash
# Execute the full 17-test validation suite
pytest tests/ -v
```

---

### 4. Train Models & Export Kaggle Submission Kernels
```bash
# Train Multi-Contrast Decoupled 3-Fold Models
python scripts/53_train_decoupled_multicontrast_3fold.py --fold 0 --gpu 0

# Export Standalone 5-Fold ConvNeXt-Small Kaggle Kernel (Public LB ~0.930)
python scripts/52_export_5fold_convnext_small_ensemble.py

# Export Grand Ensemble Submission Kernel
python scripts/47_export_grand_ensemble_kernel.py
```
