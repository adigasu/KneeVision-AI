# Skills & Learning Objectives

This project develops transferable, production-grade AI research and engineering skills through the RSNA Knee Abnormality Detection challenge.

## AI / ML

- [x] **Medical image preprocessing & DICOM**: 3D spatial slice sorting along normal vectors, VOI LUT windowing, anisotropic preservation, left/right knee laterality normalization, and 2.5D channel stacking.
- [x] **Multi-label classification**: 12 abnormality targets, Macro AUC loss formulation, Multilabel Stratified 5-Fold Splits (`splits_5fold.parquet`).
- [x] **2D / 2.5D / 3D modeling**: Anisotropic in-plane resolution preservation ($0.4\text{ mm}$ vs $3.0\text{ mm}$ slice thickness), variable depth handling ($11 \to 320$ slices), and memory-efficient batching.
- [x] **Multiple Instance Learning (MIL)**: Gated Attention MIL pooling, 12-branch label-specific attention MIL heads (`src/models/plane_decoupled_mil.py`), and masked padded collation.
- [x] **Multimodal learning**: Vision-language contrastive alignment with Spanish/multilingual radiology reports using symmetric InfoNCE loss ($\tau = 0.07$).
- [x] **Vision-Language Models**: BETO Spanish transformer and PubMed-scale BioMedCLIP text-image encoders with joint projection manifolds.
- [x] **Foundation-model adaptation**: Fine-tuning and adaptation of DINOv2 (`vit_small_patch14_dinov2`), DINOv3, ConvNeXt-Tiny, ConvNeXt-Small (50M, 320px), and ResNet-34 backbones.
- [x] **Multi-view Cross-Attention Fusion**: Inter-plane query-key-value self-attention fusing Sagittal, Coronal, and Axial planes.
- [x] **Anatomically-Routed Plane Decoupling**: Routing orthogonal planes to target heads based on radiological jurisdiction (Sagittal $\to$ ACL/Menisci, Coronal $\to$ MCL/Collaterals, Axial $\to$ PF OA/Effusion).
- [x] **Multi-Contrast MRI Decoupling**: Decoupled modeling and late-stage fusion of fluid-sensitive T2/FS and anatomical T1 sequences.
- [ ] **LoRA / PEFT**: Parameter-efficient low-rank adaptation on small labeled cohorts.

## Efficient AI

- [x] **High-throughput data preprocessing**: Multi-process parallel extraction pipeline processing 89 series/sec (`scripts/02_preprocess_and_cache.py`).
- [x] **Memory-efficient caching**: `uint8` 2.5D tensors yielding 4x disk and RAM savings.
- [x] **Columnar storage**: Snappy-compressed Parquet manifests (`splits_5fold.parquet`, `dense_labels_master.parquet`) with strict type safety.
- [x] **Mixed precision**: PyTorch Automatic Mixed Precision (AMP FP16) training with gradient scaling (`GradScaler`).
- [x] **Accuracy–latency trade-offs for Kaggle deployment**: Profiling per-slice and per-study inference latency across 9 foundation models on NVIDIA RTX A6000; engineered 48-slice tri-planar inference (76–130 ms/study) well within Kaggle's 9-hour execution quota.
- [ ] **Torch.compile & FlashAttention optimization**: Graph capture and kernel fusion for transformer attention blocks.
- [ ] **Model quantization**: Post-training INT8 / FP8 quantization for embedded medical edge devices.

## Evaluation

- [x] **Macro ROC-AUC**: RSNA competition metric computation with NaN-masked target evaluation (`src/metrics/auc_metrics.py`).
- [x] **Per-class evaluation**: Granular diagnostic breakdown across all 12 pathologies and gold standard benchmarking.
- [x] **Cross-plane ablation studies**: Disentangled comparisons of Sagittal vs Coronal vs Axial vs Tri-planar Cross-Attention vs Anatomically-Routed MIL.
- [x] **Error analysis & Ground Truth Paradox**: Disentangled evaluation identifying noise alignment in pseudo-labels versus generalization on the Gold Consensus human subset ($N=58$).
- [x] **Probability calibration & optimization**: Rank averaging, SLSQP stacking meta-learner, and Nelder-Mead per-pathology ensemble weight optimization (`kaggle_upload/weights/contrast_ensemble_weights.json`).

## ML Engineering

- [x] **Leak-free cross-validation**: Patient-level 5-fold Multilabel Stratified Split preventing data leakage.
- [x] **Configuration-driven architecture**: Modular YAML configuration files (`configs/base.yaml`, `configs/env.yaml`).
- [x] **High-performance PyTorch DataLoader**: Zero-copy memory mapping, dynamic batching, and custom MIL collation (`src/data/dataset.py`).
- [x] **Modular test suite**: Automated Pytest validation suite covering datasets, transforms, metrics, model forward passes, and tri-planar architectures (`pytest tests/`).
- [x] **Experiment tracking & logging**: TensorBoardX event generation and structured terminal/JSON metrics (`checkpoints/*_summary.json`).
- [x] **Model checkpointing & serialization**: Tracking peak Validation Macro AUC and Gold Human Consensus AUC with atomic weight saving.
- [x] **Kaggle submission & kernel packaging**: Automated standalone kernel builders (`scripts/50_export_dinov2_small_kernel.py`, `scripts/52_export_5fold_convnext_small_ensemble.py`, `scripts/47_export_grand_ensemble_kernel.py`) achieving verified Kaggle Public Leaderboard scores of **0.694** (v16) and **0.697** (v20).

## Research & Documentation

- [x] **Exploratory Data Analysis**: Publication-grade EDA notebooks and figures (`reports/figures/01_target_prevalences.png` to `08_mil_attention_simulation.png`).
- [x] **Technical architecture guide**: Mathematical and anatomical justification for 2.5D MIL vs 3D convolutions (`docs/data_analysis_and_architecture_guide.md`).
- [x] **Systematic milestone documentation**: Phase 1 through Phase 17 technical walkthroughs, benchmark results, and submission strategies (`docs/`).
- [x] **Hypothesis-driven experimentation logs**: Centralized experiment leaderboard and ablation tracking (`docs/experimentation_log.md`, `artifacts/master_experiment_results.md`).
