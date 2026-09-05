# Skills & Learning Objectives

This project is designed to develop transferable AI research and engineering
skills through the RSNA Knee Abnormality Detection problem.

## AI / ML

- [x] Medical image preprocessing & DICOM (3D spatial sorting along normal vector, VOI LUT windowing, 2.5D stacking)
- [x] Multi-label classification (12 abnormality targets, Macro AUC formulation, Multilabel Stratified Splits)
- [x] 2D / 2.5D / 3D modeling (Anisotropic resolution preservation, variable depth handling, VRAM optimization)
- [x] Multiple Instance Learning (Gated Attention MIL pooling with masked padded collation)
- [ ] Multimodal learning (Vision-Language Contrastive Alignment with Spanish radiology reports)
- [ ] Vision-Language Models (BiomedCLIP / BETO / Clinical-Longformer adaptation)
- [ ] Foundation-model adaptation (ConvNeXt-V2 / Swin / EVA-02 fine-tuning)
- [ ] LoRA / PEFT (Low-Rank parameter efficient tuning for 58 labeled studies)
- [ ] Knowledge distillation & Model ensemble

## Efficient AI

- [x] High-throughput data preprocessing (89 series/sec multi-process extraction)
- [x] Memory-efficient caching (uint8 2.5D tensors with 4x disk/RAM savings)
- [x] Columnar storage (Snappy-compressed Parquet with strict type safety)
- [x] Mixed precision (AMP FP16 training with GradScaler)
- [ ] Torch.compile & FlashAttention optimization
- [ ] Model quantization (INT8 / FP8 inference)
- [ ] Accuracy–latency trade-offs for Kaggle deployment

## Evaluation

- [x] Macro ROC-AUC (RSNA competition metric with NaN-masked target evaluation)
- [x] Per-class evaluation (12 abnormality breakdown & positive prevalence benchmarking)
- [ ] Cross-plane ablation studies (Sagittal vs Coronal vs Axial vs Tri-planar)
- [ ] Error analysis on false positives / false negatives
- [ ] Probability calibration & temperature scaling

## ML Engineering

- [x] Leak-free cross-validation (5-fold Multilabel Stratified Split on patient studies)
- [x] Configuration-driven data pipeline (Flexible image resolutions & slice depths)
- [x] High-performance PyTorch DataLoader (Zero-copy memory mapping & custom MIL collate)
- [x] Modular test suite (Pytest integration for datasets, transforms, metrics, and preprocessing)
- [ ] Experiment tracking (W&B / MLflow integration)
- [x] Model checkpointing & serialization (Best Macro AUC weight saving)
- [ ] Kaggle submission & inference kernel packaging

## Research & Documentation

- [x] Comprehensive exploratory data analysis (EDA notebooks & publication figures)
- [x] Technical architecture guide (2.5D vs 3D mathematical & clinical justification)
- [x] Systematic progress walkthroughs & milestone artifacts
- [x] Hypothesis-driven experimentation logs (Centralized experiment leaderboard in docs/experimentation_log.md)
