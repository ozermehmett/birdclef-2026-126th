# Experiment Log — BirdCLEF+ 2026

## Overview
- **Metric:** Macro AUC (non-S22 soundscape windows)
- **Hardware:** Colab Pro+ A100 / T4
- **Final score:** 0.949 Public LB · 0.944 Private LB · 135th / 4243 teams

---

## R1 — SupCon + Memory Bank

**Date:** ~May 20  
**Backbone:** EfficientNet-B0 (`tf_efficientnet_b0.ns_jft_in1k`)  
**Data:** Clean focal recordings + 66 labeled soundscapes  
**Technique:** Supervised Contrastive Loss + 4096-entry memory bank

| Fold | ns22 |
|------|------|
| 0 | 0.9121 |
| 1 | 0.8901 |
| 2 | 0.9112 |
| 3 | 0.8875 |
| 4 | 0.9224 |

**Result:** Fold 1 notably weaker. Net contribution of SupCon over plain BCE unclear.

---

## R2 — Noisy Student

**Backbone:** EfficientNet-B0  
**Technique:** Fine-tuned from R1 checkpoints using R1 pseudo labels

| Fold | ns22 |
|------|------|
| 0 | 0.8903 |
| 1 | **0.9352** |
| 2 | 0.9056 |
| 3 | 0.8881 |
| 4 | 0.9228 |

**Result:** Fold 1 is the best single-fold score across all rounds. R2 F1 selected for final ensemble.

---

## R5 — SWA (Stochastic Weight Averaging)

**Backbone:** EfficientNet-B0  
**Technique:** Clean data, SWA from epoch 30

| Fold | ns22 |
|------|------|
| 0 | 0.9015 |
| 1 | 0.8788 |
| 2 | 0.9102 |
| 3 | 0.8825 |
| 4 | 0.9239 |

**Result:** Only Fold 4 improved over R1. SWA provides marginal gains overall.

---

## R6 — ConvNeXt-Tiny (Clean)

**Backbone:** ConvNeXt-Tiny  
**Technique:** Clean data, standard training

| Fold | ns22 |
|------|------|
| 0 | 0.9000 |
| 1 | 0.8773 |
| 2 | 0.8673 |
| 3 | 0.9048 |
| 4 | 0.8973 |

**Result:** Generally below B0. Primary value: architectural diversity for pseudo label generation.

---

## R7 — ConvNeXt-Tiny (Pseudo Labels)

**Backbone:** ConvNeXt-Tiny  
**Data:** R1 pseudo labels + clean focal  
**Primary role:** Pseudo label generator for R8 (not used in final SED ensemble)

| Fold | ns22 |
|------|------|
| 0 | 0.9129 |
| 1 | 0.8692 |
| 2 | 0.8890 |
| 3 | 0.9146 |
| 4 | 0.9077 |

**Result:** Pseudo labels consistently improved ConvNeXt. R7 predictions used as soft targets for R8.

---

## R8 — EfficientNet-B0 (R7 Pseudo Labels)

**Backbone:** EfficientNet-B0  
**Data:** R7 ConvNeXt pseudo labels + clean focal

| Fold | ns22 |
|------|------|
| 0 | 0.8917 |
| 1 | 0.8975 |
| 2 | 0.9101 |
| 3 | 0.9075 |
| 4 | 0.9216 |

**Result:** Fold 3 improved by +0.020 over R1 — the only fold where R8 clearly wins among B0 models.

---

## Final Ensemble — B0 Best-Per-Fold

**Kaggle Dataset:** `ozermehmet/birdclef2026-b0-best-per-fold-onnx`

| Fold | Model | ns22 |
|------|-------|------|
| 0 | R1 | 0.9121 |
| 1 | R2 | 0.9352 |
| 2 | R1 | 0.9112 |
| 3 | R8 | 0.9075 |
| 4 | R5 | 0.9239 |

**Why B0 only?** ConvNeXt hybrid (F0/F3) had slightly better OOF average (0.9196 vs 0.9180) but caused Kaggle notebook timeout (90-min limit). Pure B0 runs in ~87 min.

---

## Submission History

| Submission | Score | Notes |
|-----------|-------|-------|
| Hybrid ConvNeXt+B0, 5 folds | TIMEOUT | ConvNeXt too slow on CPU |
| Hybrid ConvNeXt+B0, 4 folds (F0 dropped) | 0.949 | 87 min |
| B0 Best-Per-Fold, 5 folds | **0.949** | Final submission |

---

## Key Observation

Score was fixed at 0.949 regardless of SED configuration changes. Swapping backbones, changing fold count, different fold combinations — none moved the score. The bottleneck was in the ProtoSSM/MLP probes side of the inference pipeline, not the SED branch.

**Lesson learned:** Future work should also involve building or tuning the inference pipeline, not only the SED model.

---

## Abandoned Experiments

| Experiment | Reason abandoned |
|-----------|-----------------|
| ConvNeXt in final SED ensemble | 90-min CPU timeout |
| MLP probes NoPCA + (512,256) | Ran out of time before deadline |
| min_pos=2 probes | Caused timeout |
| Broader SWA usage | Marginal gains |
| Different mel spec parameters | Not tested |
| ASL loss | No improvement over BCE |
| InfoNCE distillation | No improvement, slower convergence |
| SpecMix / CutMix frequency | No improvement |
| Label smoothing | Slight degradation |
