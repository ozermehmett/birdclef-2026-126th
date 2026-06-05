# BirdCLEF+ 2026 — 126th Place Solution

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)  
**Task:** Acoustic species identification in the Pantanal, South America — 234 species, multi-label  
**Metric:** Macro AUC (non-S22 soundscape windows)  
**Final Score:** 0.949 Public LB · 0.944 Private LB · 126th / 4085 teams  
**Hardware:** Google Colab Pro+ (A100 for training, T4 fallback)  
**Inference:** Kaggle CPU (90-minute limit)

---

## Pipeline Overview

```
Raw Audio (.ogg)
      |
      v
build_waveform_cache.py        -- decode + int16 cache (~15 GB NVMe)
      |
      |------ Labeled Soundscapes (66 files, 708 labeled 5s windows)
      |------ Unlabeled Soundscapes (~10,592 files)
                    |
                    v
            split_sc_cache.py  -- labeled / unlabeled split
                    |
      +-------------+----------------+
      |                              |
      v                              v
train_sed.py (R1-R8)          pseudo_label_gen.py
EfficientNet-B0 / ConvNeXt    R1 ONNX -> soft pseudo labels
+ Perch-v2 distillation              |
      |                              |
      +-------------+----------------+
                    |
                    v
         Kaggle Inference Notebook (yukiZ pipeline, public)
         Perch-v2 -> ProtoSSM + MLP Probes + SED ensemble
         rank blend (60/40 Proto/SED)
                    |
                    v
              submission.csv
```

---

## Problem Setup

- 234 species, 28 of which are invisible to Perch (unmapped classes)
- 66 fully-labeled training soundscape files (708 windows total)
- ~10,592 unlabeled soundscape files used for pseudo labeling
- S22 site excluded from validation metric due to known label noise
- CPU-only inference on Kaggle (90-minute wall-time limit)

---

## Model Architecture

### SED Model

```
Input: raw waveform (5s, 32kHz mono)
   |
   v
MelSpectrogram  (256 mels, n_fft=2048, hop=512, fmin=20Hz, fmax=16kHz)
AmplitudeToDB   (top_db=80)
Per-sample normalization  (zero mean, unit std)
   |
   v
EfficientNet-B0  (tf_efficientnet_b0.ns_jft_in1k, pretrained ImageNet-21k)
Feature map: (B, C=1280, F', T')
   |
   +--------- Distillation branch (always active) ---------+
   |           GAP + Linear -> 1536-d                       |
   |           MSE vs frozen Perch-v2 embedding             |
   |           (ONNX, CUDA I/O binding, no CPU roundtrip)   |
   |                                                        |
   +--------- SED branch (stop-gradient from backbone) -----+
               GeMFreqPool (learnable p, init=3.0)
                   |
               Dropout(0.25) + Linear(C->512) + ReLU + Dropout(0.5)
                   |
              +----+----+
              |         |
             att       cla       (1D Conv, kernel=1)
              |         |
              +--tanh--+
              softmax over T
                   |
             weighted sum -> clip_logits   (B, 234)
             frame_max  = cla.max(dim=T)   (B, 234)

Loss = 0.5*BCE(clip_logits, y) + 0.5*BCE(frame_max, y)
     + alpha * MSE(distill_emb, perch_emb)
```

### Key Design Decisions

**Perch distillation** — Frozen Perch-v2 as teacher via ONNX with CUDA I/O binding. The student backbone learns rich bioacoustic representations without gradient through the teacher. This was the single biggest boost over training from scratch.

**Stop-gradient on SED branch** — The SED attention head does not backprop into the backbone. Only the distillation MSE updates the backbone weights. This prevents the classification head from competing with the distillation objective and stabilizes training.

**GeMFreqPool** — Generalized mean pooling with learnable exponent p (init=3.0, clamped >= 1.0) over the frequency axis. Sharper than average-pool, softer than max-pool. Lets the model adapt to each species' vocalization frequency profile.

**non-S22 AUC as checkpoint metric** — S22 is a recording site with significant label noise. Using full macro AUC as the checkpoint metric caused overfitting to noisy labels. Excluding S22 gives a cleaner validation signal.

**Focal-Soundscape MixUp** — Training focal recordings are mixed with labeled soundscape windows (beta distribution, alpha=0.4). This forces the model to handle overlapping calls and background textures simultaneously, improving generalization to the soundscape evaluation format.

---

## Training Rounds (R1 to R8)

### R1 — SupCon + Memory Bank (Baseline)

The first complete training run. Implemented Supervised Contrastive loss using a memory bank of Perch embeddings. The memory bank stores past embeddings from previous batches and uses hard negative mining against the bank.

- KD type: `supcon_membank` (SupCon distillation with 4096-entry memory bank)
- Loss: BCE (clip + frame_max) + SupCon distillation
- Data: clean focal clips + 66 labeled soundscapes
- Epochs: 25, batch: 64, LR: 5e-4 with warmup + cosine

| Fold | ns22 |
|------|------|
| F0 | 0.9121 |
| F1 | 0.8901 |
| F2 | 0.9112 |
| F3 | 0.8875 |
| F4 | 0.9224 |
| Mean | 0.9047 |

Notable: Fold 1 significantly weaker than others. This became a target for R2.

---

### R2 — Noisy Student (Pseudo Labeling Round 1)

Used R1's 5-fold ONNX ensemble to generate soft pseudo labels on unlabeled soundscapes. Then fine-tuned R1 checkpoints using the pseudo-labeled data mixed with original clean data.

- KD type: `mse` (standard MSE distillation)
- Data: pseudo-labeled unlabeled soundscapes + clean focal + labeled soundscapes
- Pseudo label filtering: power transform (gamma=2.0) + percentile threshold (p95 per class)
- Warm-start: loaded R1 fold checkpoints, continued training

| Fold | ns22 | vs R1 |
|------|------|-------|
| F0 | 0.8903 | -0.022 |
| F1 | 0.9352 | **+0.045** |
| F2 | 0.9056 | -0.006 |
| F3 | 0.8881 | +0.001 |
| F4 | 0.9228 | +0.004 |
| Mean | 0.9084 | +0.004 |

Key observation: Pseudo labels helped dramatically on Fold 1 (+0.045) but hurt or had no effect on Folds 0 and 2. Pattern is fold-split dependent — the validation soundscapes in Fold 1 had distribution more similar to the pseudo-labeled files. R2 Fold 1 (0.9352) became the best single model across all rounds.

Also observed: R2 improved Insecta AUC substantially in Fold 1 (0.9381 vs ~0.85 in R1), suggesting pseudo labels helped with the harder invisible/rare classes.

---

### Loss Function Ablation Experiments (between R2 and R5)

Several intermediate runs testing different loss configurations on single folds. None improved over the standard BCE + MSE distillation baseline:

**ASL (Asymmetric Loss)** — Tried `LOSS_TYPE="asl"` with asymmetric focusing parameters (gamma_neg=4, gamma_pos=0). In theory better for multi-label imbalanced classification. In practice, converged to similar or slightly worse ns22 than plain BCE. Added training instability.

**Focal Loss** — Tried `LOSS_TYPE="focal"` (gamma=2.0). No meaningful improvement over BCE. The distillation loss already provides implicit hard-example weighting.

**InfoNCE distillation** — Tried `KD_TYPE="infonce"` instead of MSE. Contrastive distillation should improve embedding quality. Slower convergence, similar final ns22. Not worth the added complexity for 25-epoch training.

**Hard InfoNCE + MSE combo** — `KD_TYPE="mse_hard_infonce"`. Combined MSE with InfoNCE focusing on hardest 50% negatives. Marginally better embedding quality (cosine similarity metrics) but same downstream ns22.

**SpecMix / CutMix frequency** — Tried `USE_SPECMIX=True` and `USE_CUTMIX_FREQ=True`. These augmentations blend or swap frequency bands between samples. No improvement, added compute cost.

**Time shift augmentation** — `USE_TIME_SHIFT=True` (circular shift ±1s). No measurable effect on ns22.

**Label smoothing** — `LABEL_SMOOTH=0.05`. Slight degradation.

All of these were dropped. Final config: `LOSS_TYPE="bce"`, `KD_TYPE="mse"`, no SpecMix, no label smoothing.

---

### R5 — SWA (Stochastic Weight Averaging)

Clean data training (no pseudo labels) with Stochastic Weight Averaging enabled from epoch 30 onwards. SWA averages model weights from multiple checkpoints to find a flatter loss minimum.

- KD type: `mse`
- Data: clean focal + labeled soundscapes (same as R1)
- SWA: manual implementation, state accumulation from epoch 30

| Fold | ns22 | vs R1 |
|------|------|-------|
| F0 | 0.9015 | -0.011 |
| F1 | 0.8788 | -0.011 |
| F2 | 0.9102 | -0.001 |
| F3 | 0.8825 | -0.005 |
| F4 | 0.9239 | **+0.002** |
| Mean | 0.8994 | -0.005 |

SWA helped only on Fold 4 (best fold overall). Generally worse than R1. The marginal improvement on F4 was enough to select R5 for that slot in the final ensemble.

---

### R6 — ConvNeXt-Tiny (Clean Baseline)

First ConvNeXt experiment. Switched backbone from EfficientNet-B0 to ConvNeXt-Tiny to get architecturally diverse predictions. ConvNeXt processes images differently (depthwise conv, inverted bottleneck) and may capture different spectro-temporal patterns.

- Backbone: `convnext_tiny` (pretrained)
- Data: clean focal + labeled soundscapes
- Everything else identical to R1

| Fold | ns22 | vs R1 B0 |
|------|------|----------|
| F0 | 0.9000 | -0.012 |
| F1 | 0.8773 | -0.013 |
| F2 | 0.8673 | -0.044 |
| F3 | 0.9048 | +0.017 |
| F4 | 0.8973 | -0.025 |
| Mean | 0.8893 | -0.015 |

ConvNeXt generally worse than B0 but complementary in pattern (stronger on some folds, weaker on others). The key insight from experimentation: ConvNeXt is more valuable as a pseudo-label generator than a final ensemble model — its architectural diversity produces different soft labels that B0 can then learn from, without paying the inference cost at submission time.

---

### R7 — ConvNeXt-Tiny + Pseudo Labels

Retrained ConvNeXt using R1 pseudo labels. Primary purpose: produce a high-quality, architecturally diverse set of pseudo labels for R8 — not to use ConvNeXt directly in the final SED ensemble. ConvNeXt's different inductive bias (depthwise conv, inverted bottleneck, larger receptive field) means it attends to different spectro-temporal patterns than B0. Its predictions as soft targets give R8 access to knowledge that a B0-only pseudo loop would miss.

- Backbone: ConvNeXt-Tiny
- Data: R1 pseudo-labeled unlabeled soundscapes + clean focal
- Warm-start: R6 ConvNeXt checkpoints
- Primary role: pseudo label generator for R8, not final ensemble model

| Fold | ns22 | vs R6 |
|------|------|-------|
| F0 | 0.9129 | +0.013 |
| F1 | 0.8692 | -0.008 |
| F2 | 0.8890 | +0.022 |
| F3 | 0.9146 | +0.010 |
| F4 | 0.9077 | +0.010 |
| Mean | 0.8987 | +0.009 |

Pseudo labels consistently helped ConvNeXt. R7 F3 (0.9146) is the best ConvNeXt score and became the candidate for Fold 3 in early hybrid ensemble attempts.

---

### R8 — EfficientNet-B0 + R7 Pseudo Labels

Final training round. Used R7 ConvNeXt ONNX predictions as pseudo labels (not R1 B0). The idea: ConvNeXt captures different acoustic patterns, so using its predictions as teacher gives B0 access to complementary knowledge — architectural diversity through pseudo labeling without the inference cost.

- Backbone: EfficientNet-B0
- Data: R7 ConvNeXt pseudo labels + clean focal
- Warm-start: R1 B0 checkpoints

| Fold | ns22 | vs R1 |
|------|------|-------|
| F0 | 0.8917 | -0.020 |
| F1 | 0.8975 | +0.007 |
| F2 | 0.9101 | -0.001 |
| F3 | 0.9075 | **+0.020** |
| F4 | 0.9216 | -0.001 |
| Mean | 0.9057 | +0.001 |

R8 improved Fold 3 by +0.020 over R1. This was the only fold where R8 clearly won among B0 models. R8 F3 became the Fold 3 choice in the final ensemble.

---

## Final Ensemble Strategy

After completing all runs, we picked the best model per fold (B0 only):

| Fold | Winner | Score | Runner-up |
|------|--------|-------|-----------|
| F0 | R1 | 0.9121 | R5: 0.9015 |
| F1 | R2 | 0.9352 | R8: 0.8975 |
| F2 | R1 | 0.9112 | R5: 0.9102 |
| F3 | R8 | 0.9075 | R2: 0.8881 |
| F4 | R5 | 0.9239 | R2: 0.9228 |
| **Mean** | | **0.9180** | |

**Why B0 only?** Early submissions used a hybrid (ConvNeXt for F0/F3, B0 for F1/F2/F4) which had slightly better OOF average (0.9196 vs 0.9180). However, ConvNeXt inference is 3-4x slower than B0 on CPU, causing Kaggle notebook timeout (90-minute limit). The pure B0 ensemble runs in ~87 minutes.

Kaggle Dataset: [ozermehmet/birdclef2026-b0-best-per-fold-onnx](https://www.kaggle.com/datasets/ozermehmet/birdclef2026-b0-best-per-fold-onnx)

---

## Submission History

| Submission | Score | Notes |
|-----------|-------|-------|
| Hybrid ConvNeXt+B0, 5 folds | TIMEOUT (5h+) | ConvNeXt too slow on CPU |
| Hybrid ConvNeXt+B0, 4 folds (F0 dropped) | 0.949 | 87 min, survived |
| B0 Best-Per-Fold, 5 folds | **0.949** | Final, same score |

Score was stable at 0.949 across all SED configurations tested. Swapping ConvNeXt ↔ B0, adding/removing folds — nothing moved the score. This confirmed the bottleneck was not in the SED branch.

---

## What We Controlled vs What We Didn't

This solution uses the public ProtoSSM + MLP Probes inference pipeline (yukiZ, 0.928 baseline notebook). Our contribution was only the SED branch:

- **We controlled:** SED model architecture, training technique, fold selection, ONNX export
- **We did not control:** ProtoSSM training, MLP probe tuning, prior tables, rank blending weights, post-processing (TAX smoothing, rank-aware scaling)

At 0.949, the SED branch was saturated. The real score ceiling was determined by the ProtoSSM/MLP pipeline side. Future work: participate in building the inference pipeline from scratch, or use a higher-scoring baseline (0.949+).

---

## Repository Structure

```
birdclef-2026-126th/
├── README.md
├── .gitignore
├── training/
│   └── train_sed.py             -- main SED training (R1-R8, all configs)
├── data/
│   ├── build_waveform_cache.py  -- decode .ogg -> int16 .pt cache
│   └── split_sc_cache.py        -- labeled / unlabeled soundscape split
├── pseudo/
│   └── pseudo_label_gen.py      -- soft pseudo label generation (ONNX ensemble)
├── utils/
│   └── onnx_perch_teacher.py    -- ONNX Perch-v2 teacher (CUDA I/O binding)
└── experiments/
    ├── experiment_log.md        -- chronological experiment notes
    └── fold_scores.json         -- all fold scores per round
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchaudio timm onnxruntime soundfile librosa \
            pandas tqdm scikit-learn scipy
```

### 1. Build Waveform Cache

```bash
python data/build_waveform_cache.py \
    --comp_dir /path/to/birdclef-2026 \
    --out_dir  /path/to/waveform_cache \
    --workers  8
```

Estimated time: 30-45 min on A100 Colab (~35k focal clips + 66 soundscapes).

### 2. Split Labeled / Unlabeled Soundscapes

```bash
python data/split_sc_cache.py \
    --cache_dir     /path/to/waveform_cache \
    --unlabeled_dir /path/to/unlabeled_sc_cache \
    --labels_csv    /path/to/birdclef-2026/train_soundscapes_labels.csv
```

### 3. Train SED (single fold, e.g. R1 config)

```bash
python training/train_sed.py \
    --fold 0 \
    --gpu  0 \
    --epochs 25
```

Key env variables:
```bash
export BIRDCLEF_COMP=/path/to/birdclef-2026
export BIRDCLEF_CACHE=/path/to/waveform_cache
export PERCH_ONNX_PATH=/path/to/perch_v2_no_dft.onnx
export BIRDCLEF_OUT=/path/to/outputs
export DRIVE_BACKUP_DIR=/content/drive/MyDrive/birdclef/outputs
```

Override training config via CLI:
```bash
# ASL loss experiment
python training/train_sed.py --fold 0 --gpu 0 --loss_type asl

# InfoNCE distillation
python training/train_sed.py --fold 0 --gpu 0 --kd_type infonce

# SWA enabled (R5)
python training/train_sed.py --fold 0 --gpu 0 --use_swa --swa_start_ep 15
```

### 4. Generate Pseudo Labels

```bash
python pseudo/pseudo_label_gen.py \
    --onnx_dir      /path/to/outputs/r1_supcon_membank \
    --unlabeled_dir /path/to/unlabeled_sc_cache \
    --comp_dir      /path/to/birdclef-2026 \
    --out           /path/to/outputs/r1_pseudo_unlabeled.csv \
    --gamma 2.0 \
    --percentile 95
```

---

## Acknowledgments

- **Tucker Arrants** ([bc2026-distilled-sed](https://www.kaggle.com/code/tuckerarrants/bc2026-distilled-sed-public)) — SED architecture, waveform cache format, GeMFreqPool, attention head design
- **yukiZ** — ProtoSSM + MLP probes inference pipeline (used as the inference notebook base)
- **Yaroslav Kholmirzayev** — EoS.4 post-processing (rank-aware scaling, TAX smoothing, sonotype mirroring)
- **Google Research** — Perch-v2 bird vocalization classifier (teacher model)
