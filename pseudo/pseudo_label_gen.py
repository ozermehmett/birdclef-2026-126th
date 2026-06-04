#!/usr/bin/env python3
"""
pseudo_label_gen.py
====================
Run inference with trained SED ONNX models on unlabeled soundscapes
and write soft pseudo labels to CSV.

Steps:
  1. Load fold ONNX models (ensemble)
  2. Infer 12x5s windows per file (~127k total)
  3. Logit-space ensemble (fold mean) + sigmoid
  4. Power transform (gamma=2.0) + dynamic threshold (p95 per class)
  5. Drop low-confidence windows, write high-confidence pseudo labels

Output CSV format:
    filename, start_sec, <species_1>, ..., <species_234>

Usage:
    python pseudo_label_gen.py \
        --onnx_dir      /path/to/outputs/r1_supcon_membank \
        --unlabeled_dir /path/to/unlabeled_sc_cache \
        --comp_dir      /path/to/birdclef-2026 \
        --out           /path/to/outputs/r1_pseudo_unlabeled.csv

Estimated time (A100): ~10-15 min (10592 files x 12 windows = 127k inference steps)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
import onnxruntime as ort
from tqdm.auto import tqdm

SR          = 32_000
N_FFT       = 2048
HOP         = 512
N_MELS      = 256
FMIN        = 20
FMAX        = 16_000
WINDOW_S    = 5
WINDOW_N    = SR * WINDOW_S
N_WINDOWS   = 12
NUM_CLASSES = 234

GAMMA         = 2.0
PERCENTILE    = 95.0
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.50


class MelTransform(torch.nn.Module):
    """GPU mel spectrogram transform matching the training pipeline."""
    def __init__(self, device):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        ).to(device)
        self.to_db = T.AmplitudeToDB(top_db=80).to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        m = self.to_db(self.mel(x))
        for i in range(m.size(0)):
            m[i] = (m[i] - m[i].mean()) / (m[i].std() + 1e-6)
        return m


def load_int16_pt(path):
    """Load int16 .pt cache file and return float32 numpy array."""
    arr = torch.load(path, map_location="cpu", weights_only=True)
    return arr.float().div(32767.0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir",       type=Path, required=True,
                    help="Directory with sed_fold*.onnx files")
    ap.add_argument("--unlabeled_dir",  type=Path, required=True,
                    help="Unlabeled soundscape cache (soundscape/, unlabeled_ss_cache_meta.csv)")
    ap.add_argument("--comp_dir",       type=Path, required=True,
                    help="Competition dir (for sample_submission.csv)")
    ap.add_argument("--out",            type=Path, required=True,
                    help="Output CSV path")
    ap.add_argument("--batch_size",     type=int,   default=128)
    ap.add_argument("--gpu",            type=int,   default=0)
    ap.add_argument("--gamma",          type=float, default=GAMMA)
    ap.add_argument("--percentile",     type=float, default=PERCENTILE)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ONNX models
    print(f"\nLoading ONNX models from: {args.onnx_dir}")
    onnx_files = sorted(args.onnx_dir.glob("sed_fold*.onnx"))
    onnx_files = [p for p in onnx_files
                  if p.name.startswith("sed_fold") and "_ns22" not in p.stem
                  and "_macro" not in p.stem]
    assert len(onnx_files) >= 1, f"No sed_fold*.onnx found in {args.onnx_dir}"
    print(f"  {len(onnx_files)} fold models found")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("CUDAExecutionProvider", {"device_id": args.gpu}), "CPUExecutionProvider"]
    sessions = []
    for f in onnx_files:
        sess = ort.InferenceSession(str(f), sess_options=so, providers=providers)
        if "CUDAExecutionProvider" not in sess.get_providers():
            print(f"  WARNING: {f.name} running on CPU (CUDA not available)")
        sessions.append(sess)

    # Load metadata
    meta = pd.read_csv(args.unlabeled_dir / "unlabeled_ss_cache_meta.csv")
    print(f"\nUnlabeled windows: {len(meta)} ({meta['filename'].nunique()} files)")

    sample_sub = pd.read_csv(args.comp_dir / "sample_submission.csv")
    species_cols = sample_sub.columns[1:].tolist()
    assert len(species_cols) == NUM_CLASSES

    mel_tf = MelTransform(device).to(device)
    all_probs = np.zeros((len(meta), NUM_CLASSES), dtype=np.float32)

    print(f"\nRunning inference...")
    t0 = time.time()

    grouped = meta.groupby("cache_file", sort=False)
    file_order = list(grouped.groups.keys())
    pending_wavs, pending_indices = [], []

    def flush():
        if not pending_wavs:
            return
        batch = np.stack(pending_wavs)
        batch_t = torch.from_numpy(batch).to(device)
        mel = mel_tf(batch_t).unsqueeze(1)
        mel_np = mel.cpu().numpy().astype(np.float32)

        ensemble_logits = None
        for sess in sessions:
            input_name = sess.get_inputs()[0].name
            clip_logits = sess.run(None, {input_name: mel_np})[0]
            if ensemble_logits is None:
                ensemble_logits = clip_logits.astype(np.float32)
            else:
                ensemble_logits += clip_logits.astype(np.float32)
        ensemble_logits /= len(sessions)

        probs = 1.0 / (1.0 + np.exp(-np.clip(ensemble_logits, -50, 50)))
        for i, gi in enumerate(pending_indices):
            all_probs[gi] = probs[i]
        pending_wavs.clear()
        pending_indices.clear()

    for cache_file in tqdm(file_order, desc="Files"):
        wav = load_int16_pt(args.unlabeled_dir / cache_file)
        rows = grouped.get_group(cache_file).reset_index()
        for _, row in rows.iterrows():
            s = int(row["start_sec"]) * SR
            chunk = wav[s:s + WINDOW_N]
            if len(chunk) < WINDOW_N:
                chunk = np.pad(chunk, (0, WINDOW_N - len(chunk)))
            pending_wavs.append(chunk.astype(np.float32))
            pending_indices.append(row["index"])
            if len(pending_wavs) >= args.batch_size:
                flush()
    flush()

    print(f"\nInference done: {(time.time()-t0)/60:.1f} min")
    print(f"  Raw probs: mean={all_probs.mean():.4f}, max={all_probs.max():.4f}")

    # Power transform + threshold
    print("\nApplying power transform + dynamic threshold...")
    probs_sharp = np.power(np.clip(all_probs, 0, 1), args.gamma)
    thr = np.percentile(probs_sharp, args.percentile, axis=0)
    thr = np.clip(thr, MIN_THRESHOLD, MAX_THRESHOLD)
    print(f"  Threshold: mean={thr.mean():.4f}, min={thr.min():.4f}, max={thr.max():.4f}")

    above = (probs_sharp >= thr[None, :]).any(axis=1)
    print(f"  High-confidence windows: {above.sum()}/{len(meta)} ({100*above.mean():.1f}%)")

    # Write CSV
    out_df = pd.DataFrame(probs_sharp, columns=species_cols)
    out_df.insert(0, "start_sec", meta["start_sec"].values)
    out_df.insert(0, "filename",  meta["filename"].values)
    out_df = out_df[above].reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}")
    print(f"  Rows: {len(out_df)}, files: {out_df['filename'].nunique()}")
    print(f"  Mean max prob: {probs_sharp[above].max(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
    