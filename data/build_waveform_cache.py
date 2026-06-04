#!/usr/bin/env python3
"""
build_waveform_cache.py
=======================
Build a local waveform cache from raw BirdCLEF-2026 audio files.
Produces the same directory layout expected by the training script:

    waveform_cache/
        audio/
            000000.pt   -- int16 tensor, 32kHz mono, full clip
            000001.pt
            ...
        soundscape/
            000000.pt   -- int16 tensor, 1920000 samples (60s @ 32kHz)
            ...
        audio_cache_meta.csv      (cache_file, filename, primary_label,
                                   original_idx, n_samples, duration_sec)
        soundscape_cache_meta.csv (cache_file, filename, start_sec,
                                   end_sec, label_list, site)
        soundscape_file_meta.csv  (filename, cache_file)

Estimated time (Colab A100, ~12 vCPU):
    Focal clips  : ~35,000 files  -> ~25-35 min
    Soundscapes  : 66 files       -> ~5 min
    Total        : ~30-45 min
    Disk usage   : ~15-18 GB

Usage:
    python build_waveform_cache.py \
        --comp_dir /content/birdclef-2026 \
        --out_dir  /content/waveform_cache \
        --workers  8
"""

import argparse
import gc
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

SR          = 32_000
SC_DURATION = 60
WINDOW_SEC  = 5
N_WINDOWS   = SC_DURATION // WINDOW_SEC   # 12


def _try_resample(audio, orig_sr):
    """Resample to SR. Uses librosa if available, otherwise scipy.signal.resample_poly."""
    if orig_sr == SR:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig_sr), SR)
        return resample_poly(audio, SR // g, int(orig_sr) // g).astype(np.float32)


def decode_focal_one(args):
    """Decode one focal clip to int16 .pt. Returns metadata dict."""
    idx, src_path, cache_dir = args
    cache_file = f"audio/{idx:06d}.pt"
    dst = Path(cache_dir) / cache_file
    if dst.exists():
        try:
            arr = torch.load(dst, map_location="cpu", weights_only=True)
            return {"idx": idx, "cache_file": cache_file, "n_samples": len(arr), "ok": True}
        except Exception:
            dst.unlink()
    try:
        audio, orig_sr = sf.read(src_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = _try_resample(audio, orig_sr)
        amax = float(np.abs(audio).max())
        if amax > 0:
            audio = audio / amax
        audio_int16 = (audio * 32767).clip(-32767, 32767).astype(np.int16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(audio_int16), dst)
        return {"idx": idx, "cache_file": cache_file, "n_samples": int(len(audio_int16)), "ok": True}
    except Exception as e:
        return {"idx": idx, "cache_file": cache_file, "n_samples": 0, "ok": False, "err": str(e)}


def decode_soundscape_one(args):
    """Decode one full 60s soundscape file to int16 .pt."""
    idx, src_path, cache_dir = args
    cache_file = f"soundscape/{idx:06d}.pt"
    dst = Path(cache_dir) / cache_file
    if dst.exists():
        return {"idx": idx, "cache_file": cache_file, "ok": True}
    try:
        audio, orig_sr = sf.read(src_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = _try_resample(audio, orig_sr)
        target_len = SR * SC_DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        elif len(audio) > target_len:
            audio = audio[:target_len]
        amax = float(np.abs(audio).max())
        if amax > 0:
            audio = audio / amax
        audio_int16 = (audio * 32767).clip(-32767, 32767).astype(np.int16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(audio_int16), dst)
        return {"idx": idx, "cache_file": cache_file, "ok": True}
    except Exception as e:
        return {"idx": idx, "cache_file": cache_file, "ok": False, "err": str(e)}


def build_focal_cache(comp_dir: Path, out_dir: Path, workers: int):
    """Decode all train_audio/*.ogg -> audio/{idx:06d}.pt + audio_cache_meta.csv"""
    print("\n" + "="*60)
    print("PART 1: Focal clips")
    print("="*60)

    train_df = pd.read_csv(comp_dir / "train.csv")
    print(f"train.csv: {len(train_df)} rows")

    audio_dir = comp_dir / "train_audio"
    tasks, missing = [], 0
    for original_idx, row in train_df.iterrows():
        src = audio_dir / str(row["filename"])
        if not src.exists():
            missing += 1
            continue
        tasks.append((original_idx, str(src), str(out_dir)))
    print(f"Tasks: {len(tasks)} (missing: {missing})")

    t0 = time.time()
    results, failed = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(decode_focal_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Focal decode"):
            r = f.result()
            results.append(r)
            if not r["ok"]:
                failed += 1
    print(f"Done in {(time.time()-t0)/60:.1f} min  |  ok: {len(results)-failed}, failed: {failed}")

    ok_by_idx = {r["idx"]: r for r in results if r["ok"]}
    rows = []
    for original_idx, row in train_df.iterrows():
        r = ok_by_idx.get(original_idx)
        if r is None:
            continue
        rows.append({
            "cache_file":    r["cache_file"],
            "filename":      row["filename"],
            "primary_label": row["primary_label"],
            "original_idx":  original_idx,
            "n_samples":     r["n_samples"],
            "duration_sec":  round(r["n_samples"] / SR, 2),
        })
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "audio_cache_meta.csv", index=False)
    print(f"  audio_cache_meta.csv: {len(meta)} rows")
    return meta


def build_soundscape_cache(comp_dir: Path, out_dir: Path, workers: int):
    """Decode all train_soundscapes/*.ogg -> soundscape/{idx:06d}.pt + two CSVs."""
    print("\n" + "="*60)
    print("PART 2: Soundscapes")
    print("="*60)

    sc_dir = comp_dir / "train_soundscapes"
    ogg_files = sorted(sc_dir.glob("*.ogg"))
    print(f"Soundscape files: {len(ogg_files)}")
    if len(ogg_files) == 0:
        print("WARNING: no .ogg files found")
        return None

    tasks = [(i, str(p), str(out_dir)) for i, p in enumerate(ogg_files)]
    t0 = time.time()
    results, failed = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(decode_soundscape_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Soundscape decode"):
            r = f.result()
            results.append(r)
            if not r["ok"]:
                failed += 1
    print(f"Done in {(time.time()-t0)/60:.1f} min  |  ok: {len(results)-failed}, failed: {failed}")

    FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_")
    ok_by_idx = {r["idx"]: r for r in results if r["ok"]}
    file_rows, window_rows = [], []
    for i, ogg_path in enumerate(ogg_files):
        r = ok_by_idx.get(i)
        if r is None:
            continue
        fname = ogg_path.name
        m = FNAME_RE.search(fname)
        site = m.group(2) if m else "unknown"
        file_rows.append({"filename": fname, "cache_file": r["cache_file"], "site": site})
        for w in range(N_WINDOWS):
            start_sec = w * WINDOW_SEC
            window_rows.append({
                "filename":   fname,
                "cache_file": r["cache_file"],
                "start_sec":  start_sec,
                "end_sec":    start_sec + WINDOW_SEC,
                "site":       site,
                "label_list": "",
            })

    file_meta   = pd.DataFrame(file_rows)
    window_meta = pd.DataFrame(window_rows)
    file_meta.to_csv(out_dir / "soundscape_file_meta.csv", index=False)
    window_meta.to_csv(out_dir / "soundscape_cache_meta.csv", index=False)
    print(f"  soundscape_file_meta.csv:  {len(file_meta)} rows")
    print(f"  soundscape_cache_meta.csv: {len(window_meta)} rows")
    return window_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_dir",   type=Path, required=True,
                    help="BirdCLEF-2026 dir (train.csv, train_audio/, train_soundscapes/)")
    ap.add_argument("--out_dir",    type=Path, required=True,
                    help="Output directory for waveform_cache/")
    ap.add_argument("--workers",    type=int,  default=8,
                    help="Parallel decode workers (default: 8)")
    ap.add_argument("--skip_focal", action="store_true")
    ap.add_argument("--skip_sc",    action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audio").mkdir(exist_ok=True)
    (args.out_dir / "soundscape").mkdir(exist_ok=True)

    print(f"Comp dir : {args.comp_dir}")
    print(f"Out dir  : {args.out_dir}")
    print(f"Workers  : {args.workers}")

    if not args.skip_focal:
        build_focal_cache(args.comp_dir, args.out_dir, args.workers)
        gc.collect()
    if not args.skip_sc:
        build_soundscape_cache(args.comp_dir, args.out_dir, args.workers)

    total_bytes = sum(f.stat().st_size for f in args.out_dir.rglob("*.pt"))
    print(f"\nCache built. Total size: {total_bytes / 1e9:.1f} GB")
    print(f"Files: {sum(1 for _ in args.out_dir.rglob('*.pt'))} .pt + 3 CSV")


if __name__ == "__main__":
    main()
    