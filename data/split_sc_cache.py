#!/usr/bin/env python3
"""
split_sc_cache.py
==================
Split the waveform_cache into labeled and unlabeled soundscape sets:
  - Labeled   (66 files)     -> waveform_cache/soundscape/          (for training)
  - Unlabeled (~10592 files) -> unlabeled_sc_cache/soundscape/      (for pseudo labeling)

Moves existing .pt files; does not re-decode from source audio.

Usage:
    python split_sc_cache.py \
        --cache_dir     /path/to/waveform_cache \
        --unlabeled_dir /path/to/unlabeled_sc_cache \
        --labels_csv    /path/to/birdclef-2026/train_soundscapes_labels.csv

Estimated time: ~2-5 min (file move + CSV rewrite)
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

FNAME_RE   = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_")
SC_DURATION = 60
WINDOW_SEC  = 5
N_WINDOWS   = SC_DURATION // WINDOW_SEC   # 12


def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",     type=Path, required=True,
                    help="Existing waveform_cache (audio/, soundscape/, *.csv)")
    ap.add_argument("--unlabeled_dir", type=Path, required=True,
                    help="Destination for unlabeled soundscape cache (new directory)")
    ap.add_argument("--labels_csv",    type=Path, required=True,
                    help="train_soundscapes_labels.csv (defines the 66 labeled files)")
    args = ap.parse_args()

    # Load labeled filenames
    labels_df = pd.read_csv(args.labels_csv)
    labeled_fnames = set(labels_df["filename"].astype(str).unique())
    print(f"Labeled files (from labels CSV): {len(labeled_fnames)}")

    # Load existing soundscape file metadata
    file_meta_path = args.cache_dir / "soundscape_file_meta.csv"
    file_meta = pd.read_csv(file_meta_path)
    print(f"Total soundscape files in cache: {len(file_meta)}")

    # Split
    file_meta["is_labeled"] = file_meta["filename"].isin(labeled_fnames)
    labeled_meta   = file_meta[file_meta["is_labeled"]].reset_index(drop=True)
    unlabeled_meta = file_meta[~file_meta["is_labeled"]].reset_index(drop=True)
    print(f"  Labeled  : {len(labeled_meta)}")
    print(f"  Unlabeled: {len(unlabeled_meta)}")

    if len(labeled_meta) == 0:
        print("ERROR: no filenames matched between labels CSV and cache.")
        print(f"  Labels CSV sample : {list(labeled_fnames)[:3]}")
        print(f"  Cache sample      : {file_meta['filename'].head(3).tolist()}")
        return

    if not (50 <= len(labeled_meta) <= 100):
        print(f"WARNING: labeled count {len(labeled_meta)} is outside the expected range (~66)")

    # Move unlabeled .pt files to new directory
    print(f"\nMoving unlabeled .pt files to {args.unlabeled_dir}/soundscape/ ...")
    (args.unlabeled_dir / "soundscape").mkdir(parents=True, exist_ok=True)

    moved, failed = 0, 0
    for new_idx, (_, row) in enumerate(unlabeled_meta.iterrows()):
        src = args.cache_dir / row["cache_file"]
        dst = args.unlabeled_dir / f"soundscape/{new_idx:06d}.pt"
        if not src.exists():
            failed += 1
            continue
        shutil.move(str(src), str(dst))
        moved += 1
        if moved % 2000 == 0:
            print(f"  Moved {moved}/{len(unlabeled_meta)}")
    print(f"  Total moved: {moved}, failed: {failed}")

    # Write unlabeled metadata (one row per 5s window)
    unlabeled_rows = []
    for new_idx, (_, row) in enumerate(unlabeled_meta.iterrows()):
        if new_idx >= moved:
            continue
        fname = row["filename"]
        m = FNAME_RE.search(fname)
        site = m.group(2) if m else "unknown"
        for w in range(N_WINDOWS):
            unlabeled_rows.append({
                "filename":   fname,
                "cache_file": f"soundscape/{new_idx:06d}.pt",
                "start_sec":  w * WINDOW_SEC,
                "end_sec":    (w + 1) * WINDOW_SEC,
                "site":       site,
            })
    unlab_meta = pd.DataFrame(unlabeled_rows)
    unlab_csv  = args.unlabeled_dir / "unlabeled_ss_cache_meta.csv"
    unlab_meta.to_csv(unlab_csv, index=False)
    print(f"  Wrote {unlab_csv}: {len(unlab_meta)} rows")

    # Rewrite labeled metadata in original cache dir
    print(f"\nRewriting labeled metadata in {args.cache_dir} ...")
    labeled_meta[["filename", "cache_file", "site"]].to_csv(file_meta_path, index=False)
    print(f"  soundscape_file_meta.csv: {len(labeled_meta)} rows")

    window_meta_path = args.cache_dir / "soundscape_cache_meta.csv"
    old_window_meta  = pd.read_csv(window_meta_path)
    new_window_meta  = old_window_meta[
        old_window_meta["filename"].isin(labeled_fnames)
    ].reset_index(drop=True)
    new_window_meta.to_csv(window_meta_path, index=False)
    print(f"  soundscape_cache_meta.csv: {len(new_window_meta)} rows")

    # Disk usage summary
    cache_size     = dir_size(args.cache_dir) / 1e9
    unlabeled_size = dir_size(args.unlabeled_dir) / 1e9
    print(f"\nDisk usage:")
    print(f"  {args.cache_dir}: {cache_size:.1f} GB  (training cache)")
    print(f"  {args.unlabeled_dir}: {unlabeled_size:.1f} GB  (pseudo label cache)")
    print(f"  Total: {cache_size + unlabeled_size:.1f} GB")
    print("\nDone.")


if __name__ == "__main__":
    main()
    