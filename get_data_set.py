#!/usr/bin/env python3
"""
BUT-PPG (PhysioNet) downloader & converter to .npy

- Downloads the zip (if missing)
- Unpacks to a temp dir
- Detects subject folders by scanning for *.hea files
- Converts WFDB records:
    <subject>/<subject>_ECG.dat/.hea  ->  <out>/<subject>/ECG.npy   (shape [L, C])
    <subject>/<subject>_PPG.dat/.hea  ->  <out>/<subject>/PPG.npy   (shape [L, C])

Usage:
  python3 get_data_set.py --root data/PPG

Notes:
- Requires `wfdb` (`pip install wfdb`)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
import sys

import requests
import numpy as np
from tqdm import tqdm

try:
    import wfdb  # pip install wfdb
except Exception:
    print("ERROR: This dataset is WFDB-formatted. Please install:\n  pip install wfdb")
    sys.exit(1)

BUT_PPG_URL = (
    "https://physionet.org/static/published-projects/"
    "butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0.zip"
)

def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def read_wfdb_signal(record_stem: Path) -> np.ndarray:
    """
    record_stem = path without suffix, e.g. /.../100001/100001_ECG
    returns float32 array of shape [L, C]
    """
    rec = wfdb.rdrecord(str(record_stem))
    sig = np.asarray(rec.p_signal, dtype=np.float32)  # [L, C]
    return sig

def find_subject_dirs(root: Path) -> list[Path]:
    """
    Find subject directories by locating any *.hea files and collecting their parents.
    Example matched path:
      .../brno-.../100001/100001_ECG.hea
    """
    hea_files = list(root.rglob("*.hea"))
    if not hea_files:
        # Help debug
        print("\n[DEBUG] No .hea files found. Top-level entries after unzip:")
        for p in sorted(root.iterdir()):
            print("  -", p.name)
        raise FileNotFoundError("No WFDB headers (*.hea) found after extraction.")
    subj_dirs = sorted({p.parent for p in hea_files})
    return subj_dirs

def convert_subject(subj_dir: Path, out_dir: Path, overwrite: bool) -> int:
    """
    Convert available ECG/PPG WFDB files in a subject dir to .npy.
    Returns number of files written.
    """
    written = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert ECG if present
    ecg_hea = next(subj_dir.glob("*_ECG.hea"), None)
    if ecg_hea:
        out_path = out_dir / "ECG.npy"
        if overwrite or not out_path.exists():
            try:
                ecg_stem = ecg_hea.with_suffix("")  # rm .hea
                ecg = read_wfdb_signal(ecg_stem)
                np.save(out_path, ecg)
                written += 1
            except Exception as e:
                print(f"Warning: failed ECG in {subj_dir.name}: {e}")

    # Convert PPG if present
    ppg_hea = next(subj_dir.glob("*_PPG.hea"), None)
    if ppg_hea:
        out_path = out_dir / "PPG.npy"
        if overwrite or not out_path.exists():
            try:
                ppg_stem = ppg_hea.with_suffix("")  # rm .hea
                ppg = read_wfdb_signal(ppg_stem)
                np.save(out_path, ppg)
                written += 1
            except Exception as e:
                print(f"Warning: failed PPG in {subj_dir.name}: {e}")

    return written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/PPG",
                    help="Output folder for processed .npy files (and zip if downloaded).")
    ap.add_argument("--zip", default=None,
                    help="Optional path to existing zip; default is <root>.zip.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Do not download; expect the zip to exist.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing .npy files.")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Keep the downloaded zip (default: keep).")
    args = ap.parse_args()

    out_root = Path(args.root)
    out_root.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip) if args.zip else out_root.with_suffix(".zip")

    # Download if needed
    if not args.skip_download and not zip_path.exists():
        print("Downloading BUT-PPG …")
        download_file(BUT_PPG_URL, zip_path)
    else:
        if zip_path.exists():
            print(f"Zip found at: {zip_path} (skip download)")
        else:
            print("ERROR: --skip-download given but zip not found:", zip_path)
            sys.exit(1)

    # Unpack & process
    print("Unpacking …")
    total_written = 0
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        shutil.unpack_archive(str(zip_path), str(tmpdir))

        # Example top: tmp/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/100001/...
        top_entries = list(tmpdir.iterdir())
        if len(top_entries) == 1 and top_entries[0].is_dir():
            dataset_root = top_entries[0]
        else:
            dataset_root = tmpdir  # fallback: search from the unzip root

        # Find subjects by .hea files
        try:
            subject_dirs = find_subject_dirs(dataset_root)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        # Convert each subject
        for subj in subject_dirs:
            out_subj = out_root / subj.name
            total_written += convert_subject(subj, out_subj, overwrite=args.overwrite)

    print(f"Done! Wrote {total_written} files to {out_root}\n(Zip retained at {zip_path})")

if __name__ == "__main__":
    main()
