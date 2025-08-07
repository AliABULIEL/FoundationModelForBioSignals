#!/usr/bin/env python3
"""
Example downloader for the BUT-PPG 2.0 dataset on PhysioNet.
Adjust the URL / unpack logic for any other corpus.
"""
import tarfile, shutil, tempfile, requests, os, argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r, open(dst, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True)
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/PPG", help="output folder")
    args = p.parse_args()
    url = ("https://physionet.org/static/published-projects/"
           "butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0.zip")
    zip_path = Path(args.root).with_suffix(".zip")
    print("Downloading BUT-PPG …")
    download_file(url, zip_path)

    print("Unpacking …")
    tmp = tempfile.TemporaryDirectory()
    shutil.unpack_archive(zip_path, tmp.name)
    src = Path(tmp.name) / "but-ppg-2.0.0" / "data"
    dst_root = Path(args.root)
    for subj in src.iterdir():               # already one folder per subject
        dst = dst_root / subj.name
        dst.mkdir(parents=True, exist_ok=True)
        for raw in subj.glob("*.csv"):
            arr = np.loadtxt(raw, delimiter=",")        # → shape [L]
            np.save(dst / (raw.stem + ".npy"), arr.astype("float32"))
    print("Done! Folder tree ready for `main.py pretrain`.")

if __name__ == "__main__":
    main()
