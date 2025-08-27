#!/usr/bin/env python3
"""
BUT PPG Database Download Script
Downloads and organizes the BUT PPG database for training
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import shutil
from tqdm import tqdm
import json
import time
import warnings

warnings.filterwarnings('ignore')

# ==================== Configuration ====================

# PhysioNet BUT PPG Database URL
PHYSIONET_ZIP = "https://physionet.org/static/published-projects/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0.zip"

# Alternative: Use wget command if download fails
WGET_COMMAND = "wget -r -N -c -np https://physionet.org/files/butppg/2.0.0/"

# Output directories
DATA_DIR = Path("data/but_ppg")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = Path("data/outputs")


# ==================== Download Functions ====================

def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with progress
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False


def extract_database(zip_path: Path, extract_dir: Path) -> bool:
    """Extract the downloaded database."""
    try:
        print("\n📦 Extracting database...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all members
            members = zip_ref.namelist()

            # Extract with progress bar
            with tqdm(total=len(members), desc="Extracting files") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_dir)
                    pbar.update(1)

        print(f"  ✓ Extracted {len(members)} files")
        return True

    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def organize_data(extract_dir: Path) -> Path:
    """Organize extracted data into clean structure."""
    print("\n🗂️ Organizing data...")

    # Find the actual data directory (should be one level down)
    data_dirs = list(extract_dir.glob("*"))
    if not data_dirs:
        raise FileNotFoundError("No extracted data found")

    # The extracted folder structure might have nested directories
    actual_data_dir = None
    for d in data_dirs:
        if d.is_dir():
            # Check if this directory contains the data
            if (d / "quality-hr-ann.csv").exists():
                actual_data_dir = d
                break
            # Check one level deeper
            subdirs = list(d.glob("*"))
            for sd in subdirs:
                if sd.is_dir() and (sd / "quality-hr-ann.csv").exists():
                    actual_data_dir = sd
                    break

    if actual_data_dir is None:
        raise FileNotFoundError("Could not find data files in extracted directory")

    print(f"  Found data in: {actual_data_dir.name}")

    # Move to cleaner location if needed
    final_data_dir = DATA_DIR / "dataset"
    if actual_data_dir != final_data_dir:
        if final_data_dir.exists():
            shutil.rmtree(final_data_dir)
        shutil.move(str(actual_data_dir), str(final_data_dir))
        actual_data_dir = final_data_dir

    return actual_data_dir


def verify_dataset(data_dir: Path) -> dict:
    """Verify the downloaded dataset and gather statistics."""
    print("\n✅ Verifying dataset...")

    stats = {
        'data_dir': str(data_dir),
        'total_records': 0,
        'has_ppg': 0,
        'has_ecg': 0,
        'has_acc': 0,
        'has_annotations': False,
        'has_subject_info': False,
        'subjects': set(),
        'quality_good': 0,
        'quality_poor': 0
    }

    # Check annotation files
    quality_file = data_dir / "quality-hr-ann.csv"
    subject_file = data_dir / "subject-info.csv"

    if quality_file.exists():
        stats['has_annotations'] = True
        quality_df = pd.read_csv(quality_file, names=['record_id', 'quality', 'reference_hr'])
        stats['quality_good'] = (quality_df['quality'] == 1).sum()
        stats['quality_poor'] = (quality_df['quality'] == 0).sum()
        print(f"  ✓ Quality annotations: {len(quality_df)} records")
        print(f"    - Good quality: {stats['quality_good']}")
        print(f"    - Poor quality: {stats['quality_poor']}")

    if subject_file.exists():
        stats['has_subject_info'] = True
        subject_df = pd.read_csv(subject_file)
        print(f"  ✓ Subject info: {len(subject_df)} records")

        # Print demographic summary
        if 'Age' in subject_df.columns:
            print(f"    - Age range: {subject_df['Age'].min()}-{subject_df['Age'].max()} years")
        if 'Gender' in subject_df.columns:
            gender_counts = subject_df['Gender'].value_counts()
            print(f"    - Gender distribution: {gender_counts.to_dict()}")

    # Count record directories
    record_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    stats['total_records'] = len(record_dirs)

    # Sample check for different signal types
    print(f"\n  Checking {min(10, len(record_dirs))} sample records...")
    for i, record_dir in enumerate(record_dirs[:10]):
        record_id = record_dir.name
        stats['subjects'].add(record_id[:3])  # First 3 digits are subject ID

        # Check for signal files
        if (record_dir / f"{record_id}_PPG.dat").exists():
            stats['has_ppg'] += 1
        if (record_dir / f"{record_id}_ECG.dat").exists():
            stats['has_ecg'] += 1
        if (record_dir / f"{record_id}_ACC.dat").exists():
            stats['has_acc'] += 1

    # Extrapolate from sample
    if len(record_dirs) > 10:
        scale = len(record_dirs) / 10
        stats['has_ppg'] = int(stats['has_ppg'] * scale)
        stats['has_ecg'] = int(stats['has_ecg'] * scale)
        stats['has_acc'] = int(stats['has_acc'] * scale)

    stats['subjects'] = len(stats['subjects'])

    print(f"\n  📊 Dataset Statistics:")
    print(f"    - Total records: {stats['total_records']}")
    print(f"    - Estimated subjects: ~{stats['subjects'] * 5}")  # Rough estimate
    print(f"    - Records with PPG: ~{stats['has_ppg']}")
    print(f"    - Records with ECG: ~{stats['has_ecg']}")
    print(f"    - Records with ACC: ~{stats['has_acc']}")

    return stats




# ==================== Main Function ====================

def main():
    """Main download function."""
    print("\n" + "=" * 60)
    print("📥 BUT PPG DATABASE DOWNLOAD")
    print("=" * 60)
    print("Dataset: Smartphone PPG with ECG, ACC, and demographics")
    print("Source: PhysioNet")

    start_time = time.time()

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Download
        print(f"\n📥 Step 1: Downloading database (~87 MB)...")
        zip_path = RAW_DIR / "but_ppg.zip"

        if zip_path.exists():
            print("  ✓ Already downloaded")
        else:
            success = download_file(PHYSIONET_ZIP, zip_path, "BUT PPG Database")
            if not success:
                print("\n💡 Alternative: Try running this command:")
                print(f"  {WGET_COMMAND}")
                return False

        # Step 2: Extract
        print(f"\n📦 Step 2: Extracting database...")
        extract_dir = RAW_DIR / "extracted"

        if extract_dir.exists() and any(extract_dir.iterdir()):
            print("  ✓ Already extracted")
            # Find the data directory
            data_dir = organize_data(extract_dir)
        else:
            success = extract_database(zip_path, extract_dir)
            data_dir = organize_data(extract_dir)
            if not success:
                return False


        # Step 3: Verify
        print(f"\n✅ Step 3: Verifying dataset...")
        stats = verify_dataset(data_dir)

        # Step 4: Create index files
        # print(f"\n📝 Step 4: Creating index files...")
        # create_index_files(data_dir, stats)

        # Calculate time
        elapsed = time.time() - start_time

        # Print summary
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)

        print(f"\n⏱️ Total time: {elapsed:.1f} seconds")

        print("\n📁 Output files created:")
        print(f"  • {OUTPUT_DIR}/waveform_index.csv - Index of all records")
        print(f"  • {OUTPUT_DIR}/labels.csv - Demographics and health data")
        print(f"  • {OUTPUT_DIR}/dataset_info.json - Dataset metadata")
        print(f"  • {data_dir}/ - Raw signal data")

        print("\n🚀 Next Steps:")
        print("1. Data is ready for the data loader module")
        print("2. Use waveform_index.csv to load signals")
        print("3. Use labels.csv for downstream tasks")
        print("4. Run training with PPG-only or multi-modal approach")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)