# prefetch_vitaldb_all_modalities.py
import vitaldb
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def prefetch_vitaldb_data(modality='ppg', max_cases=6000):
    """Download and cache VitalDB data for specified modality."""

    # Track mapping
    track_map = {
        'ppg': 'PLETH',
        'ecg': 'ECG_II',  # or 'ECG_V5' if you prefer
    }

    if modality not in track_map:
        raise ValueError(f"Modality {modality} not supported")

    track_name = track_map[modality]
    cache_dir = Path("data/vitaldb_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pre-caching {modality.upper()} data (track: {track_name})...")

    # Get all cases
    all_cases = vitaldb.list_cases()[:max_cases]

    # Filter cases that have the required track
    valid_cases = []
    print("Filtering cases with required signal...")
    for case_id in tqdm(all_cases, desc="Checking"):
        tracks = vitaldb.list_tracks(case_id)
        if tracks and track_name in tracks:
            valid_cases.append(case_id)

    print(f"Found {len(valid_cases)} cases with {track_name}")

    # Download and cache
    failed = []
    skipped = []

    for case_id in tqdm(valid_cases, desc=f"Caching {modality.upper()}"):
        cache_file = cache_dir / f"{case_id}_{modality}_full.npy"

        if cache_file.exists():
            skipped.append(case_id)
            continue

        try:
            # Download signal
            signal = vitaldb.load_case(case_id, [track_name])

            if signal is not None and len(signal) > 0:
                # Handle 2D arrays (multiple channels)
                if signal.ndim == 2:
                    signal = signal[:, 0]  # Take first channel

                # Remove NaN values
                signal = signal[~np.isnan(signal)]

                # Only save if we have enough data (at least 20 seconds)
                min_samples = 20 * 100  # 20 seconds at 100Hz
                if len(signal) >= min_samples:
                    # Save as float32 to save space
                    np.save(cache_file, signal.astype(np.float32))
                else:
                    failed.append((case_id, "Too short"))

        except Exception as e:
            failed.append((case_id, str(e)))

    # Report results
    print(f"\n📊 Results for {modality.upper()}:")
    print(f"  ✅ Cached: {len(valid_cases) - len(failed) - len(skipped)} new files")
    print(f"  ⏭️  Skipped: {len(skipped)} (already cached)")
    print(f"  ❌ Failed: {len(failed)}")

    if failed[:5]:  # Show first 5 failures
        print("\n  First few failures:")
        for case_id, reason in failed[:5]:
            print(f"    Case {case_id}: {reason}")

    # Check cache statistics
    cache_files = list(cache_dir.glob(f"*_{modality}_full.npy"))
    if cache_files:
        cache_size = sum(f.stat().st_size for f in cache_files) / (1024 ** 3)
        print(f"\n📁 {modality.upper()} cache: {len(cache_files)} files, {cache_size:.2f} GB")


def prefetch_all_modalities():
    """Cache both PPG and ECG data."""

    print("=" * 60)
    print("VITALDB COMPLETE CACHING")
    print("=" * 60)

    # Cache PPG
    print("\n1. Caching PPG data...")
    prefetch_vitaldb_data('ppg', max_cases=6000)

    # Cache ECG
    print("\n2. Caching ECG data...")
    prefetch_vitaldb_data('ecg', max_cases=6000)

    print("\n" + "=" * 60)
    print("🎉 All modalities cached successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='all',
                        choices=['ppg', 'ecg', 'all'],
                        help='Which modality to cache')
    parser.add_argument('--max-cases', type=int, default=6000,
                        help='Maximum number of cases to cache')

    args = parser.parse_args()

    if args.modality == 'all':
        prefetch_all_modalities()
    else:
        prefetch_vitaldb_data(args.modality, args.max_cases)