# prefetch_vitaldb_all_modalities_fixed.py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# VitalDB API functions (from your documentation)
api_url = "https://api.vitaldb.net"


def get_all_cases():
    """Get all available case IDs."""
    df = pd.read_csv(f"{api_url}/cases")
    return df['caseid'].tolist()


def find_cases_with_track(track_name):
    """Find cases that have a specific track."""
    dftrks = pd.read_csv(f"{api_url}/trks")
    return list(dftrks.loc[dftrks['tname'].str.endswith(track_name), 'caseid'].unique())


def load_case_data(caseid, track_name, interval=1 / 100):
    """Load case data for a specific track."""
    try:
        dftrks = pd.read_csv(f"{api_url}/trks")
        tid_values = dftrks.loc[(dftrks['caseid'] == caseid) &
                                (dftrks['tname'].str.endswith(track_name)), 'tid'].values

        if len(tid_values) == 0:
            return None

        tid = tid_values[0]
        url = f"{api_url}/{tid}"
        dtvals = pd.read_csv(url, na_values='-nan(ind)', dtype=np.float32).values

        if len(dtvals) == 0:
            return None

        # Process the signal (from your load_trk function)
        dtvals[:, 0] /= interval
        nsamp = int(np.nanmax(dtvals[:, 0])) + 1
        ret = np.full(nsamp, np.nan)

        if np.isnan(dtvals[:, 0]).any():  # wave track
            if nsamp != len(dtvals):
                ret = np.take(dtvals[:, 1], np.linspace(0, len(dtvals) - 1, nsamp).astype(np.int64))
            else:
                ret = dtvals[:, 1]
        else:  # numeric track
            for idx, val in dtvals:
                ret[int(idx)] = val

        return ret

    except Exception as e:
        print(f"Error loading case {caseid}: {e}")
        return None


def prefetch_vitaldb_data(modality='ppg', max_cases=6000):
    """Download and cache VitalDB data for specified modality."""

    # Track mapping
    track_map = {
        'ppg': 'PLETH',
        'ecg': 'ECG_II',
    }

    if modality not in track_map:
        raise ValueError(f"Modality {modality} not supported")

    track_name = track_map[modality]
    cache_dir = Path("data/vitaldb_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Finding cases with {track_name}...")

    # Find cases that have this track
    valid_cases = find_cases_with_track(track_name)[:max_cases]

    print(f"Found {len(valid_cases)} cases with {track_name}")

    # Download and cache
    failed = []
    skipped = []
    cached = []

    for case_id in tqdm(valid_cases, desc=f"Caching {modality.upper()}"):
        cache_file = cache_dir / f"{case_id}_{modality}_full.npy"

        if cache_file.exists():
            skipped.append(case_id)
            continue



        try:
            # Download signal using the API
            signal = load_case_data(case_id, track_name)

            if signal is not None and len(signal) > 0:
                # Remove NaN values
                signal = signal[~np.isnan(signal)]

                # Only save if we have enough data (at least 20 seconds at 100Hz)
                min_samples = 20 * 100
                if len(signal) >= min_samples:
                    np.save(cache_file, signal.astype(np.float32))
                    cached.append(case_id)
                else:
                    failed.append((case_id, "Too short"))

        except Exception as e:
            failed.append((case_id, str(e)))

    # Report results
    print(f"\n📊 Results for {modality.upper()}:")
    print(f"  ✅ Cached: {len(cached)} new files")
    print(f"  ⏭️  Skipped: {len(skipped)} (already cached)")
    print(f"  ❌ Failed: {len(failed)}")

    if failed[:5]:
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