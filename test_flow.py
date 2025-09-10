# load_vitaldb_real_demographics.py
"""
Load REAL demographics from VitalDB using the proper method.
The issue is that load_clinical_data() returns structure but no data.
We need to use a different approach to get the actual demographics.
"""

import vitaldb
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
import json
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')


def load_real_vitaldb_demographics():
    """
    Load real demographics from VitalDB using the correct method.
    """
    print("=" * 80)
    print("LOADING REAL VITALDB DEMOGRAPHICS")
    print("=" * 80)

    cache_dir = Path('data/vitaldb_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # First, let's understand what data is actually available
    print("\n1. Checking available data sources...")

    # Method 1: Use vitaldb.read_csv to get the clinical data
    print("\n   Trying vitaldb.read_csv for clinical data...")
    try:
        # VitalDB stores clinical data in a specific format
        # Try to read the clinical data CSV directly
        clinical_url = "https://api.vitaldb.net/clinical"  # This is the typical URL structure
        clinical_data = vitaldb.read_csv('clinical')

        if clinical_data is not None and not clinical_data.empty:
            print(f"   ✓ Loaded clinical data: {clinical_data.shape}")
            print(f"   Columns: {list(clinical_data.columns)[:10]}")

            # Save it
            clinical_data.to_csv(cache_dir / 'clinical_data_real.csv', index=False)
            print(f"   ✓ Saved to {cache_dir / 'clinical_data_real.csv'}")

            return clinical_data
    except Exception as e:
        print(f"   ✗ Could not load via read_csv: {e}")

    # Method 2: Download the dataset information
    print("\n   Trying to get dataset information...")
    try:
        # Get the dataset description which should include demographics
        dataset_info = vitaldb.dataset()

        if dataset_info is not None:
            print(f"   ✓ Got dataset info")

            # Check what's in it
            if hasattr(dataset_info, 'shape'):
                print(f"   Shape: {dataset_info.shape}")

            # Save for inspection
            if isinstance(dataset_info, pd.DataFrame):
                dataset_info.to_csv(cache_dir / 'dataset_info.csv', index=False)
                print(f"   Saved to {cache_dir / 'dataset_info.csv'}")

                # Check if it has demographic columns
                demo_cols = ['age', 'sex', 'height', 'weight', 'bmi']
                available = [col for col in demo_cols if col in dataset_info.columns]
                if available:
                    print(f"   ✓ Found demographics: {available}")
                    return dataset_info
    except Exception as e:
        print(f"   ✗ Could not get dataset info: {e}")

    # Method 3: Use the API directly with proper parameters
    print("\n   Trying direct API access with parameters...")
    try:
        # VitalDB API might need specific parameters
        # Try loading with all available data
        import requests

        # VitalDB public API endpoint
        api_url = "https://api.vitaldb.net/cases"

        print(f"   Fetching from {api_url}...")
        response = requests.get(api_url)

        if response.status_code == 200:
            cases_data = pd.DataFrame(response.json())
            print(f"   ✓ Got cases data: {cases_data.shape}")
            print(f"   Columns: {list(cases_data.columns)[:15]}")

            # Save it
            cases_data.to_csv(cache_dir / 'cases_data_api.csv', index=False)
            print(f"   ✓ Saved to {cache_dir / 'cases_data_api.csv'}")

            # Check for demographics
            demo_cols = ['age', 'sex', 'height', 'weight', 'bmi', 'asa']
            available = [col for col in demo_cols if col in cases_data.columns]
            if available:
                print(f"   ✓ Found demographics: {available}")

                # Show statistics
                for col in available:
                    if col in ['age', 'height', 'weight', 'bmi']:
                        valid = cases_data[col].dropna()
                        if len(valid) > 0:
                            print(f"     {col}: mean={valid.mean():.1f}, std={valid.std():.1f}")

                return cases_data
        else:
            print(f"   ✗ API returned status {response.status_code}")

    except Exception as e:
        print(f"   ✗ API access failed: {e}")

    return None


def load_demographics_for_ppg_cases():
    """
    Load demographics specifically for cases that have PPG data.
    """
    print("\n" + "=" * 80)
    print("LOADING DEMOGRAPHICS FOR PPG CASES")
    print("=" * 80)

    cache_dir = Path('data/vitaldb_cache')

    # Step 1: Load PPG cases
    print("\n1. Loading PPG cases...")
    ppg_track = 'SNUADC/PLETH'
    ppg_cases = vitaldb.find_cases([ppg_track])
    print(f"   ✓ Found {len(ppg_cases)} PPG cases")

    # Step 2: Load demographics
    print("\n2. Loading demographics...")
    demographics = load_real_vitaldb_demographics()

    if demographics is not None and not demographics.empty:
        print(f"\n3. Filtering demographics for PPG cases...")

        # Make sure caseid column exists
        if 'caseid' in demographics.columns:
            # Filter for PPG cases only
            ppg_demographics = demographics[demographics['caseid'].isin(ppg_cases)].copy()

            print(f"   ✓ Found demographics for {len(ppg_demographics)} PPG cases")

            # Save filtered demographics
            ppg_demographics.to_csv(cache_dir / 'ppg_demographics.csv', index=False)
            print(f"   ✓ Saved to {cache_dir / 'ppg_demographics.csv'}")

            # Show statistics
            print("\n   PPG Cases Demographics:")
            demo_cols = ['age', 'sex', 'height', 'weight', 'bmi']
            for col in demo_cols:
                if col in ppg_demographics.columns:
                    if col == 'sex':
                        counts = ppg_demographics[col].value_counts()
                        print(f"   {col}: {counts.to_dict()}")
                    else:
                        valid = ppg_demographics[col].dropna()
                        if len(valid) > 0:
                            print(f"   {col}: mean={valid.mean():.1f}, std={valid.std():.1f}, "
                                  f"n={len(valid)}/{len(ppg_demographics)}")

            return ppg_demographics
        else:
            print(f"   ✗ No 'caseid' column found in demographics")
            print(f"   Available columns: {list(demographics.columns)}")
    else:
        print("   ✗ Could not load demographics")

    return None


def test_vitaldb_file_structure():
    """
    Test to understand VitalDB file structure better.
    """
    print("\n" + "=" * 80)
    print("TESTING VITALDB FILE STRUCTURE")
    print("=" * 80)

    # Check what files are available
    print("\n1. Checking available files/tracks...")

    # Get a sample case
    ppg_cases = vitaldb.find_cases(['SNUADC/PLETH'])
    if ppg_cases:
        case_id = ppg_cases[0]
        print(f"\n   Testing with case {case_id}...")

        # Try to get all available tracks for this case
        try:
            # Use vital_trks to get all tracks for a case
            all_tracks = vitaldb.vital_trks(case_id)
            print(f"   ✓ Available tracks for case {case_id}:")
            for track in all_tracks[:20]:  # Show first 20
                print(f"     - {track}")
            if len(all_tracks) > 20:
                print(f"     ... and {len(all_tracks) - 20} more")

            # Check if there are any demographic/clinical tracks
            clinical_keywords = ['age', 'sex', 'height', 'weight', 'bmi', 'demo', 'clinical']
            clinical_tracks = [t for t in all_tracks if any(k in t.lower() for k in clinical_keywords)]

            if clinical_tracks:
                print(f"\n   ✓ Found potential demographic tracks: {clinical_tracks}")
            else:
                print(f"\n   ✗ No obvious demographic tracks found")

        except Exception as e:
            print(f"   ✗ Could not get tracks: {e}")


def main():
    """
    Main function to load real VitalDB demographics.
    """
    print("\n" + "=" * 80)
    print("VITALDB REAL DEMOGRAPHICS LOADER")
    print("=" * 80)

    # First, test the file structure
    test_vitaldb_file_structure()

    # Then load demographics
    ppg_demographics = load_demographics_for_ppg_cases()

    if ppg_demographics is not None and not ppg_demographics.empty:
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"\n✓ Loaded real demographics for {len(ppg_demographics)} PPG cases")
        print(f"✓ Data saved to data/vitaldb_cache/ppg_demographics.csv")

        return ppg_demographics
    else:
        print("\n" + "=" * 80)
        print("TROUBLESHOOTING NEEDED")
        print("=" * 80)
        print("\nThe demographics are not loading properly.")
        print("\nPossible solutions:")
        print("1. Check if you need to authenticate with VitalDB")
        print("2. The demographics might be in a different format")
        print("3. Try downloading the full VitalDB dataset manually from:")
        print("   https://vitaldb.net/dataset")
        print("\n4. You can also try this code to explore the API:")
        print("""
import vitaldb
import pandas as pd

# List all available data
print(dir(vitaldb))

# Try to get clinical data with authentication if needed
# vitaldb.login('your_username', 'your_password')  # If you have an account

# Explore what's available
help(vitaldb.load_clinical_data)
help(vitaldb.read_csv)
""")

        return None


if __name__ == "__main__":
    demographics = main()