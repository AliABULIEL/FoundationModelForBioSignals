# test_vitaldb_working.py
"""
Working test for VitalDB using the correct API functions
Based on the actual available functions we discovered
"""

import vitaldb
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def test_find_cases_with_ppg():
    """Find cases that have PPG data using the correct API."""
    print("=" * 60)
    print("TEST 1: FINDING CASES WITH PPG DATA")
    print("=" * 60)

    try:
        # PPG track names to search for
        ppg_tracks = [
            'SNUADC/PLETH',
            'Solar8000/PLETH',
            'Primus/PLETH',
            'Datex-Ohmeda S/5/PLETH'
        ]

        all_ppg_cases = set()

        # Find cases for each PPG track type
        for track in ppg_tracks:
            print(f"\nSearching for cases with {track}...")
            try:
                # find_cases requires track_names parameter
                cases = vitaldb.find_cases([track])
                if cases:
                    print(f"  ✓ Found {len(cases)} cases")
                    all_ppg_cases.update(cases)
                else:
                    print(f"  ✗ No cases found")
            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"\n✓ Total unique cases with PPG: {len(all_ppg_cases)}")

        # Convert to list and sort
        ppg_case_list = sorted(list(all_ppg_cases))

        # Show first few cases
        if ppg_case_list:
            print(f"  First 10 cases: {ppg_case_list[:10]}")

        return ppg_case_list

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def test_load_clinical_data():
    """Load clinical/demographic data using the correct API."""
    print("\n" + "=" * 60)
    print("TEST 2: LOADING CLINICAL/DEMOGRAPHIC DATA")
    print("=" * 60)

    try:
        print("Loading clinical data from VitalDB...")

        # Use load_clinical_data function
        clinical_data = vitaldb.load_clinical_data()

        if clinical_data is not None:
            if isinstance(clinical_data, pd.DataFrame):
                print(f"✓ Loaded clinical data DataFrame")
                print(f"  Shape: {clinical_data.shape}")
                print(f"  Columns ({len(clinical_data.columns)} total):")

                # Show first 20 columns
                for col in clinical_data.columns[:20]:
                    print(f"    - {col}")
                if len(clinical_data.columns) > 20:
                    print(f"    ... and {len(clinical_data.columns) - 20} more columns")

                # Check for demographic columns
                demo_cols = ['age', 'sex', 'height', 'weight', 'bmi', 'caseid']
                available_demo = [col for col in demo_cols if col in clinical_data.columns]

                print(f"\n  Available demographic columns: {available_demo}")

                # Show sample data
                if len(clinical_data) > 0:
                    print(f"\n  First 5 rows of demographic data:")
                    if available_demo:
                        print(clinical_data[available_demo].head())

                return clinical_data
            else:
                print(f"✓ Loaded clinical data (type: {type(clinical_data)})")
                return clinical_data
        else:
            print("✗ No clinical data returned")
            return None

    except Exception as e:
        print(f"❌ Error loading clinical data: {e}")
        return None


def test_load_lab_data():
    """Test loading lab data which might contain additional info."""
    print("\n" + "=" * 60)
    print("TEST 3: LOADING LAB DATA")
    print("=" * 60)

    try:
        print("Loading lab data from VitalDB...")

        lab_data = vitaldb.load_lab_data()

        if lab_data is not None:
            if isinstance(lab_data, pd.DataFrame):
                print(f"✓ Loaded lab data DataFrame")
                print(f"  Shape: {lab_data.shape}")
                print(f"  Columns: {list(lab_data.columns)[:10]}")

                return lab_data
            else:
                print(f"✓ Loaded lab data (type: {type(lab_data)})")
                return lab_data
        else:
            print("✗ No lab data returned")
            return None

    except Exception as e:
        print(f"❌ Error loading lab data: {e}")
        return None


def test_load_case_with_tracks():
    """Test loading a specific case with track names."""
    print("\n" + "=" * 60)
    print("TEST 4: LOADING SPECIFIC CASE DATA")
    print("=" * 60)

    try:
        # First, find a case with PPG
        ppg_track = 'SNUADC/PLETH'
        cases_with_ppg = vitaldb.find_cases([ppg_track])

        if not cases_with_ppg:
            print("✗ No cases found with PPG")
            return None

        case_id = cases_with_ppg[0]
        print(f"Loading case {case_id} with {ppg_track}...")

        # Load case with specific tracks
        case_data = vitaldb.load_case(case_id, [ppg_track])

        if case_data is not None:
            if isinstance(case_data, pd.DataFrame):
                print(f"✓ Loaded case data DataFrame")
                print(f"  Shape: {case_data.shape}")
                print(f"  Columns: {list(case_data.columns)}")

                # Check data statistics
                if ppg_track in case_data.columns:
                    ppg_data = case_data[ppg_track].dropna()
                    print(f"\n  PPG data statistics:")
                    print(f"    Length: {len(ppg_data)}")
                    print(f"    Range: [{ppg_data.min():.2f}, {ppg_data.max():.2f}]")
                    print(f"    Mean: {ppg_data.mean():.2f}")
            else:
                print(f"✓ Loaded case data (type: {type(case_data)})")

            return case_data
        else:
            print("✗ No case data returned")
            return None

    except Exception as e:
        print(f"❌ Error loading case: {e}")
        return None


def test_vital_recs():
    """Test the vital_recs function to get available recordings."""
    print("\n" + "=" * 60)
    print("TEST 5: GETTING VITAL RECORDINGS LIST")
    print("=" * 60)

    try:
        print("Getting list of vital recordings...")

        # Get list of recordings
        recs = vitaldb.vital_recs()

        if recs is not None:
            if isinstance(recs, list):
                print(f"✓ Found {len(recs)} recordings")
                print(f"  First 10: {recs[:10]}")
            elif isinstance(recs, pd.DataFrame):
                print(f"✓ Found recordings DataFrame with shape {recs.shape}")
                print(recs.head())
            else:
                print(f"✓ Found recordings (type: {type(recs)})")

            return recs
        else:
            print("✗ No recordings found")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def create_demographics_mapping(clinical_data: pd.DataFrame, ppg_cases: List) -> Dict:
    """Create a mapping of case IDs to demographics."""
    print("\n" + "=" * 60)
    print("CREATING DEMOGRAPHICS MAPPING")
    print("=" * 60)

    demographics = {}

    if clinical_data is None or len(ppg_cases) == 0:
        print("✗ Missing clinical data or PPG cases")
        return demographics

    # Check if caseid column exists
    if 'caseid' in clinical_data.columns:
        # Filter to only PPG cases
        ppg_clinical = clinical_data[clinical_data['caseid'].isin(ppg_cases)]
        print(f"✓ Found clinical data for {len(ppg_clinical)} PPG cases")

        # Create mapping
        for _, row in ppg_clinical.iterrows():
            case_id = row['caseid']
            demographics[case_id] = {
                'age': row.get('age', -1),
                'sex': row.get('sex', -1),
                'height': row.get('height', -1),
                'weight': row.get('weight', -1),
                'bmi': row.get('bmi', -1)
            }

        # Calculate statistics
        ages = [d['age'] for d in demographics.values() if d['age'] > 0]
        if ages:
            print(f"\nDemographic statistics for PPG cases:")
            print(f"  Age: mean={np.mean(ages):.1f}, std={np.std(ages):.1f}")
            print(f"  Cases with demographics: {len(demographics)}")
    else:
        print("✗ No 'caseid' column in clinical data")
        print(f"  Available columns: {list(clinical_data.columns)[:10]}")

    return demographics


def main():
    """Run all tests with correct API usage."""
    print("\n" + "=" * 80)
    print("VITALDB DATA LOADING - CORRECT API USAGE")
    print("=" * 80)

    # Test 1: Find cases with PPG
    ppg_cases = test_find_cases_with_ppg()

    # Test 2: Load clinical/demographic data
    clinical_data = test_load_clinical_data()

    # Test 3: Load lab data
    lab_data = test_load_lab_data()

    # Test 4: Load specific case
    case_data = test_load_case_with_tracks()

    # Test 5: Get recordings list
    recordings = test_vital_recs()

    # Create demographics mapping if we have the data
    if clinical_data is not None and len(ppg_cases) > 0:
        demographics = create_demographics_mapping(clinical_data, ppg_cases)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n✓ Found {len(ppg_cases)} cases with PPG data")
    print(f"✓ Clinical data available: {clinical_data is not None}")
    print(f"✓ Lab data available: {lab_data is not None}")

    print("\nNext steps:")
    print("1. Use the PPG case list for creating train/val/test splits")
    print("2. Map clinical data to cases for demographic information")
    print("3. Load PPG signals using vitaldb.load_case() with proper track names")

    # Save the PPG case list for later use
    if ppg_cases:
        import json
        from pathlib import Path

        output_dir = Path('data/vitaldb_cache')
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'ppg_cases.json', 'w') as f:
            json.dump(ppg_cases, f)
        print(f"\n✓ Saved PPG case list to {output_dir / 'ppg_cases.json'}")

    return ppg_cases, clinical_data, lab_data


if __name__ == "__main__":
    ppg_cases, clinical_data, lab_data = main()