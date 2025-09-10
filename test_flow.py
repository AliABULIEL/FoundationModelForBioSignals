# load_vitaldb_real_api.py
"""
Load REAL VitalDB demographics using the official API as described in the paper.
Based on the Scientific Data paper (Lee et al., 2022).
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import vitaldb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class VitalDBRealDataLoader:
    """Load real VitalDB data including demographics using the official API."""

    def __init__(self, cache_dir: str = 'data/vitaldb_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Official VitalDB API endpoints from the paper
        self.api_base = "https://api.vitaldb.net"
        self.cases_url = f"{self.api_base}/cases"
        self.trks_url = f"{self.api_base}/trks"
        self.labs_url = f"{self.api_base}/labs"

    def load_clinical_information(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the clinical information file which contains demographics.
        According to the paper, this includes 74 perioperative clinical parameters.
        """
        print("=" * 80)
        print("LOADING REAL CLINICAL INFORMATION FROM VITALDB API")
        print("=" * 80)

        cache_file = self.cache_dir / 'clinical_information.csv'

        # Check cache first
        if not force_reload and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            clinical_df = pd.read_csv(cache_file)
            print(f"✓ Loaded {len(clinical_df)} cases from cache")
            return clinical_df

        # Download from API
        print(f"\nDownloading from: {self.cases_url}")
        print("This may take a moment...")

        try:
            # Download the clinical information CSV
            response = requests.get(self.cases_url)

            if response.status_code == 200:
                # Save the raw response
                with open(cache_file, 'wb') as f:
                    f.write(response.content)

                # Load as DataFrame
                clinical_df = pd.read_csv(cache_file)

                print(f"✓ Successfully downloaded {len(clinical_df)} cases")
                print(f"✓ Saved to: {cache_file}")

                # Show available columns
                print(f"\nColumns ({len(clinical_df.columns)} total):")
                demo_cols = ['caseid', 'subjectid', 'age', 'sex', 'height', 'weight', 'bmi', 'asa']

                for col in demo_cols:
                    if col in clinical_df.columns:
                        print(f"  ✓ {col}")
                    else:
                        print(f"  ✗ {col} (missing)")

                # Show other available columns
                other_cols = [col for col in clinical_df.columns if col not in demo_cols]
                if len(other_cols) > 10:
                    print(f"\n  Plus {len(other_cols)} other columns including:")
                    for col in other_cols[:10]:
                        print(f"    - {col}")
                    print(f"    ... and {len(other_cols) - 10} more")

                return clinical_df

            else:
                print(f"✗ API returned status {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ Error downloading clinical data: {e}")
            return None

    def load_track_list(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the track list to identify which cases have PPG data.
        """
        print("\n" + "=" * 80)
        print("LOADING TRACK LIST")
        print("=" * 80)

        cache_file = self.cache_dir / 'track_list.csv'

        if not force_reload and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            tracks_df = pd.read_csv(cache_file)
            print(f"✓ Loaded {len(tracks_df)} tracks from cache")
            return tracks_df

        print(f"Downloading from: {self.trks_url}")

        try:
            response = requests.get(self.trks_url)

            if response.status_code == 200:
                with open(cache_file, 'wb') as f:
                    f.write(response.content)

                tracks_df = pd.read_csv(cache_file)
                print(f"✓ Downloaded {len(tracks_df)} tracks")
                print(f"✓ Saved to: {cache_file}")

                return tracks_df
            else:
                print(f"✗ API returned status {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def find_ppg_cases(self, tracks_df: pd.DataFrame) -> List[int]:
        """
        Find cases that have PPG data from the track list.
        """
        print("\n" + "=" * 80)
        print("FINDING PPG CASES")
        print("=" * 80)

        # PPG track names from the paper
        ppg_tracks = [
            'SNUADC/PLETH',
            'Solar8000/PLETH',
            'Primus/PLETH',
            'Datex-Ohmeda S/5/PLETH'
        ]

        ppg_cases = set()

        for track_name in ppg_tracks:
            if 'tname' in tracks_df.columns:
                cases = tracks_df[tracks_df['tname'] == track_name]['caseid'].unique()
                ppg_cases.update(cases)
                print(f"  {track_name}: {len(cases)} cases")

        ppg_cases = sorted(list(ppg_cases))
        print(f"\n✓ Total unique PPG cases: {len(ppg_cases)}")

        # Save PPG cases list
        ppg_file = self.cache_dir / 'ppg_cases.json'
        with open(ppg_file, 'w') as f:
            json.dump(ppg_cases, f)
        print(f"✓ Saved to: {ppg_file}")

        return ppg_cases

    def get_ppg_demographics(self, clinical_df: pd.DataFrame, ppg_cases: List[int]) -> pd.DataFrame:
        """
        Extract demographics for PPG cases only.
        """
        print("\n" + "=" * 80)
        print("EXTRACTING DEMOGRAPHICS FOR PPG CASES")
        print("=" * 80)

        # Filter clinical data for PPG cases
        ppg_clinical = clinical_df[clinical_df['caseid'].isin(ppg_cases)].copy()

        print(f"✓ Found clinical data for {len(ppg_clinical)} PPG cases")

        # Show demographic statistics
        print("\nDemographic Statistics:")
        print("-" * 40)

        if 'age' in ppg_clinical.columns:
            ages = ppg_clinical['age'].dropna()
            print(f"Age (n={len(ages)}):")
            print(f"  Mean: {ages.mean():.1f} years")
            print(f"  Std: {ages.std():.1f} years")
            print(f"  Range: {ages.min():.0f} - {ages.max():.0f}")

        if 'sex' in ppg_clinical.columns:
            sex_counts = ppg_clinical['sex'].value_counts()
            print(f"\nSex distribution:")
            for sex, count in sex_counts.items():
                pct = (count / len(ppg_clinical)) * 100
                label = "Male" if sex in ['M', 1, 'Male'] else "Female"
                print(f"  {label}: {count} ({pct:.1f}%)")

        if 'bmi' in ppg_clinical.columns:
            bmi = ppg_clinical['bmi'].dropna()
            if len(bmi) > 0:
                print(f"\nBMI (n={len(bmi)}):")
                print(f"  Mean: {bmi.mean():.1f} kg/m²")
                print(f"  Std: {bmi.std():.1f} kg/m²")

        if 'height' in ppg_clinical.columns:
            height = ppg_clinical['height'].dropna()
            if len(height) > 0:
                print(f"\nHeight (n={len(height)}):")
                print(f"  Mean: {height.mean():.1f} cm")
                print(f"  Std: {height.std():.1f} cm")

        if 'weight' in ppg_clinical.columns:
            weight = ppg_clinical['weight'].dropna()
            if len(weight) > 0:
                print(f"\nWeight (n={len(weight)}):")
                print(f"  Mean: {weight.mean():.1f} kg")
                print(f"  Std: {weight.std():.1f} kg")

        # Save PPG demographics
        output_file = self.cache_dir / 'ppg_demographics_real.csv'
        ppg_clinical.to_csv(output_file, index=False)
        print(f"\n✓ Saved PPG demographics to: {output_file}")

        return ppg_clinical

    def create_balanced_splits(
            self,
            ppg_demographics: pd.DataFrame,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            random_seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create balanced train/val/test splits based on demographics.
        """
        print("\n" + "=" * 80)
        print("CREATING BALANCED SPLITS")
        print("=" * 80)

        np.random.seed(random_seed)

        # Group by age ranges
        ppg_demographics['age_group'] = pd.cut(
            ppg_demographics['age'],
            bins=[0, 30, 50, 70, 100],
            labels=['<30', '30-50', '50-70', '70+']
        )

        train_cases = []
        val_cases = []
        test_cases = []

        # Stratified split by age group
        for age_group in ppg_demographics['age_group'].unique():
            if pd.isna(age_group):
                continue

            group_cases = ppg_demographics[
                ppg_demographics['age_group'] == age_group
                ]['caseid'].values

            np.random.shuffle(group_cases)

            n = len(group_cases)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_cases.extend(group_cases[:n_train])
            val_cases.extend(group_cases[n_train:n_train + n_val])
            test_cases.extend(group_cases[n_train + n_val:])

        print(f"✓ Split sizes:")
        print(f"  Train: {len(train_cases)} cases")
        print(f"  Val: {len(val_cases)} cases")
        print(f"  Test: {len(test_cases)} cases")

        # Save splits
        splits = {
            'train': train_cases,
            'val': val_cases,
            'test': test_cases
        }

        splits_file = self.cache_dir / 'ppg_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f)
        print(f"✓ Saved splits to: {splits_file}")

        return train_cases, val_cases, test_cases


def main():
    """Main function to load and process VitalDB data."""
    print("\n" + "=" * 80)
    print("VITALDB REAL DATA LOADING PIPELINE")
    print("=" * 80)

    loader = VitalDBRealDataLoader()

    # Step 1: Load clinical information (includes demographics)
    print("\n[STEP 1] Loading clinical information...")
    clinical_df = loader.load_clinical_information(force_reload=False)

    if clinical_df is None or clinical_df.empty:
        print("\n✗ Failed to load clinical data")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try again later (API might be temporarily down)")
        print("3. Download manually from: https://api.vitaldb.net/cases")
        return None

    # Step 2: Load track list
    print("\n[STEP 2] Loading track list...")
    tracks_df = loader.load_track_list(force_reload=False)

    if tracks_df is None:
        print("✗ Failed to load track list")
        return None

    # Step 3: Find PPG cases
    print("\n[STEP 3] Finding PPG cases...")
    ppg_cases = loader.find_ppg_cases(tracks_df)

    # Step 4: Get demographics for PPG cases
    print("\n[STEP 4] Extracting PPG demographics...")
    ppg_demographics = loader.get_ppg_demographics(clinical_df, ppg_cases)

    # Step 5: Create balanced splits
    print("\n[STEP 5] Creating balanced splits...")
    train_cases, val_cases, test_cases = loader.create_balanced_splits(ppg_demographics)

    print("\n" + "=" * 80)
    print("SUCCESS! REAL DEMOGRAPHICS LOADED")
    print("=" * 80)

    print(f"\n✓ Total PPG cases with demographics: {len(ppg_demographics)}")
    print(f"✓ All data saved to: {loader.cache_dir}")

    print("\nYou can now use these files:")
    print(f"  - {loader.cache_dir}/ppg_demographics_real.csv")
    print(f"  - {loader.cache_dir}/ppg_cases.json")
    print(f"  - {loader.cache_dir}/ppg_splits.json")

    return ppg_demographics


if __name__ == "__main__":
    demographics = main()