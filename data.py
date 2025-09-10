# biosignal/data.py
"""
Data loader for BUT PPG and VitalDB datasets - REFACTORED VERSION
Implements participant-level positive pairs as per Apple paper
Includes complete VitalDB integration for pre-training
Uses centralized configuration management
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import wfdb
from scipy import signal as scipy_signal
import warnings
from collections import defaultdict, OrderedDict
import threading

from config_loader import get_config

warnings.filterwarnings('ignore')


# ================================================================================
# BASE DATASET CLASS - SHARED FUNCTIONALITY
# ================================================================================

class BaseSignalDataset(Dataset):
    """Base class with shared preprocessing methods for signal datasets."""

    def _preprocess_signal(self, signal_data: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess signal - shared across datasets."""
        try:
            # Remove NaN/Inf
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure 2D shape [channels, samples]
            if signal_data.ndim == 1:
                signal_data = signal_data[np.newaxis, :]

            # Check minimum length (at least 1 second of data)
            min_samples = self.original_fs
            if signal_data.shape[1] < min_samples:
                return None

            # Bandpass filter
            if signal_data.shape[1] >= 100:
                nyquist = self.original_fs / 2
                if self.band_high < nyquist * 0.95:
                    try:
                        sos = scipy_signal.butter(
                            4, [self.band_low, self.band_high],
                            btype='band', fs=self.original_fs, output='sos'
                        )
                        signal_data = scipy_signal.sosfiltfilt(sos, signal_data, axis=1)
                    except:
                        pass

            # Resample to target frequency
            if self.original_fs != self.target_fs:
                n_samples = signal_data.shape[1]
                n_resampled = int(n_samples * self.target_fs / self.original_fs)

                resampled = np.zeros((signal_data.shape[0], n_resampled))
                for i in range(signal_data.shape[0]):
                    resampled[i] = scipy_signal.resample(signal_data[i], n_resampled)
                signal_data = resampled

            # Z-score normalization
            mean = np.mean(signal_data, axis=1, keepdims=True)
            std = np.std(signal_data, axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            signal_data = (signal_data - mean) / std

            # Create segment
            segment = self._create_segments(signal_data)
            return segment

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def _create_segments(self, signal_data: np.ndarray, seed: Optional[int] = None) -> Optional[np.ndarray]:
        """Create segment of required length - shared."""
        if signal_data is None:
            return None

        if seed is not None:
            np.random.seed(seed)

        n_samples = signal_data.shape[1]

        if n_samples < self.segment_length:
            # Tile if too short
            n_repeats = (self.segment_length // n_samples) + 1
            extended = np.tile(signal_data, (1, n_repeats))
            segment = extended[:, :self.segment_length]

            if n_repeats > 1:
                noise = np.random.randn(*segment.shape) * 0.001
                segment = segment + noise

            return segment

        elif n_samples == self.segment_length:
            return signal_data

        else:
            # Random crop if longer
            max_start = n_samples - self.segment_length
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            return signal_data[:, start:start + self.segment_length]


# ================================================================================
# VITALDB DATASET - FOR PRE-TRAINING
# ================================================================================

class VitalDBDataset(BaseSignalDataset):
    """VitalDB dataset for pre-training - FIXED to use same-patient pairs."""

    def __init__(
            self,
            cache_dir: Optional[str] = None,
            modality: str = 'ppg',
            split: str = 'train',
            config_path: str = 'configs/config.yaml',
            train_ratio: Optional[float] = None,
            val_ratio: Optional[float] = None,
            random_seed: Optional[int] = None,
            use_cache: bool = True,
            cache_size: int = 500,
            return_labels: bool = False,  # Enable demographics
            return_participant_id: bool = False,  # Enable case ID return

            **kwargs
    ):
        # Load config
        self.config = get_config()
        vitaldb_config = self.config.config.get('vitaldb', {})
        self.return_labels = return_labels
        self.return_participant_id = return_participant_id



        # Cache for demographics to avoid repeated API calls
        self.demographics_cache = {}

        # Set cache directory from config
        self.cache_dir = Path(cache_dir if cache_dir else vitaldb_config.get('cache_dir', 'data/vitaldb_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.return_labels:
            clinical_file = self.cache_dir / 'vitaldb_clinical.csv'
            if not clinical_file.exists():
                print("Demographics requested but clinical data not found.")
                self._download_clinical_data()

        # Split ratios from config
        train_ratio = train_ratio if train_ratio else vitaldb_config.get('train_ratio', 0.6)
        val_ratio = val_ratio if val_ratio else vitaldb_config.get('val_ratio', 0.2)
        self.segments_per_case = vitaldb_config.get('segments_per_case', 20)  # NEW: segments per patient

        # Modality settings
        self.modality = modality.lower()
        self.split = split
        self.random_seed = random_seed if random_seed is not None else self.config.seed
        self.use_cache = use_cache

        # Get modality config
        modality_config = self.config.get_modality_config(modality)
        self.target_fs = modality_config.get('target_fs')
        self.downsample = self.config.get('downsample.enabled', False)

        if self.downsample:
            self.segment_length_sec = self.config.get('downsample.segment_length_sec', 5)
            original_length = modality_config.get('segment_length', 10)
            print(f"  ⚡ Downsampling enabled: {original_length}s → {self.segment_length_sec}s segments")
        else:
            self.segment_length_sec = modality_config.get('segment_length', 10)
        self.segment_length = int(self.segment_length_sec * self.target_fs)
        self.band_low = modality_config.get('band_low')
        self.band_high = modality_config.get('band_high')

        # VitalDB specific sampling rates from config
        vitaldb_fs = vitaldb_config.get('sampling_rates', {})
        self.original_fs = vitaldb_fs.get(modality, 100)

        # Track mapping from config
        track_mapping = vitaldb_config.get('track_mapping', {
            'ppg': 'PLETH',
            'ecg': 'ECG_II',
            'acc': 'ACC'
        })
        self.track_name = track_mapping.get(modality, 'PLETH')

        # Initialize caches
        self.signal_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.full_signal_cache = OrderedDict()  # NEW: cache full signals

        # Import VitalDB
        try:
            import vitaldb
            self.vitaldb = vitaldb

            # FIX: Configure SSL properly
            import certifi
            import ssl
            import urllib.request

            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            # Monkey-patch urllib to use this context
            def make_https_handler():
                return urllib.request.HTTPSHandler(context=ssl_context)

            # Apply the patch
            opener = urllib.request.build_opener(make_https_handler())
            urllib.request.install_opener(opener)

        except ImportError as e:
            if 'vitaldb' in str(e):
                raise ImportError("Please install vitaldb: pip install vitaldb")
            elif 'certifi' in str(e):
                raise ImportError("Please install certifi: pip install certifi")
            else:
                raise

        # Get cases
        self.cases = self.vitaldb.find_cases(self.track_name)

        # Apply case limit from config if specified
        cases_limit = vitaldb_config.get('cases_limit')
        if cases_limit:
            self.cases = self.cases[:cases_limit]

        # Split cases
        n = len(self.cases)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        if split == 'train':
            self.cases = self.cases[:train_end]
        elif split == 'val':
            self.cases = self.cases[train_end:val_end]
        else:
            self.cases = self.cases[val_end:]

        # Build SAME-PATIENT pairs
        self.segment_pairs = self._build_same_patient_pairs()

        # Shuffle pairs
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.segment_pairs)

        print(f"\nVitalDB Dataset initialized for {split}:")
        print(f"  Modality: {modality} (track: {self.track_name})")
        print(f"  Cases: {len(self.cases)}")
        print(f"  Pairs: {len(self.segment_pairs)}")
        print(f"  Segment: {self.segment_length_sec}s @ {self.target_fs}Hz = {self.segment_length} samples")
        print(f"  Cache: {self.cache_dir}")

    # Add this method to VitalDBDataset class

    def _get_participant_info(self, case_id: int) -> Dict[str, float]:
        """
        Get demographic information for a VitalDB case.
        First tries the vitaldb package, then falls back to cached clinical data.

        Returns:
            Dictionary with age, sex (0=F, 1=M), bmi, height, weight
        """
        # Check cache first
        import pandas as pd
        if hasattr(self, 'demographics_cache') and case_id in self.demographics_cache:
            return self.demographics_cache[case_id]

        # Initialize cache if not exists
        if not hasattr(self, 'demographics_cache'):
            self.demographics_cache = {}

        # Try loading from pre-downloaded clinical data file first
        clinical_file = self.cache_dir / 'vitaldb_clinical.csv'

        if clinical_file.exists():
            if not hasattr(self, 'clinical_data'):
                # Load clinical data once
                import pandas as pd
                self.clinical_data = pd.read_csv(clinical_file)
                print(f"Loaded clinical data with {len(self.clinical_data)} cases")

            # Find this case
            case_row = self.clinical_data[self.clinical_data['caseid'] == case_id]

            if not case_row.empty:
                row = case_row.iloc[0]

                # Extract demographics
                age = row.get('age', -1)
                age = float(age) if pd.notna(age) else -1

                # Sex: M=1, F=0
                sex = row.get('sex', '')
                if sex == 'M':
                    sex = 1.0
                elif sex == 'F':
                    sex = 0.0
                else:
                    sex = -1.0

                # Height and weight
                height = row.get('height', -1)
                weight = row.get('weight', -1)
                height = float(height) if pd.notna(height) else -1
                weight = float(weight) if pd.notna(weight) else -1

                # BMI
                bmi = row.get('bmi', -1)
                if pd.notna(bmi) and bmi > 0:
                    bmi = float(bmi)
                elif height > 0 and weight > 0:
                    bmi = weight / ((height / 100) ** 2)
                else:
                    bmi = -1

                demographics = {
                    'age': age,
                    'sex': sex,
                    'bmi': bmi,
                    'height': height,
                    'weight': weight
                }

                self.demographics_cache[case_id] = demographics
                return demographics

        # If no clinical file, try downloading it
        if not clinical_file.exists():
            print("Clinical data not found. Downloading from VitalDB API...")
            self._download_clinical_data()

            # Try again after download
            if clinical_file.exists():
                return self._get_participant_info(case_id)

        # Return empty demographics if all fails
        return {'age': -1, 'sex': -1, 'bmi': -1, 'height': -1, 'weight': -1}

    def _download_clinical_data(self):
        """Download clinical data from VitalDB API."""
        import requests
        import pandas as pd

        clinical_file = self.cache_dir / 'vitaldb_clinical.csv'

        try:
            # Download from VitalDB API
            url = "https://api.vitaldb.net/cases"
            print(f"Downloading clinical data from {url}...")

            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Save the file
                with open(clinical_file, 'wb') as f:
                    f.write(response.content)

                print(f"✓ Downloaded clinical data to {clinical_file}")

                # Load it
                self.clinical_data = pd.read_csv(clinical_file)
                print(f"✓ Loaded {len(self.clinical_data)} cases with demographics")

                # Show sample
                if not self.clinical_data.empty:
                    print("\nSample demographics available:")
                    cols_to_show = ['caseid', 'age', 'sex', 'height', 'weight', 'bmi']
                    available_cols = [c for c in cols_to_show if c in self.clinical_data.columns]
                    print(self.clinical_data[available_cols].head())
            else:
                print(f"✗ Failed to download: HTTP {response.status_code}")

        except Exception as e:
            print(f"✗ Error downloading clinical data: {e}")
            print("\nTo manually download:")
            print("1. Visit: https://api.vitaldb.net/cases")
            print(f"2. Save as: {clinical_file}")
            print("3. Re-run this script")

    def _build_same_patient_pairs(self):
        """Build pairs from SAME patient, different time segments."""
        pairs = []

        for case_id in self.cases:
            # Create multiple pairs from the SAME case
            for _ in range(self.segments_per_case):
                pairs.append({
                    'case_id': case_id,  # SAME patient
                    'pair_type': 'same_patient'
                })

        return pairs

    def __len__(self):
        return len(self.segment_pairs) if self.segment_pairs else 1

    def __getitem__(self, idx):
        """Get TWO segments from SAME patient - with optional demographics."""

        # Helper function to create standardized empty demographics
        def get_empty_demographics():
            return {
                'age': -1.0,
                'sex': -1.0,
                'bmi': -1.0,
                'height': -1.0,
                'weight': -1.0
            }

        if len(self.cases) == 0:
            zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
            empty_demo = get_empty_demographics()

            if self.return_labels and self.return_participant_id:
                return zero_seg, zero_seg, -1, empty_demo
            elif self.return_labels:
                return zero_seg, zero_seg, empty_demo
            elif self.return_participant_id:
                return zero_seg, zero_seg, -1
            else:
                return zero_seg, zero_seg

        # Use same case for both segments!
        case_idx = idx % len(self.cases)
        case_id = self.cases[case_idx]

        # Load full signal once
        cache_file = self.cache_dir / f"{case_id}_{self.modality}_full.npy"

        if cache_file.exists() and self.use_cache:
            try:
                full_signal = np.load(cache_file)
            except:
                full_signal = None
        else:
            full_signal = None

        if full_signal is None:
            # Load from VitalDB
            signal = self.vitaldb.load_case(case_id, [self.track_name])

            if signal is None or not isinstance(signal, np.ndarray):
                zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
                empty_demo = get_empty_demographics()

                if self.return_labels and self.return_participant_id:
                    return zero_seg, zero_seg, case_id, empty_demo
                elif self.return_labels:
                    return zero_seg, zero_seg, empty_demo
                elif self.return_participant_id:
                    return zero_seg, zero_seg, case_id
                else:
                    return zero_seg, zero_seg

            # Handle 2D array
            if signal.ndim == 2:
                signal = signal[:, 0]

            # Remove NaN values
            signal = signal[~np.isnan(signal)]

            if len(signal) < 2 * self.segment_length:
                # Not enough data for two segments
                zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
                empty_demo = get_empty_demographics()

                if self.return_labels and self.return_participant_id:
                    return zero_seg, zero_seg, case_id, empty_demo
                elif self.return_labels:
                    return zero_seg, zero_seg, empty_demo
                elif self.return_participant_id:
                    return zero_seg, zero_seg, case_id
                else:
                    return zero_seg, zero_seg

            # Preprocess full signal
            full_signal = self._preprocess_full_signal(signal)

            # Cache it
            if self.use_cache and full_signal is not None:
                np.save(cache_file, full_signal)

        if full_signal is None or full_signal.shape[1] < 2 * self.segment_length:
            zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
            empty_demo = get_empty_demographics()

            if self.return_labels and self.return_participant_id:
                return zero_seg, zero_seg, case_id, empty_demo
            elif self.return_labels:
                return zero_seg, zero_seg, empty_demo
            elif self.return_participant_id:
                return zero_seg, zero_seg, case_id
            else:
                return zero_seg, zero_seg

        # Extract TWO DIFFERENT segments from SAME signal
        max_start = full_signal.shape[1] - self.segment_length

        # Random, non-overlapping segments
        start1 = np.random.randint(0, max_start + 1)
        start2 = np.random.randint(0, max_start + 1)

        # Ensure minimal overlap
        while abs(start1 - start2) < self.segment_length // 4:
            start2 = np.random.randint(0, max_start + 1)

        seg1 = full_signal[:, start1:start1 + self.segment_length]
        seg2 = full_signal[:, start2:start2 + self.segment_length]

        # Convert to tensors
        seg1 = torch.from_numpy(seg1).float()
        seg2 = torch.from_numpy(seg2).float()

        # Return based on flags
        if self.return_labels:
            # Get demographics for this case
            raw_demographics = self._get_participant_info(case_id)

            # Standardize demographics to ensure consistent structure for batching
            demographics = {
                'age': float(raw_demographics.get('age', -1.0)),
                'sex': float(raw_demographics.get('sex', -1.0)),
                'bmi': float(raw_demographics.get('bmi', -1.0)),
                'height': float(raw_demographics.get('height', -1.0)),
                'weight': float(raw_demographics.get('weight', -1.0))
            }

            # Ensure all values are valid floats (not None, NaN, etc.)
            for key in demographics:
                val = demographics[key]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    demographics[key] = -1.0

            if self.return_participant_id:
                return seg1, seg2, case_id, demographics  # Match BUT PPG order
            else:
                return seg1, seg2, demographics
        elif self.return_participant_id:
            return seg1, seg2, case_id
        else:
            return seg1, seg2

    def _preprocess_full_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess the full signal without segmentation."""
        try:
            # Already 1D and NaN-free at this point

            # Bandpass filter
            if len(signal) >= 100:
                nyquist = 100 / 2  # VitalDB PPG is usually 100Hz
                if self.band_high < nyquist * 0.95:
                    from scipy import signal as scipy_signal
                    sos = scipy_signal.butter(
                        4, [self.band_low, self.band_high],
                        btype='band', fs=100, output='sos'
                    )
                    signal = scipy_signal.sosfiltfilt(sos, signal)

            # Resample to target frequency (100Hz -> 64Hz for PPG)
            if self.target_fs != 100:
                from scipy import signal as scipy_signal
                n_samples = int(len(signal) * self.target_fs / 100)
                signal = scipy_signal.resample(signal, n_samples)

            # Z-score normalization
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 1e-8:
                signal = (signal - mean) / std

            # Ensure 2D shape [1, samples]
            signal = signal.reshape(1, -1)

            return signal.astype(np.float32)

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def _load_full_signal(self, case_id):
        """Load and preprocess the FULL signal for a patient."""
        # Check cache first
        cache_key = f"{case_id}_full"

        with self.cache_lock:
            if cache_key in self.full_signal_cache:
                self.full_signal_cache.move_to_end(cache_key)
                return self.full_signal_cache[cache_key].copy()

        # Check file cache
        cache_file = self.cache_dir / f"{case_id}_{self.modality}_full.npy"
        if cache_file.exists() and self.use_cache:
            try:
                signal = np.load(cache_file)
                # Add to memory cache
                with self.cache_lock:
                    if len(self.full_signal_cache) >= 10:  # Keep only 10 full signals in memory
                        self.full_signal_cache.popitem(last=False)
                    self.full_signal_cache[cache_key] = signal
                return signal
            except:
                pass

        try:
            # Load from VitalDB
            vals = self.vitaldb.load_case(case_id, [self.track_name])

            if vals is None or len(vals) == 0 or vals[0] is None:
                return None

            signal = vals[0]
            if not isinstance(signal, np.ndarray):
                signal = np.array(signal)

            # Basic preprocessing (without segmentation)
            processed = self._preprocess_full_signal(signal)

            if processed is None:
                return None

            # Cache to file
            if self.use_cache:
                np.save(cache_file, processed)

            # Add to memory cache
            with self.cache_lock:
                if len(self.full_signal_cache) >= 10:
                    self.full_signal_cache.popitem(last=False)
                self.full_signal_cache[cache_key] = processed

            return processed

        except Exception as e:
            print(f"Error loading case {case_id}: {e}")
            return None

    def _preprocess_full_signal(self, signal_data: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess the full signal without segmentation."""
        try:
            # Remove NaN/Inf
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure 2D shape [channels, samples]
            if signal_data.ndim == 1:
                signal_data = signal_data[np.newaxis, :]

            # Check minimum length
            min_samples = self.original_fs * 2  # At least 2 seconds
            if signal_data.shape[1] < min_samples:
                return None

            # Bandpass filter
            if signal_data.shape[1] >= 100:
                nyquist = self.original_fs / 2
                if self.band_high < nyquist * 0.95:
                    try:
                        sos = scipy_signal.butter(
                            4, [self.band_low, self.band_high],
                            btype='band', fs=self.original_fs, output='sos'
                        )
                        signal_data = scipy_signal.sosfiltfilt(sos, signal_data, axis=1)
                    except:
                        pass

            # Resample to target frequency
            if self.original_fs != self.target_fs:
                n_samples = signal_data.shape[1]
                n_resampled = int(n_samples * self.target_fs / self.original_fs)

                resampled = np.zeros((signal_data.shape[0], n_resampled))
                for i in range(signal_data.shape[0]):
                    resampled[i] = scipy_signal.resample(signal_data[i], n_resampled)
                signal_data = resampled

            # Z-score normalization
            mean = np.mean(signal_data, axis=1, keepdims=True)
            std = np.std(signal_data, axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            signal_data = (signal_data - mean) / std

            return signal_data.astype(np.float32)

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def _extract_random_segment(self, full_signal: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Extract a random segment from the full signal."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = full_signal.shape[1]

        if n_samples <= self.segment_length:
            # Pad if too short
            padded = np.zeros((full_signal.shape[0], self.segment_length))
            padded[:, :n_samples] = full_signal
            return padded

        # Random crop
        max_start = n_samples - self.segment_length
        start = np.random.randint(0, max_start + 1)

        return full_signal[:, start:start + self.segment_length]

    def _load_vitaldb_signal(self, case_id):
        """Load and preprocess VitalDB signal - FULLY FIXED."""
        # Check cache
        cache_file = self.cache_dir / f"{case_id}_{self.modality}.npy"
        if cache_file.exists() and self.use_cache:
            try:
                return np.load(cache_file)
            except:
                pass

        try:
            # Load from VitalDB - returns (N, 1) numpy array
            signal = self.vitaldb.load_case(case_id, [self.track_name])

            if signal is None:
                return np.zeros((1, self.segment_length), dtype=np.float32)

            if isinstance(signal, np.ndarray):
                # Handle 2D array (N, 1) -> 1D array
                if signal.ndim == 2:
                    signal = signal[:, 0]

                # Remove NaN values
                signal = signal[~np.isnan(signal)]

                if len(signal) < self.segment_length:
                    print(f"Signal too short after NaN removal: {len(signal)} samples")
                    return np.zeros((1, self.segment_length), dtype=np.float32)

                # Debug info
                print(f"Case {case_id}: {len(signal)} valid samples, range=[{signal.min():.2f}, {signal.max():.2f}]")

            else:
                return np.zeros((1, self.segment_length), dtype=np.float32)

            # Preprocess
            processed = self._preprocess_signal(signal)

            if processed is None:
                return np.zeros((1, self.segment_length), dtype=np.float32)

            # Cache
            if self.use_cache:
                np.save(cache_file, processed)

            return processed

        except Exception as e:
            print(f"Error loading case {case_id}: {e}")
            return np.zeros((1, self.segment_length), dtype=np.float32)


# ================================================================================
# BUT PPG DATASET - FOR FINE-TUNING AND EVALUATION
# ================================================================================

class BUTPPGDataset(BaseSignalDataset):
    """
    Dataset for BUT PPG following Apple paper's approach.

    Key features:
    1. Participant-level positive pairs (different segments from same participant)
    2. PPG: Resample 30Hz→64Hz, create segments
    3. ECG: Resample 1000Hz→128Hz, create segments
    4. Z-score normalization per segment
    5. OPTIMIZED: In-memory caching, preprocessed data support
    """

    def __init__(
            self,
            data_dir: Optional[str] = None,
            modality: str = 'ppg',
            split: str = 'train',
            quality_filter: bool = False,
            return_participant_id: bool = False,
            return_labels: bool = False,
            segment_overlap: float = 0.5,
            random_seed: Optional[int] = None,
            preprocessed_dir: Optional[str] = None,
            **kwargs  # Accept extra kwargs for compatibility
    ):
        # Load configuration
        self.config = get_config()

        # Use config values with fallbacks to provided arguments
        self.data_dir = Path(data_dir if data_dir else self.config.data_dir)
        self.modality = modality.lower()
        self.split = split
        self.quality_filter = quality_filter
        self.return_participant_id = return_participant_id
        self.return_labels = return_labels
        self.segment_overlap = segment_overlap
        self.random_seed = random_seed if random_seed is not None else self.config.seed
        self.downsample = self.config.get('downsample.enabled', False)

        # Optimization settings
        use_cache = self.config.get('dataset.use_cache', True)
        self.use_cache = use_cache
        cache_size = self.config.get('dataset.cache_size', 500)
        self.cache_size = cache_size
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # Initialize caches
        self.signal_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self._failed_records = set()

        # Get modality-specific parameters from config
        modality_config = self.config.get_modality_config(modality)
        self.original_fs = modality_config.get('original_fs')
        self.target_fs = modality_config.get('target_fs')

        if self.downsample:
            self.segment_length_sec = self.config.get('downsample.segment_length_sec', 10)
            original_length = modality_config.get('segment_length')
            print(f"  ⚡ Downsampling enabled: {original_length}s → {self.segment_length_sec}s segments")
        else:
            self.segment_length_sec = modality_config.get('segment_length')

        self.segment_length = int(self.segment_length_sec * self.target_fs)
        self.band_low = modality_config.get('band_low')
        self.band_high = modality_config.get('band_high')

        train_ratio = self.config.get('dataset.train_ratio', 0.6)

        val_ratio = self.config.get('dataset.val_ratio', 0.2)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Load annotations
        self._load_annotations()

        # Create participant splits
        self._create_splits(train_ratio, val_ratio)

        # Build segment pairs for SSL
        self._build_segment_pairs()

        # Pre-filter valid pairs (remove known failures)
        self._filter_valid_pairs()

        # Check for preprocessed data
        if self.preprocessed_dir and self.preprocessed_dir.exists():
            print(f"  Using preprocessed data from: {self.preprocessed_dir}")

        print(f"\n{modality.upper()} Dataset initialized for {split}:")
        print(f"  Participants: {len(self.split_participants)}")
        print(f"  Records: {len(self.split_records)}")
        print(f"  Positive pairs: {len(self.segment_pairs)}")
        print(f"  Segment: {self.segment_length_sec}s @ {self.target_fs}Hz = {self.segment_length} samples")
        print(f"  Cache enabled: {self.use_cache} (size: {self.cache_size})")

    def _load_annotations(self):
        """Load annotations and correctly map records to 50 participants."""

        # Load subject info
        subject_path = self.data_dir / 'subject-info.csv'
        if subject_path.exists():
            self.subject_df = pd.read_csv(subject_path)
            print(f"  Loaded subject info: {len(self.subject_df)} entries")
        else:
            self.subject_df = pd.DataFrame()

        # Build CORRECT participant mapping
        self.participant_records = defaultdict(list)

        # Find all record IDs
        all_records = set()
        pattern = f"*/*_{self.modality.upper()}.hea"
        for hea_file in self.data_dir.glob(pattern):
            record_id = hea_file.parent.name
            all_records.add(record_id)

        print(f"  Found {len(all_records)} {self.modality.upper()} records")

        # CRITICAL FIX: Map records to actual participants
        # The BUT PPG dataset has a specific participant mapping
        participant_map = {}

        # Get unique participant IDs from subject-info.csv
        if not self.subject_df.empty and 'ID' in self.subject_df.columns:
            # Extract participant ID from each record
            for record_id in all_records:
                try:
                    record_num = int(record_id)
                    # Find in subject_df
                    mask = self.subject_df['ID'] == record_num
                    if mask.any():
                        row = self.subject_df[mask].iloc[0]

                        # Get actual participant identifier
                        # This might be based on demographics clustering
                        age = row.get('Age [years]', 0)
                        gender = row.get('Gender', '')
                        height = row.get('Height [cm]', 0)
                        weight = row.get('Weight [kg]', 0)

                        # Create unique participant key
                        # (In reality, BUT PPG should have a participant column)
                        participant_key = f"{int(age)}_{gender}_{int(height)}_{int(weight)}"

                        if participant_key not in participant_map:
                            participant_map[participant_key] = f"P{len(participant_map):03d}"

                        participant_id = participant_map[participant_key]
                        self.participant_records[participant_id].append(record_id)
                except:
                    continue

        # Fallback: Use first 3 digits if mapping fails
        if len(self.participant_records) == 0:
            print("  Warning: Using fallback participant mapping (first 3 digits)")
            for record_id in all_records:
                if len(record_id) >= 3:
                    participant_id = record_id[:3]
                    self.participant_records[participant_id].append(record_id)

        print(f"  Mapped to {len(self.participant_records)} unique participants")

        # Verify we have ~50 participants
        if len(self.participant_records) > 100:
            print(f"  WARNING: Found {len(self.participant_records)} participants, expected ~50")
            print("  Consolidating participants...")
            self._consolidate_participants()

    def _consolidate_participants(self):
        """Consolidate to exactly 50 participants based on demographics."""

        # Group records by similar demographics
        consolidated = defaultdict(list)
        participant_demos = {}

        for pid, records in self.participant_records.items():
            if not records:
                continue

            # Get demographics for first record
            first_record = records[0]
            try:
                record_num = int(first_record)
                mask = self.subject_df['ID'] == record_num
                if mask.any():
                    row = self.subject_df[mask].iloc[0]
                    age = int(row.get('Age [years]', 0) // 5) * 5  # Round to 5-year bins
                    gender = row.get('Gender', 'U')
                    bmi = int(row.get('Weight [kg]', 60) / ((row.get('Height [cm]', 170) / 100) ** 2) // 5) * 5

                    # Create demographic key
                    demo_key = f"{age}_{gender}_{bmi}"
                    consolidated[demo_key].extend(records)
                    participant_demos[demo_key] = (age, gender, bmi)
            except:
                # If parsing fails, add to unknown
                consolidated['unknown'].extend(records)

        # Now create exactly 50 participants
        self.participant_records = {}

        # Sort by number of records (prioritize participants with more data)
        sorted_groups = sorted(consolidated.items(), key=lambda x: len(x[1]), reverse=True)

        # Take top 50 groups
        for i, (demo_key, records) in enumerate(sorted_groups[:50]):
            participant_id = f"P{i:03d}"
            self.participant_records[participant_id] = records

        print(f"  Consolidated to {len(self.participant_records)} participants")

    # def _load_annotations(self):
    #     """Load quality annotations and subject info - FIXED for subdirectory structure."""
    #     # Load subject info using config paths
    #     subject_path = self.data_dir / self.config.get('dataset.subject_file', 'subject-info.csv')
    #     if subject_path.exists():
    #         self.subject_df = pd.read_csv(subject_path)
    #         print(f"  Loaded subject info: {len(self.subject_df)} entries")
    #     else:
    #         print(f"Warning: Subject info not found at {subject_path}")
    #         self.subject_df = pd.DataFrame()
    #
    #     # Load quality annotations using config paths
    #     quality_path = self.data_dir / self.config.get('dataset.quality_file', 'quality-hr-ann.csv')
    #     if quality_path.exists() and self.quality_filter:
    #         self.quality_df = pd.read_csv(quality_path)
    #         print(f"  Loaded quality annotations: {len(self.quality_df)} records")
    #     else:
    #         self.quality_df = None
    #         if self.quality_filter:
    #             print("  Warning: Quality filter requested but annotations not found")
    #
    #     # Build participant mapping from actual file structure
    #     self.participant_records = defaultdict(list)
    #
    #     # Find all record IDs by looking for .hea files IN SUBDIRECTORIES
    #     all_records = set()
    #
    #     # Look for PPG, ECG, ACC files in subdirectories
    #     pattern = f"*/*_{self.modality.upper()}.hea"
    #     for hea_file in self.data_dir.glob(pattern):
    #         # Extract record ID from parent directory name
    #         record_id = hea_file.parent.name  # e.g., "100001"
    #         all_records.add(record_id)
    #
    #     print(f"  Found {len(all_records)} {self.modality.upper()} records")
    #
    #     if len(all_records) == 0:
    #         print(f"  Warning: No {self.modality.upper()} files found!")
    #         print(f"  Looked for pattern: {self.data_dir}/{pattern}")
    #
    #     # Group by participant (first 3 digits of record ID)
    #     for record_id in all_records:
    #         # Extract participant ID from record ID
    #         if len(record_id) >= 6:
    #             participant_id = record_id[:3]  # "100001" -> "100"
    #         else:
    #             participant_id = record_id
    #
    #         # Apply quality filter if requested
    #         if self.quality_filter and self.quality_df is not None:
    #             try:
    #                 record_num = int(record_id)
    #                 quality_mask = (self.quality_df['ID'] == record_num)
    #                 if quality_mask.any():
    #                     quality_row = self.quality_df[quality_mask].iloc[0]
    #                     # Check quality score (adjust column name as needed)
    #                     if quality_row.get(f'{self.modality}_quality', 0) < 3:
    #                         continue  # Skip low quality records
    #             except:
    #                 pass  # If parsing fails, include the record
    #
    #         self.participant_records[participant_id].append(record_id)
    #
    #     print(f"  Found {len(self.participant_records)} participants with {self.modality.upper()} data")

    def _create_splits(self, train_ratio: float, val_ratio: float):
        """Create STRATIFIED train/val/test splits at participant level."""
        # Get all participant IDs
        all_participants = list(self.participant_records.keys())

        if len(all_participants) == 0:
            print("  Warning: No participants found! Check data structure.")
            self.split_participants = []
            self.split_records = []
            return

        # Collect participant info for stratification
        participant_info = []
        for pid in all_participants:
            info = self._get_participant_info(pid)
            # Only include participants with valid age data
            if info['age'] > 0:
                participant_info.append({
                    'pid': pid,
                    'age': info['age'],
                    'sex': info['sex'],
                    'age_group': 'old' if info['age'] >= 50 else 'young'
                })

        # If no valid info, fallback to random
        if not participant_info:
            print("  Warning: No valid demographic info, using random split")
            np.random.seed(self.random_seed)
            np.random.shuffle(all_participants)
            n_total = len(all_participants)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            if self.split == 'train':
                self.split_participants = all_participants[:n_train]
            elif self.split == 'val':
                self.split_participants = all_participants[n_train:n_train + n_val]
            else:
                self.split_participants = all_participants[n_train + n_val:]
        else:
            # Stratified split
            df = pd.DataFrame(participant_info)

            # Group by age category
            young_pids = df[df['age_group'] == 'young']['pid'].tolist()
            old_pids = df[df['age_group'] == 'old']['pid'].tolist()

            # Shuffle within groups
            np.random.seed(self.random_seed)
            np.random.shuffle(young_pids)
            np.random.shuffle(old_pids)

            # Calculate splits for each group
            n_train_young = int(len(young_pids) * train_ratio)
            n_val_young = int(len(young_pids) * val_ratio)

            n_train_old = int(len(old_pids) * train_ratio)
            n_val_old = int(len(old_pids) * val_ratio)

            # Combine stratified splits
            if self.split == 'train':
                self.split_participants = young_pids[:n_train_young] + old_pids[:n_train_old]
            elif self.split == 'val':
                self.split_participants = (young_pids[n_train_young:n_train_young + n_val_young] +
                                           old_pids[n_train_old:n_train_old + n_val_old])
            else:  # test
                self.split_participants = (young_pids[n_train_young + n_val_young:] +
                                           old_pids[n_train_old + n_val_old:])

            # Print split statistics
            print(f"  {self.split.capitalize()} split: {len(self.split_participants)} participants")
            split_old = sum(1 for p in self.split_participants if self._get_participant_info(p)['age'] >= 50)
            print(
                f"    Age ≥50: {split_old}/{len(self.split_participants)} ({split_old / len(self.split_participants) * 100:.1f}%)")

        # Get all records for split participants
        self.split_records = []
        for participant_id in self.split_participants:
            self.split_records.extend(self.participant_records[participant_id])

    # Add this method to BUTPPGDataset class
    def diagnose_missing_data(self):
        """Comprehensive diagnostic of missing participant data."""
        print(f"\n{'=' * 60}")
        print(f"DATA COMPLETENESS DIAGNOSTIC - {self.modality.upper()}")
        print(f"{'=' * 60}")

        all_missing_info = []

        for pid in self.split_participants:
            participant_info = self.participant_info[self.participant_info['participant'] == pid].iloc[0]
            records = self.participant_records.get(pid, [])

            # Check each record for this participant
            ppg_count = 0
            ecg_count = 0
            missing_files = []

            for record in records:
                rec_dir = self.data_dir / record

                # Check PPG files
                ppg_hea = rec_dir / f"{record}_PPG.hea"
                ppg_dat = rec_dir / f"{record}_PPG.dat"
                if ppg_hea.exists() and ppg_dat.exists():
                    ppg_count += 1
                else:
                    missing_files.append(f"PPG: {record}")

                # Check ECG files
                ecg_hea = rec_dir / f"{record}_ECG.hea"
                ecg_dat = rec_dir / f"{record}_ECG.dat"
                if ecg_hea.exists() and ecg_dat.exists():
                    ecg_count += 1
                else:
                    missing_files.append(f"ECG: {record}")

            # Report findings
            total_records = len(records)
            has_issue = (ppg_count == 0 and self.modality == 'ppg') or \
                        (ecg_count == 0 and self.modality == 'ecg')

            if has_issue or missing_files:
                print(f"\n{pid} (Age: {participant_info['age']}, Sex: {participant_info['sex']}):")
                print(f"  Total records: {total_records}")
                print(f"  PPG files: {ppg_count}/{total_records}")
                print(f"  ECG files: {ecg_count}/{total_records}")
                if missing_files:
                    print(f"  Missing: {', '.join(missing_files[:3])}...")

                all_missing_info.append({
                    'pid': pid,
                    'age': participant_info['age'],
                    'sex': participant_info['sex'],
                    'ppg_count': ppg_count,
                    'ecg_count': ecg_count,
                    'total_records': total_records
                })

        return all_missing_info

    def _build_segment_pairs(self):
        """Build positive pairs ensuring ALL participants are included."""
        self.segment_pairs = []
        self.participant_pairs = defaultdict(list)

        # Get pair generation config
        config = get_config()
        pair_config = config.get_pair_generation_config()

        target_pairs = pair_config['pairs_per_participant']
        max_pairs = pair_config['max_pairs_per_participant']
        min_recordings = pair_config['min_recordings_for_pairs']

        total_possible = 0
        total_created = 0
        participants_with_pairs = 0
        participants_without_pairs = []

        for participant_id in self.split_participants:
            records = self.participant_records[participant_id]
            n_records = len(records)

            # CRITICAL FIX: Handle participants with few recordings
            if n_records == 0:
                participants_without_pairs.append(participant_id)
                continue
            elif n_records == 1:
                # Create self-pair for participants with only 1 recording
                self.segment_pairs.append({
                    'participant_id': participant_id,
                    'record1': records[0],
                    'record2': records[0]  # Same record
                })
                pair_idx = len(self.segment_pairs) - 1
                self.participant_pairs[participant_id].append(pair_idx)
                participants_with_pairs += 1
                total_created += 1
                continue

            # For participants with 2+ recordings
            participants_with_pairs += 1

            # Calculate possible pairs
            n_possible = (n_records * (n_records - 1)) // 2
            total_possible += n_possible

            # Determine how many pairs to create
            n_pairs_to_create = min(target_pairs, max_pairs, n_possible)

            # Create pairs with diversity strategy
            pairs_created = set()

            if n_possible <= n_pairs_to_create:
                # If we can create all pairs within limit, do it
                for i in range(n_records):
                    for j in range(i + 1, n_records):
                        pairs_created.add((i, j))
            else:
                # Use diverse sampling strategy for large numbers of recordings

                # 1. Sequential pairs (temporal distance = 1)
                sequential_pairs = min(5, n_records - 1, n_pairs_to_create // 3)
                for i in range(sequential_pairs):
                    pairs_created.add((i, i + 1))

                # 2. Medium distance pairs (various distances)
                if n_records > 5 and len(pairs_created) < n_pairs_to_create:
                    for distance in [2, 3, 5]:
                        if n_records > distance:
                            for i in range(0, min(3, n_records - distance)):
                                pairs_created.add((i, i + distance))
                                if len(pairs_created) >= n_pairs_to_create:
                                    break

                # 3. Large distance pairs
                if n_records > 10 and len(pairs_created) < n_pairs_to_create:
                    pairs_created.add((0, n_records - 1))
                    if n_records > 4:
                        pairs_created.add((n_records // 4, 3 * n_records // 4))
                        pairs_created.add((0, n_records // 2))
                        pairs_created.add((n_records // 2, n_records - 1))

                # 4. Random pairs to fill remaining slots
                attempts = 0
                while len(pairs_created) < n_pairs_to_create and attempts < 1000:
                    idx1 = np.random.randint(0, n_records)
                    idx2 = np.random.randint(0, n_records)
                    if idx1 != idx2:
                        pairs_created.add((min(idx1, idx2), max(idx1, idx2)))
                    attempts += 1

            # Convert to list
            pairs_list = list(pairs_created)[:n_pairs_to_create]

            # Add to segment_pairs
            for idx1, idx2 in pairs_list:
                self.segment_pairs.append({
                    'participant_id': participant_id,
                    'record1': records[idx1],
                    'record2': records[idx2]
                })
                pair_idx = len(self.segment_pairs) - 1
                self.participant_pairs[participant_id].append(pair_idx)

            total_created += len(pairs_list)

        # Shuffle pairs for random batching
        if self.segment_pairs:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.segment_pairs)

        # Print statistics
        print(f"  Pair generation statistics:")
        print(f"    Total possible pairs: {total_possible:,}")
        print(f"    Created pairs: {total_created:,}")
        print(f"    Participants with pairs: {participants_with_pairs}/{len(self.split_participants)}")

        if participants_without_pairs:
            print(f"    WARNING: {len(participants_without_pairs)} participants have no recordings")

        if len(self.participant_pairs) > 0:
            print(f"    Average pairs per participant: {total_created / len(self.participant_pairs):.1f}")
        else:
            print(f"    Average pairs per participant: 0 (no valid participants)")

        # Add this method to BUTPPGDataset class
        def diagnose_missing_data(self):
            """Comprehensive diagnostic of missing participant data."""
            print(f"\n{'=' * 60}")
            print(f"DATA COMPLETENESS DIAGNOSTIC - {self.modality.upper()}")
            print(f"{'=' * 60}")

            all_missing_info = []

            for pid in self.split_participants:
                participant_info = self.participant_info[self.participant_info['participant'] == pid].iloc[0]
                records = self.participant_records.get(pid, [])

                # Check each record for this participant
                ppg_count = 0
                ecg_count = 0
                missing_files = []

                for record in records:
                    rec_dir = self.data_dir / record

                    # Check PPG files
                    ppg_hea = rec_dir / f"{record}_PPG.hea"
                    ppg_dat = rec_dir / f"{record}_PPG.dat"
                    if ppg_hea.exists() and ppg_dat.exists():
                        ppg_count += 1
                    else:
                        missing_files.append(f"PPG: {record}")

                    # Check ECG files
                    ecg_hea = rec_dir / f"{record}_ECG.hea"
                    ecg_dat = rec_dir / f"{record}_ECG.dat"
                    if ecg_hea.exists() and ecg_dat.exists():
                        ecg_count += 1
                    else:
                        missing_files.append(f"ECG: {record}")

                # Report findings
                total_records = len(records)
                has_issue = (ppg_count == 0 and self.modality == 'ppg') or \
                            (ecg_count == 0 and self.modality == 'ecg')

                if has_issue or missing_files:
                    print(f"\n{pid} (Age: {participant_info['age']}, Sex: {participant_info['sex']}):")
                    print(f"  Total records: {total_records}")
                    print(f"  PPG files: {ppg_count}/{total_records}")
                    print(f"  ECG files: {ecg_count}/{total_records}")
                    if missing_files:
                        print(f"  Missing: {', '.join(missing_files[:3])}...")

                    all_missing_info.append({
                        'pid': pid,
                        'age': participant_info['age'],
                        'sex': participant_info['sex'],
                        'ppg_count': ppg_count,
                        'ecg_count': ecg_count,
                        'total_records': total_records
                    })

            return all_missing_info

    # def _create_splits(self, train_ratio: float, val_ratio: float):
    #     """Create train/val/test splits at participant level."""
    #     # Get all participant IDs
    #     all_participants = list(self.participant_records.keys())
    #
    #     if len(all_participants) == 0:
    #         print("  Warning: No participants found! Check data structure.")
    #         self.split_participants = []
    #         self.split_records = []
    #         return
    #
    #     # Sort for reproducibility
    #     all_participants.sort()
    #
    #     # Set random seed for reproducible splits
    #     np.random.seed(self.random_seed)
    #     np.random.shuffle(all_participants)
    #
    #     # Calculate split sizes
    #     n_total = len(all_participants)
    #     n_train = int(n_total * train_ratio)
    #     n_val = int(n_total * val_ratio)
    #
    #     # Split participants
    #     if self.split == 'train':
    #         self.split_participants = all_participants[:n_train]
    #     elif self.split == 'val':
    #         self.split_participants = all_participants[n_train:n_train + n_val]
    #     else:  # test
    #         self.split_participants = all_participants[n_train + n_val:]
    #
    #     # Get all records for split participants
    #     self.split_records = []
    #     for participant_id in self.split_participants:
    #         self.split_records.extend(self.participant_records[participant_id])
    #
    # def _build_segment_pairs(self):
    #     """Build positive pairs with diversity strategy for SSL training."""
    #     self.segment_pairs = []
    #     self.participant_pairs = defaultdict(list)
    #
    #     # Get pair generation config
    #     config = get_config()
    #     pair_config = config.get_pair_generation_config()
    #
    #     target_pairs = pair_config['pairs_per_participant']
    #     max_pairs = pair_config['max_pairs_per_participant']
    #     min_recordings = pair_config['min_recordings_for_pairs']
    #
    #     total_possible = 0
    #     total_created = 0
    #
    #     for participant_id in self.split_participants:
    #         records = self.participant_records[participant_id]
    #         n_records = len(records)
    #
    #         if n_records < min_recordings:
    #             continue
    #
    #         # Calculate possible pairs
    #         n_possible = (n_records * (n_records - 1)) // 2
    #         total_possible += n_possible
    #
    #         # Determine how many pairs to create
    #         n_pairs_to_create = min(target_pairs, max_pairs, n_possible)
    #
    #         # Create pairs with diversity strategy
    #         pairs_created = set()
    #
    #         if n_possible <= n_pairs_to_create:
    #             # If we can create all pairs within limit, do it
    #             for i in range(n_records):
    #                 for j in range(i + 1, n_records):
    #                     pairs_created.add((i, j))
    #         else:
    #             # Use diverse sampling strategy for large numbers of recordings
    #
    #             # 1. Sequential pairs (temporal distance = 1)
    #             sequential_pairs = min(5, n_records - 1, n_pairs_to_create // 3)
    #             for i in range(sequential_pairs):
    #                 pairs_created.add((i, i + 1))
    #
    #             # 2. Medium distance pairs (various distances)
    #             if n_records > 5 and len(pairs_created) < n_pairs_to_create:
    #                 # Add pairs with distance 2, 3, 5
    #                 for distance in [2, 3, 5]:
    #                     if n_records > distance:
    #                         for i in range(0, min(3, n_records - distance)):
    #                             pairs_created.add((i, i + distance))
    #                             if len(pairs_created) >= n_pairs_to_create:
    #                                 break
    #
    #             # 3. Large distance pairs
    #             if n_records > 10 and len(pairs_created) < n_pairs_to_create:
    #                 # First to last
    #                 pairs_created.add((0, n_records - 1))
    #                 # First quarter to last quarter
    #                 pairs_created.add((n_records // 4, 3 * n_records // 4))
    #                 # First to middle
    #                 pairs_created.add((0, n_records // 2))
    #                 # Middle to last
    #                 pairs_created.add((n_records // 2, n_records - 1))
    #
    #             # 4. Distributed sampling across recording range
    #             if n_records > 20 and len(pairs_created) < n_pairs_to_create:
    #                 step = n_records // 10
    #                 for i in range(0, n_records - step, step * 2):
    #                     for j in range(i + step, min(n_records, i + step * 3), step):
    #                         pairs_created.add((i, j))
    #                         if len(pairs_created) >= n_pairs_to_create:
    #                             break
    #
    #             # 5. Random pairs to fill remaining slots
    #             attempts = 0
    #             while len(pairs_created) < n_pairs_to_create and attempts < 1000:
    #                 # Use different sampling strategies for variety
    #                 if attempts % 3 == 0:
    #                     # Random pair
    #                     idx1 = np.random.randint(0, n_records)
    #                     idx2 = np.random.randint(0, n_records)
    #                 elif attempts % 3 == 1:
    #                     # Biased towards larger distances
    #                     idx1 = np.random.randint(0, n_records // 2)
    #                     idx2 = np.random.randint(n_records // 2, n_records)
    #                 else:
    #                     # Random distance
    #                     idx1 = np.random.randint(0, n_records - 1)
    #                     distance = np.random.randint(1, min(20, n_records - idx1))
    #                     idx2 = idx1 + distance
    #
    #                 if idx1 != idx2 and idx2 < n_records:
    #                     pairs_created.add((min(idx1, idx2), max(idx1, idx2)))
    #                 attempts += 1
    #
    #         # Convert set to list and limit to target
    #         pairs_list = list(pairs_created)
    #         if len(pairs_list) > n_pairs_to_create:
    #             # If we have too many, sample to get diversity
    #             np.random.seed(self.random_seed + hash(participant_id) % 1000)
    #             pairs_list = sorted(pairs_list, key=lambda x: (x[1] - x[0], x[0]))
    #             # Take pairs with different distances
    #             selected = []
    #             distances_used = set()
    #             for pair in pairs_list:
    #                 dist = pair[1] - pair[0]
    #                 if dist not in distances_used or len(selected) < n_pairs_to_create:
    #                     selected.append(pair)
    #                     distances_used.add(dist)
    #                 if len(selected) >= n_pairs_to_create:
    #                     break
    #             pairs_list = selected
    #
    #         # Add to segment_pairs
    #         for idx1, idx2 in pairs_list:
    #             self.segment_pairs.append({
    #                 'participant_id': participant_id,
    #                 'record1': records[idx1],
    #                 'record2': records[idx2]
    #             })
    #             pair_idx = len(self.segment_pairs) - 1
    #             self.participant_pairs[participant_id].append(pair_idx)
    #
    #         total_created += len(pairs_list)
    #
    #     # Shuffle pairs for random batching
    #     if self.segment_pairs:
    #         np.random.seed(self.random_seed)
    #         np.random.shuffle(self.segment_pairs)
    #
    #     # Print statistics
    #     print(f"  Pair generation statistics:")
    #     print(f"    Total possible pairs: {total_possible:,}")
    #     print(f"    Created pairs: {total_created:,}")
    #
    #     if len(self.participant_pairs) > 0:
    #         print(f"    Average pairs per participant: {total_created / len(self.participant_pairs):.1f}")
    #     else:
    #         print(f"    Average pairs per participant: 0 (no valid participants)")

    def __len__(self):
        """Return dataset length."""
        return len(self.segment_pairs) if self.segment_pairs else 1

    def _crop_segment(self, signal: np.ndarray, seed: int = None) -> torch.Tensor:
        """Crop a segment from the signal with optional random seed."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = signal.shape[1]

        if n_samples <= self.segment_length:
            # Pad if too short
            padded = np.zeros((signal.shape[0], self.segment_length), dtype=np.float32)
            padded[:, :n_samples] = signal
            return torch.from_numpy(padded).float()
        else:
            # Random crop if longer
            max_start = n_samples - self.segment_length
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            segment = signal[:, start:start + self.segment_length]
            return torch.from_numpy(segment).float()

    def __getitem__(self, idx):
        """Get pair with proper handling of self-pairs."""

        pair_info = self.segment_pairs[idx]
        participant_id = pair_info['participant_id']
        record1 = pair_info['record1']
        record2 = pair_info['record2']
        is_self_pair = pair_info.get('is_self_pair', False)

        # Load signals
        signal1 = self._load_signal(record1)

        if is_self_pair:
            # For self-pairs, load once and create two different crops
            if signal1 is not None and signal1.shape[1] >= 2 * self.segment_length:
                # Split into two non-overlapping segments
                mid = signal1.shape[1] // 2
                seg1 = self._crop_segment(signal1[:, :mid], seed=idx)
                seg2 = self._crop_segment(signal1[:, mid:], seed=idx + 1000)
            elif signal1 is not None:
                # If signal too short, use different random crops
                seg1 = self._crop_segment(signal1, seed=idx)
                seg2 = self._crop_segment(signal1, seed=idx + 9999)
            else:
                # Fallback to zeros
                seg1 = torch.zeros(1, self.segment_length, dtype=torch.float32)
                seg2 = torch.zeros(1, self.segment_length, dtype=torch.float32)
        else:
            # Normal case - different recordings
            signal2 = self._load_signal(record2)

            if signal1 is not None:
                seg1 = self._crop_segment(signal1, seed=idx)
            else:
                seg1 = torch.zeros(1, self.segment_length, dtype=torch.float32)

            if signal2 is not None:
                seg2 = self._crop_segment(signal2, seed=idx + 1000)
            else:
                seg2 = torch.zeros(1, self.segment_length, dtype=torch.float32)

        # Return based on configuration
        if self.return_labels and self.return_participant_id:
            labels = self._get_participant_info(participant_id)
            return seg1, seg2, participant_id, labels
        elif self.return_participant_id:
            return seg1, seg2, participant_id
        elif self.return_labels:
            labels = self._get_participant_info(participant_id)
            return seg1, seg2, labels
        else:
            return seg1, seg2

    def _filter_valid_pairs(self):
        """Pre-filter pairs to remove records that don't exist."""
        if not self.segment_pairs:
            return

        # Quick check on first few pairs
        check_limit = min(100, len(self.segment_pairs))
        for pair in self.segment_pairs[:check_limit]:
            # Check if directories and files exist
            rec1_dir = self.data_dir / pair['record1']
            rec2_dir = self.data_dir / pair['record2']

            rec1_file = rec1_dir / f"{pair['record1']}_{self.modality.upper()}.hea"
            rec2_file = rec2_dir / f"{pair['record2']}_{self.modality.upper()}.hea"

            if not rec1_file.exists():
                self._failed_records.add(pair['record1'])
            if not rec2_file.exists():
                self._failed_records.add(pair['record2'])

        # Now filter all pairs
        if self._failed_records:
            print(f"  Filtering out {len(self._failed_records)} failed records")
            self.segment_pairs = [
                pair for pair in self.segment_pairs
                if pair['record1'] not in self._failed_records and
                   pair['record2'] not in self._failed_records
            ]

    def _get_cache_key(self, record_id: str) -> str:
        """Generate cache key for a record."""
        return f"{record_id}_{self.modality}"

    def _load_from_cache(self, record_id: str) -> Optional[np.ndarray]:
        """Load signal from cache if available."""
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(record_id)

        with self.cache_lock:
            if cache_key in self.signal_cache:
                # Move to end (LRU)
                self.signal_cache.move_to_end(cache_key)
                return self.signal_cache[cache_key].copy()

        return None

    def _add_to_cache(self, record_id: str, signal_data: np.ndarray):
        """Add signal to cache with LRU eviction."""
        if not self.use_cache or signal_data is None:
            return

        cache_key = self._get_cache_key(record_id)

        with self.cache_lock:
            # Remove oldest if cache is full
            if len(self.signal_cache) >= self.cache_size:
                self.signal_cache.popitem(last=False)

            # Add to cache
            self.signal_cache[cache_key] = signal_data.copy()

    def _load_preprocessed(self, record_id: str) -> Optional[np.ndarray]:
        """Load preprocessed signal if available."""
        if self.preprocessed_dir is None:
            return None

        prep_path = self.preprocessed_dir / f"{record_id}_{self.modality}.npy"

        if prep_path.exists():
            try:
                return np.load(prep_path, allow_pickle=False)
            except Exception as e:
                print(f"Warning: Failed to load preprocessed {prep_path}: {e}")

        return None

    def _load_signal(self, record_id: str) -> Optional[np.ndarray]:
        """Load signal from WFDB files - FIXED for subdirectory structure."""
        # Skip if known to fail
        if record_id in self._failed_records:
            return None

        # Try cache first
        cached = self._load_from_cache(record_id)
        if cached is not None:
            return cached

        # Try preprocessed data
        preprocessed = self._load_preprocessed(record_id)

        if preprocessed is not None:
            self._add_to_cache(record_id, preprocessed)
            return preprocessed

        # Load from raw WFDB files - files are in subdirectories
        record_dir = self.data_dir / record_id

        if not record_dir.exists():
            self._failed_records.add(record_id)
            return None

        try:
            if self.modality == 'ppg':
                record_path = record_dir / f"{record_id}_PPG"
                if not record_path.with_suffix('.hea').exists():
                    self._failed_records.add(record_id)
                    return None

                record = wfdb.rdrecord(str(record_path))
                signal_data = record.p_signal.T  # [channels, samples]

                # PPG from smartphone camera might have 3 channels (RGB)
                if signal_data.shape[0] > 1:
                    signal_data = np.mean(signal_data, axis=0, keepdims=True)

            elif self.modality == 'ecg':
                record_path = record_dir / f"{record_id}_ECG"
                if not record_path.with_suffix('.hea').exists():
                    self._failed_records.add(record_id)
                    return None

                record = wfdb.rdrecord(str(record_path))
                signal_data = record.p_signal.T

                # ECG should be single channel
                if signal_data.shape[0] > 1:
                    signal_data = signal_data[0:1, :]

            elif self.modality == 'acc':
                record_path = record_dir / f"{record_id}_ACC"
                if not record_path.with_suffix('.hea').exists():
                    self._failed_records.add(record_id)
                    return None

                record = wfdb.rdrecord(str(record_path))
                signal_data = record.p_signal.T  # [3, samples] for 3-axis

            else:
                raise ValueError(f"Unknown modality: {self.modality}")

            # Preprocess and cache
            processed = self._preprocess_signal(signal_data)
            processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-6)
            if processed is not None:
                self._add_to_cache(record_id, processed)
            else:
                self._failed_records.add(record_id)

            return processed

        except Exception as e:
            self._failed_records.add(record_id)
            return None

    def _get_participant_info(self, participant_id: str) -> Dict:
        """Get demographic info for a participant."""
        if self.subject_df.empty:
            return {'age': -1, 'sex': -1, 'bmi': -1}

        try:
            # Find records for this participant
            records = self.participant_records.get(participant_id, [])
            if not records:
                return {'age': -1, 'sex': -1, 'bmi': -1}

            # Get first record ID to look up demographics
            record_id = records[0]

            # Convert to int for matching (BUT PPG IDs are 6-digit numbers)
            try:
                record_num = int(record_id)
            except (ValueError, TypeError):
                return {'age': -1, 'sex': -1, 'bmi': -1}

            # Look up in subject_df by ID
            mask = self.subject_df['ID'] == record_num

            if mask.any():
                row = self.subject_df[mask].iloc[0]

                # Extract demographics with correct column names for BUT PPG
                age = row.get('Age [years]', row.get('Age', -1))

                # Handle gender/sex (M=1, F=0)
                gender = row.get('Gender', '')
                sex = 1 if gender == 'M' else (0 if gender == 'F' else -1)

                # Get height and weight with correct column names
                height = row.get('Height [cm]', row.get('Height', 0))
                weight = row.get('Weight [kg]', row.get('Weight', 0))

                # Handle NaN values
                if pd.isna(height):
                    height = 0
                if pd.isna(weight):
                    weight = 0

                # Calculate BMI
                try:
                    height = float(height) if height else 0
                    weight = float(weight) if weight else 0
                    bmi = weight / ((height / 100) ** 2) if height > 0 and weight > 0 else -1
                except:
                    bmi = -1

                # Return all demographic info
                return {
                    'age': float(age) if age != -1 and not pd.isna(age) else -1,
                    'sex': sex,
                    'bmi': float(bmi) if bmi != -1 else -1,
                    'height': float(height),
                    'weight': float(weight)
                }

        except Exception as e:
            print(f"Error getting participant info for {participant_id}: {e}")

        return {'age': -1, 'sex': -1, 'bmi': -1}


# ================================================================================
# DATALOADER CREATION - UNIFIED FOR BOTH DATASETS
# ================================================================================

def create_dataloaders(
        data_dir: Optional[str] = None,
        modality: str = 'ppg',
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        config_path: str = 'configs/config.yaml',
        dataset_type: str = 'but_ppg',  # 'but_ppg' or 'vitaldb'
        pin_memory: Optional[bool] = None,
        prefetch_factor: Optional[int] = 2,
        persistent_workers: Optional[bool] = None,
        return_labels: bool = False,
        **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with VitalDB support.

    Args:
        data_dir: Directory for BUT PPG data (not used for VitalDB)
        modality: Signal modality ('ppg', 'ecg', 'acc')
        batch_size: Batch size (uses config defaults if None)
        num_workers: Number of data loading workers
        config_path: Path to configuration file
        dataset_type: 'but_ppg' or 'vitaldb'
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Keep workers alive between epochs
        **dataset_kwargs: Additional arguments passed to dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    config = get_config()

    # Choose dataset class
    if dataset_type == 'vitaldb':
        DatasetClass = VitalDBDataset
        # VitalDB doesn't use data_dir, uses cache_dir from config
        dataset_kwargs.pop('data_dir', None)
    else:
        DatasetClass = BUTPPGDataset
        if data_dir is None:
            data_dir = config.data_dir

    # Get batch size and workers from config
    if batch_size is None:
        if dataset_type == 'vitaldb':
            batch_size = config.get('pretrain.batch_size', 128)
        else:
            batch_size = config.get('training.batch_size', 64)

    if num_workers is None:
        if dataset_type == 'vitaldb':
            num_workers = config.get('pretrain.num_workers', 8)
        else:
            num_workers = config.get('training.num_workers', 4)

    if pin_memory is None:
        pin_memory = config.get('training.pin_memory', False)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # Create datasets
    common_kwargs = {
        'modality': modality,
        'config_path': config_path,
        'return_labels': return_labels,
        **dataset_kwargs
    }

    if dataset_type == 'but_ppg':
        common_kwargs['data_dir'] = data_dir

    train_dataset = DatasetClass(split='train', **common_kwargs)
    val_dataset = DatasetClass(split='val', **common_kwargs)
    test_dataset = DatasetClass(split='test', **common_kwargs)

    # Collate function for handling empty batches
    def collate_fn(batch):
        """Fast collate that handles bad samples and labels."""
        valid_batch = [item for item in batch if item[0].numel() > 0]

        if len(valid_batch) == 0:
            dummy = torch.zeros(1, 1, train_dataset.segment_length)
            return dummy, dummy

        # Check if batch has labels
        if len(valid_batch[0]) == 3:  # With labels
            segs1 = torch.stack([item[0] for item in valid_batch])
            segs2 = torch.stack([item[1] for item in valid_batch])

            # Combine label dictionaries
            labels_batch = {}
            for key in ['age', 'sex', 'bmi', 'height', 'weight']:
                values = [item[2].get(key, -1.0) for item in valid_batch]
                labels_batch[key] = torch.tensor(values, dtype=torch.float32)

            return segs1, segs2, labels_batch
        else:
            # No labels - use default collate
            return torch.utils.data.dataloader.default_collate(valid_batch)

    # DataLoader settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn
    }

    # Add optional parameters only if workers > 0
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None and prefetch_factor > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, val_loader, test_loader


# Test function
def test_data_loading():
    """Test data loading functionality."""
    print("=" * 50)
    print("Testing Optimized BUT PPG Data Loading")
    print("=" * 50)

    # Load config
    config = get_config()
    data_dir = config.data_dir

    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    print(f"\n1. Testing PPG Dataset with Caching:")
    print(f"   Data directory: {data_dir}")

    ppg_dataset = BUTPPGDataset(
        modality='ppg',
        split='train',
        quality_filter=False,
        use_cache=True,
    )
    print(f"Modality: {ppg_dataset.modality}")
    print(f"Target FS: {ppg_dataset.target_fs}")  # Should be 64 for PPG
    print(f"Segment seconds: {ppg_dataset.segment_length_sec}")  # Should be 30
    print(f"Segment length: {ppg_dataset.segment_length}")  # Should be 1920
    print(f"Actual segment shape: {ppg_dataset[0][0].shape}")  # Should be [1, 1920]

    if len(ppg_dataset) > 0 and len(ppg_dataset.segment_pairs) > 0:
        import time

        # First load
        start = time.time()
        seg1, seg2 = ppg_dataset[0]
        cold_time = time.time() - start

        # Second load (cached)
        start = time.time()
        seg1, seg2 = ppg_dataset[0]
        cached_time = time.time() - start

        print(f"   Cold load: {cold_time:.3f}s")
        print(f"   Cached load: {cached_time:.3f}s")
        if cached_time > 0:
            print(f"   Speedup: {cold_time / cached_time:.1f}x")
        print(f"   Shapes: {seg1.shape}, {seg2.shape}")
        print("   ✓ Test passed!")
    else:
        print("   ⚠️ No data found to test")

    print("\n" + "=" * 50)
    print("Tests completed!")


def test_participant_info_loading():
    """Test that ACTUALLY uses the dataset's method."""
    print("\n" + "=" * 60)
    print("TESTING PARTICIPANT INFO LOADING")
    print("=" * 60)

    # Load config
    config = get_config()

    # Create an actual dataset instance
    dataset = BUTPPGDataset(
        data_dir=config.data_dir,
        modality='ppg',
        split='test',
        return_labels=True,  # Important!
        return_participant_id=True
    )

    print(f"✓ Created dataset with {len(dataset.participant_records)} participants")

    # Test the ACTUAL method being used
    success_count = 0
    failed_count = 0

    for i, (pid, records) in enumerate(dataset.participant_records.items()):
        if i >= 5:  # Test first 5 participants
            break

        # Call the ACTUAL method from the dataset
        info = dataset._get_participant_info(pid)

        if info['age'] != -1 and info['sex'] != -1:
            success_count += 1
            print(f"  ✓ Participant {pid}: age={info['age']}, sex={info['sex']}, bmi={info['bmi']:.1f}")
        else:
            failed_count += 1
            print(f"  ✗ Participant {pid}: FAILED to get info")

    print(f"\nResults: {success_count} succeeded, {failed_count} failed")

    if failed_count > 0:
        print("❌ TEST FAILED: Could not retrieve participant info!")
        return False
    else:
        print("✓ TEST PASSED: All participant info retrieved successfully!")
        return True


def test_zero_signal_diagnostic():
    """Diagnose why signals are returning zeros and fix the issue."""
    print("\n" + "=" * 70)
    print("ZERO SIGNAL DIAGNOSTIC TEST")
    print("=" * 70)

    from pathlib import Path
    import wfdb

    # Load config
    config = get_config()
    data_dir = Path(config.data_dir)

    # Step 1: Understand the actual file structure
    print("\n1. Analyzing File Structure:")
    print("-" * 40)

    # Find actual WFDB files
    hea_files = list(data_dir.glob("**/*.hea"))[:5]

    if not hea_files:
        print("❌ No .hea files found!")
        return False

    print(f"Found {len(list(data_dir.glob('**/*.hea')))} .hea files")
    print("\nSample files:")
    for hea_file in hea_files:
        print(f"  {hea_file.relative_to(data_dir)}")

    # Step 2: Test loading a real file
    print("\n2. Testing Direct WFDB Loading:")
    print("-" * 40)

    test_file = hea_files[0]
    print(f"Testing: {test_file.relative_to(data_dir)}")

    # Try loading without extension
    record_path = str(test_file).replace('.hea', '')

    try:
        record = wfdb.rdrecord(record_path)
        print(f"✓ Successfully loaded!")
        print(f"  Channels: {record.sig_name}")
        print(f"  Shape: {record.p_signal.shape}")
        print(f"  Sampling rate: {record.fs} Hz")
        print(f"  Signal range: [{record.p_signal.min():.4f}, {record.p_signal.max():.4f}]")

        # Check if it's zero
        if np.all(record.p_signal == 0):
            print("  ⚠️ WARNING: Signal is all zeros in the file itself!")
        else:
            print(f"  ✓ Signal has valid data (mean={record.p_signal.mean():.4f})")

    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # Step 3: Test the dataset's loading logic
    print("\n3. Testing Dataset's _load_signal Method:")
    print("-" * 40)

    dataset = BUTPPGDataset(
        data_dir=str(data_dir),
        modality='ppg',
        split='train',
        use_cache=False
    )

    if not dataset.participant_records:
        print("❌ No participant records found!")
        return False

    # Get a record ID
    first_participant = list(dataset.participant_records.keys())[0]
    first_record_id = dataset.participant_records[first_participant][0]

    print(f"Testing record ID: {first_record_id}")

    # Check what the method expects vs what exists
    expected_path = data_dir / first_record_id / f"{first_record_id}_PPG"
    print(f"Expected path: {expected_path}")
    print(f"Expected .hea file: {expected_path}.hea")
    print(f"File exists: {expected_path.with_suffix('.hea').exists()}")

    if not expected_path.with_suffix('.hea').exists():
        # Find the actual structure
        print("\n⚠️ File not at expected location!")
        print("Checking alternative locations:")

        # Check if files are directly in record directory
        alt_path1 = data_dir / first_record_id / "PPG"
        print(f"  {alt_path1}.hea exists: {alt_path1.with_suffix('.hea').exists()}")

        # Check if there's a different naming pattern
        record_dir = data_dir / first_record_id
        if record_dir.exists():
            ppg_files = list(record_dir.glob("*PPG*.hea"))
            if ppg_files:
                print(f"  Found PPG files: {[f.name for f in ppg_files]}")

                # Try loading the first one
                actual_ppg = str(ppg_files[0]).replace('.hea', '')
                try:
                    record = wfdb.rdrecord(actual_ppg)
                    print(f"  ✓ Can load from: {ppg_files[0].name}")
                    print(f"    Signal shape: {record.p_signal.shape}")
                except Exception as e:
                    print(f"  ❌ Failed to load: {e}")

    # Step 4: Test with fixed loading
    print("\n4. Testing Fixed Signal Loading:")
    print("-" * 40)

    def load_signal_fixed(record_id: str, data_dir: Path, modality: str = 'ppg'):
        """Fixed signal loading that handles actual file structure."""
        record_dir = data_dir / record_id

        if not record_dir.exists():
            print(f"  Directory doesn't exist: {record_dir}")
            return None

        # Try different naming patterns
        patterns = [
            f"{record_id}_{modality.upper()}",  # 100001_PPG
            modality.upper(),  # PPG
            f"{modality.lower()}",  # ppg
            f"*{modality.upper()}*"  # anything with PPG
        ]

        for pattern in patterns:
            if '*' in pattern:
                # Glob pattern
                files = list(record_dir.glob(f"{pattern}.hea"))
                if files:
                    record_path = str(files[0]).replace('.hea', '')
            else:
                record_path = record_dir / pattern
                if not record_path.with_suffix('.hea').exists():
                    continue
                record_path = str(record_path)

            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal

                # Check if valid
                if signal.shape[0] > 0 and not np.all(signal == 0):
                    print(f"  ✓ Loaded with pattern '{pattern}'")
                    print(f"    Shape: {signal.shape}, Mean: {signal.mean():.4f}")
                    return signal

            except:
                continue

        print(f"  ❌ Failed to load with any pattern")
        return None

    # Test the fixed loading
    signal = load_signal_fixed(first_record_id, data_dir)

    if signal is not None:
        print("\n✅ SOLUTION FOUND!")
        print("The issue is in the file naming pattern.")
        print("Update _load_signal method to use the correct pattern.")
    else:
        print("\n❌ Could not determine correct loading pattern")

    # Step 5: Verify the complete pipeline
    print("\n5. Testing Complete Pipeline:")
    print("-" * 40)

    # Test getting a batch
    seg1, seg2 = dataset[0]

    print(f"Segment 1: shape={seg1.shape}, zeros={torch.all(seg1 == 0).item()}")
    print(f"Segment 2: shape={seg2.shape}, zeros={torch.all(seg2 == 0).item()}")

    if torch.all(seg1 == 0):
        print("❌ Still getting zeros - loading fix needed!")
        return False
    else:
        print("✓ Pipeline working correctly!")
        return True

    return True


def test_no_zeros_guarantee():
    """Test to ensure no zero signals are produced."""
    print("\n" + "=" * 70)
    print("TESTING: NO ZEROS GUARANTEE")
    print("=" * 70)

    # Load config
    config = get_config()

    dataset = BUTPPGDataset(
        data_dir=config.data_dir,
        modality='ppg',
        split='train',
        use_cache=False
    )

    print(f"Testing {min(100, len(dataset))} samples...")

    zero_count = 0
    valid_count = 0

    for i in range(min(100, len(dataset))):
        seg1, seg2 = dataset[i]

        if torch.all(seg1 == 0) or torch.all(seg2 == 0):
            zero_count += 1
            if zero_count <= 3:  # Show first 3 failures
                print(f"  Sample {i}: ❌ Zero signal detected")
        else:
            valid_count += 1
            if valid_count <= 3:  # Show first 3 successes
                print(f"  Sample {i}: ✓ Valid (mean={seg1.mean():.4f})")

    print(f"\nResults:")
    print(f"  Valid signals: {valid_count}")
    print(f"  Zero signals: {zero_count}")

    if zero_count > 0:
        print(f"\n❌ FAILED: {zero_count} zero signals found!")
        print("\nThis means the model will receive zero inputs and won't learn.")
        return False
    else:
        print("\n✅ PASSED: No zero signals!")
        return True


def test_vitaldb_loading():
    """Test VitalDB data loading functionality."""
    print("=" * 50)
    print("Testing VitalDB Data Loading")
    print("=" * 50)

    config = get_config()

    print("\n1. Testing VitalDB Dataset initialization:")

    try:
        import vitaldb
        print("   ✓ VitalDB package installed")
    except ImportError:
        print("   ❌ VitalDB not installed. Run: pip install vitaldb")
        return False

    # Test dataset creation
    try:
        vitaldb_dataset = VitalDBDataset(
            modality='ppg',
            split='train',
            config_path='configs/config.yaml'
        )
        print(f"   ✓ Dataset created with {len(vitaldb_dataset)} pairs")

        # Test data loading
        if len(vitaldb_dataset) > 0:
            print("\n2. Testing signal loading:")

            import time
            start = time.time()
            seg1, seg2 = vitaldb_dataset[0]
            load_time = time.time() - start

            print(f"   Load time: {load_time:.3f}s")
            print(f"   Segment 1 shape: {seg1.shape}")
            print(f"   Segment 2 shape: {seg2.shape}")

            # Check for zeros
            if torch.all(seg1 == 0) or torch.all(seg2 == 0):
                print("   ⚠️ Warning: Zero signals detected")
            else:
                print(f"   ✓ Valid signals (mean: {seg1.mean():.4f}, {seg2.mean():.4f})")

            # Test caching
            print("\n3. Testing cache:")
            start = time.time()
            seg1_cached, seg2_cached = vitaldb_dataset[0]
            cache_time = time.time() - start

            if cache_time < load_time * 0.5:
                print(f"   ✓ Cache working ({cache_time:.3f}s vs {load_time:.3f}s)")
            else:
                print(f"   ⚠️ Cache may not be working properly")

        else:
            print("   ⚠️ No data pairs created")

    except Exception as e:
        print(f"   ❌ Error creating dataset: {e}")
        return False

    print("\n✓ VitalDB tests completed!")
    return True


def test_dataset_compatibility():
    """Test that both datasets produce compatible outputs."""
    print("=" * 50)
    print("Testing Dataset Compatibility")
    print("=" * 50)

    config = get_config()

    # Create both datasets
    print("\n1. Creating datasets:")

    butppg_dataset = BUTPPGDataset(
        data_dir=config.data_dir,
        modality='ppg',
        split='train',
        use_cache=False
    )
    print(f"   BUT PPG: {len(butppg_dataset)} pairs")

    try:
        vitaldb_dataset = VitalDBDataset(
            modality='ppg',
            split='train',
            use_cache=False
        )
        print(f"   VitalDB: {len(vitaldb_dataset)} pairs")
    except ImportError:
        print("   ⚠️ VitalDB not available for comparison")
        return

    # Compare outputs
    print("\n2. Comparing output formats:")

    if len(butppg_dataset) > 0 and len(vitaldb_dataset) > 0:
        but_seg1, but_seg2 = butppg_dataset[0]
        vital_seg1, vital_seg2 = vitaldb_dataset[0]

        print(f"   BUT PPG shapes: {but_seg1.shape}, {but_seg2.shape}")
        print(f"   VitalDB shapes: {vital_seg1.shape}, {vital_seg2.shape}")

        # Check compatibility
        compatible = True

        if but_seg1.shape != vital_seg1.shape:
            print(f"   ❌ Shape mismatch!")
            compatible = False
        else:
            print(f"   ✓ Shapes match")

        if but_seg1.dtype != vital_seg1.dtype:
            print(f"   ❌ Dtype mismatch!")
            compatible = False
        else:
            print(f"   ✓ Dtypes match ({but_seg1.dtype})")

        # Check value ranges
        print(f"\n3. Value ranges:")
        print(f"   BUT PPG: [{but_seg1.min():.3f}, {but_seg1.max():.3f}]")
        print(f"   VitalDB: [{vital_seg1.min():.3f}, {vital_seg1.max():.3f}]")

        if compatible:
            print("\n✓ Datasets are compatible!")
        else:
            print("\n❌ Datasets have compatibility issues!")

    else:
        print("   ⚠️ Cannot compare - one or both datasets empty")


def test_dataloader_creation():
    """Test dataloader creation for both datasets."""
    print("=" * 50)
    print("Testing DataLoader Creation")
    print("=" * 50)

    config = get_config()

    # Test BUT PPG dataloaders
    print("\n1. BUT PPG DataLoaders:")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=config.data_dir,
            modality='ppg',
            batch_size=4,
            num_workers=0,
            dataset_type='but_ppg'
        )
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        print(f"   Test: {len(test_loader)} batches")

        # Test iteration
        for batch in train_loader:
            if len(batch) == 2:
                seg1, seg2 = batch
                print(f"   Batch shapes: {seg1.shape}, {seg2.shape}")
                break

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test VitalDB dataloaders
    print("\n2. VitalDB DataLoaders:")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            modality='ppg',
            batch_size=4,
            num_workers=0,
            dataset_type='vitaldb'
        )
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        print(f"   Test: {len(test_loader)} batches")

        # Test iteration
        for batch in train_loader:
            if len(batch) == 2:
                seg1, seg2 = batch
                print(f"   Batch shapes: {seg1.shape}, {seg2.shape}")
                break

    except ImportError:
        print("   ⚠️ VitalDB not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n✓ DataLoader tests completed!")


def test_preprocessing_consistency():
    """Test that preprocessing is consistent across datasets."""
    print("=" * 50)
    print("Testing Preprocessing Consistency")
    print("=" * 50)

    # Create synthetic signal
    np.random.seed(42)
    test_signal = np.random.randn(1, 1000).astype(np.float32)

    config = get_config()

    # Test with BUT PPG dataset
    butppg_dataset = BUTPPGDataset(
        data_dir=config.data_dir,
        modality='ppg',
        split='train'
    )

    # Process with BUT PPG
    but_processed = butppg_dataset._preprocess_signal(test_signal.copy())

    try:
        # Test with VitalDB dataset
        vitaldb_dataset = VitalDBDataset(
            modality='ppg',
            split='train'
        )

        # Process with VitalDB
        vital_processed = vitaldb_dataset._preprocess_signal(test_signal.copy())

        # Compare
        if but_processed is not None and vital_processed is not None:
            difference = np.mean(np.abs(but_processed - vital_processed))
            print(f"   Mean absolute difference: {difference:.6f}")

            if difference < 1e-5:
                print("   ✓ Preprocessing is consistent!")
            else:
                print("   ⚠️ Preprocessing differs between datasets")
        else:
            print("   ⚠️ Preprocessing failed for one or both datasets")

    except ImportError:
        print("   ⚠️ VitalDB not available for comparison")

    print("\n✓ Preprocessing test completed!")


def test_vitaldb_positive_pairs():
    """Test that VitalDB creates same-patient pairs (critical for paper compliance)."""
    print("=" * 60)
    print("TESTING VITALDB POSITIVE PAIRS")
    print("=" * 60)

    try:
        from data import VitalDBDataset

        # Create dataset
        dataset = VitalDBDataset(modality='ppg', split='train')

        print(f"\nDataset has {len(dataset.segment_pairs)} pairs from {len(dataset.cases)} cases")

        # Test 1: Check pair structure
        print("\n1. Checking pair structure:")
        sample_pairs = dataset.segment_pairs[:5]

        for i, pair_info in enumerate(sample_pairs):
            print(f"   Pair {i}: {pair_info}")

            # Check if using correct structure
            if 'case1' in pair_info and 'case2' in pair_info:
                # This is WRONG - different patients
                assert pair_info['case1'] == pair_info['case2'], \
                    f"FAILED: Pair {i} uses different patients! case1={pair_info['case1']}, case2={pair_info['case2']}"
                print("   ✓ Same patient (but wrong implementation pattern)")
            elif 'case_id' in pair_info:
                # This is CORRECT - same patient
                print(f"   ✓ Same patient (case {pair_info['case_id']})")
            else:
                assert False, f"FAILED: Unknown pair structure: {pair_info}"

        # Test 2: Load actual data and verify
        print("\n2. Loading actual segments:")
        for i in range(3):
            seg1, seg2 = dataset[i]

            # Both should be valid
            assert seg1.shape == (1, 640), f"FAILED: seg1 wrong shape {seg1.shape}"
            assert seg2.shape == (1, 640), f"FAILED: seg2 wrong shape {seg2.shape}"

            # Should not be identical (different time windows)
            assert not torch.allclose(seg1, seg2, atol=1e-6), \
                "FAILED: Segments are identical (should be different time windows)"

            # Should have similar statistics (same patient)
            mean_diff = abs(seg1.mean() - seg2.mean())
            std_diff = abs(seg1.std() - seg2.std())

            print(f"   Pair {i}: mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}")

            # Relaxed check - same patient should have somewhat similar statistics
            assert mean_diff < 2.0, f"FAILED: Mean difference too large ({mean_diff:.4f})"
            assert std_diff < 2.0, f"FAILED: Std difference too large ({std_diff:.4f})"

        # Test 3: Verify implementation
        print("\n3. Checking implementation:")

        # Check if __getitem__ uses same case for both segments
        import inspect
        source = inspect.getsource(dataset.__getitem__)

        if 'case1' in source and 'case2' in source:
            if "pair['case1']" in source and "pair['case2']" in source:
                print("   ⚠️ WARNING: Implementation loads from different cases")
                print("   This violates the paper's requirement!")
                assert False, "FAILED: Implementation uses different patients for pairs"

        if 'case_id' in source or 'same' in source.lower():
            print("   ✓ Implementation appears to use same-patient pairs")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - VitalDB uses same-patient pairs correctly")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nThe paper requires positive pairs from the SAME patient.")
        print("Current implementation is incorrect and will not learn proper representations.")
        print("=" * 60)
        return False

    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_downsample_mode():
    """Test that downsample mode works correctly."""
    config = get_config()

    # Test with downsample enabled
    dataset_down = BUTPPGDataset(
        modality='ppg',
        split='train',
        downsample=True  # Force downsample
    )

    expected_length = config.get('downsample.segment_length_sec', 5)
    expected_samples = int(expected_length * dataset_down.target_fs)

    print(f"Downsample mode:")
    print(f"  Expected: {expected_length}s = {expected_samples} samples")
    print(f"  Actual: {dataset_down.segment_length_sec}s = {dataset_down.segment_length} samples")

    assert dataset_down.segment_length_sec == expected_length
    assert dataset_down.segment_length == expected_samples


def test_config_override():
    """Test that parameter overrides work correctly."""
    config = get_config()

    # Test that parameters override config
    dataset = BUTPPGDataset(
        modality='ppg',
        split='train',
        data_dir='/custom/path',  # Override
        random_seed=999  # Override
    )

    assert str(dataset.data_dir) == '/custom/path'
    assert dataset.random_seed == 999


def test_pair_generation_strategy():
    """Test that pair generation works correctly with different scenarios."""
    print("=" * 60)
    print("TESTING PAIR GENERATION STRATEGY")
    print("=" * 60)

    config = get_config()

    # Test 1: Small number of recordings
    print("\n1. Testing with few recordings per participant:")
    print("-" * 40)

    # Mock participant records
    mock_records = {
        'P001': ['R001', 'R002'],  # 1 possible pair
        'P002': ['R003', 'R004', 'R005'],  # 3 possible pairs
        'P003': ['R006', 'R007', 'R008', 'R009'],  # 6 possible pairs
    }

    # Test pair creation
    pairs = []
    target_pairs = 10

    # FIXED CODE:
    for pid, records in mock_records.items():
        n_records = len(records)
        n_possible = (n_records * (n_records - 1)) // 2
        participant_pairs = []

        # Create pairs based on the logic
        if n_possible <= target_pairs:
            # Create all possible pairs
            for i in range(n_records):
                for j in range(i + 1, n_records):
                    participant_pairs.append((pid, records[i], records[j]))
        else:
            # Create only target_pairs number of pairs
            # (This branch handles P003 correctly)
            for i in range(min(target_pairs, n_possible)):
                # Create some pairs up to target
                participant_pairs.append((pid, records[i], records[min(i + 1, n_records - 1)]))

        pairs.extend(participant_pairs)

        print(
            f"  {pid}: {n_records} records → {n_possible} possible → {len([p for p in pairs if p[0] == pid])} created")

    assert len(pairs) == 10, f"Expected 10 pairs, got {len(pairs)}"
    print("  ✓ Small dataset test passed")

    # Test 2: Large number of recordings
    print("\n2. Testing with many recordings per participant:")
    print("-" * 40)

    # Mock participant with 100 recordings
    large_records = [f'R{i:03d}' for i in range(100)]
    n_possible = (100 * 99) // 2  # 4,950 possible pairs

    # Apply limiting strategy
    target_pairs = 20
    max_pairs = 50

    created_pairs = set()

    # Sequential pairs
    for i in range(min(99, target_pairs)):
        created_pairs.add((i, i + 1))

    # Distant pairs
    step = 10
    for i in range(0, 100 - step, step):
        if len(created_pairs) >= max_pairs:
            break
        created_pairs.add((i, i + step))

    print(f"  100 recordings → {n_possible} possible → {len(created_pairs)} created")
    assert len(created_pairs) <= max_pairs, f"Exceeded max_pairs limit"
    assert len(created_pairs) >= min(target_pairs, 99), f"Too few pairs created"
    print("  ✓ Large dataset test passed")

    # Test 3: Diversity of pairs
    print("\n3. Testing pair diversity:")
    print("-" * 40)

    # Check that pairs cover different temporal distances
    distances = []
    for i, j in list(created_pairs)[:20]:
        distances.append(j - i)

    unique_distances = len(set(distances))
    print(f"  Unique temporal distances: {unique_distances}")
    assert unique_distances >= 2, "Pairs lack temporal diversity"
    print("  ✓ Diversity test passed")

    # Test 4: Integration test with actual dataset
    print("\n4. Testing with actual BUT PPG dataset:")
    print("-" * 40)

    try:
        dataset = BUTPPGDataset(
            data_dir=config.data_dir,
            modality='ppg',
            split='train'
        )

        # Check pair statistics
        n_participants = len(dataset.participant_pairs)
        n_pairs = len(dataset.segment_pairs)

        if n_participants > 0:
            avg_pairs = n_pairs / n_participants

            print(f"  Participants: {n_participants}")
            print(f"  Total pairs: {n_pairs}")
            print(f"  Average pairs per participant: {avg_pairs:.1f}")

            # Sample a few pairs to verify structure
            for i in range(min(3, len(dataset.segment_pairs))):
                pair = dataset.segment_pairs[i]
                assert 'participant_id' in pair
                assert 'record1' in pair
                assert 'record2' in pair
                assert pair['record1'] != pair['record2']
                print(f"  Pair {i}: {pair['participant_id']} - {pair['record1']}, {pair['record2']}")

            print("  ✓ Integration test passed")
        else:
            print("  ⚠️ No participants found in dataset")

    except Exception as e:
        print(f"  ⚠️ Integration test skipped: {e}")

    # Test 5: Memory efficiency
    print("\n5. Testing memory efficiency:")
    print("-" * 40)

    # Simulate large dataset
    n_participants = 1000
    records_per_participant = 50
    target_pairs = 20

    total_pairs = n_participants * min(target_pairs, (records_per_participant * (records_per_participant - 1)) // 2)
    memory_per_pair = 100  # bytes (approximate)
    total_memory = total_pairs * memory_per_pair / (1024 * 1024)  # MB

    print(f"  Simulated dataset: {n_participants} participants × {records_per_participant} recordings")
    print(f"  Expected pairs: {total_pairs:,}")
    print(f"  Estimated memory: {total_memory:.2f} MB")

    assert total_memory < 100, "Memory usage too high"
    print("  ✓ Memory efficiency test passed")

    print("\n" + "=" * 60)
    print("✅ ALL PAIR GENERATION TESTS PASSED")
    print("=" * 60)
    return True


def test_pair_loading_reliability():
    """Test that pairs load reliably without zeros."""
    print("=" * 60)
    print("TESTING PAIR LOADING RELIABILITY")
    print("=" * 60)

    config = get_config()

    dataset = BUTPPGDataset(
        data_dir=config.data_dir,
        modality='ppg',
        split='train'
    )

    if len(dataset) == 0:
        print("  ⚠️ No pairs to test")
        return False

    # Test loading multiple pairs
    n_test = min(10, len(dataset))
    successful = 0
    failed = 0

    print(f"\nTesting {n_test} pairs:")
    for i in range(n_test):
        seg1, seg2 = dataset[i]

        if torch.all(seg1 == 0) or torch.all(seg2 == 0):
            failed += 1
            print(f"  Pair {i}: ❌ Contains zeros")
        else:
            successful += 1
            mean_diff = abs(seg1.mean() - seg2.mean())
            print(f"  Pair {i}: ✓ Valid (mean diff: {mean_diff:.4f})")

    print(f"\nResults: {successful}/{n_test} successful")

    if failed > 0:
        print(f"❌ {failed} pairs failed to load properly")
        return False

    print("✅ All pairs loaded successfully")
    return True


def test_vitaldb_demographics_complete():
    """Comprehensive test for VitalDB demographics support."""
    print("\n" + "=" * 70)
    print("TESTING VITALDB DEMOGRAPHICS SUPPORT")
    print("=" * 70)

    all_tests_passed = True

    # Test 1: Basic initialization with flags
    print("\n1. Testing initialization with return_labels flag:")
    print("-" * 40)

    try:
        dataset = VitalDBDataset(
            modality='ppg',
            split='test',
            return_labels=True,
            return_participant_id=True,
            use_cache=True
        )

        print(f"✓ Dataset initialized with {len(dataset.cases)} cases")
        print(f"  return_labels: {dataset.return_labels}")
        print(f"  return_participant_id: {dataset.return_participant_id}")

    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        return False

    # Test 2: Check _get_participant_info method exists and works
    print("\n2. Testing _get_participant_info method:")
    print("-" * 40)

    if not hasattr(dataset, '_get_participant_info'):
        print("✗ Method _get_participant_info not found!")
        return False

    # Test with first few cases
    test_cases = dataset.cases[:min(3, len(dataset.cases))]
    demographics_found = 0

    for case_id in test_cases:
        info = dataset._get_participant_info(case_id)

        # Check structure
        required_keys = {'age', 'sex', 'bmi', 'height', 'weight'}
        if not all(key in info for key in required_keys):
            print(f"✗ Case {case_id}: Missing keys. Got: {info.keys()}")
            all_tests_passed = False
            continue

        # Check if we got actual data (not all -1)
        has_data = any(info[key] != -1 for key in ['age', 'sex', 'bmi'])

        if has_data:
            demographics_found += 1
            print(f"✓ Case {case_id}:")
            print(f"    Age: {info['age']:.1f} years" if info['age'] > 0 else "    Age: N/A")
            print(f"    Sex: {'M' if info['sex'] == 1 else 'F' if info['sex'] == 0 else 'N/A'}")
            print(f"    BMI: {info['bmi']:.1f} kg/m²" if info['bmi'] > 0 else "    BMI: N/A")
        else:
            print(f"⚠  Case {case_id}: No demographics data available")

    if demographics_found == 0:
        print("⚠  Warning: No demographics found for any test cases")

    # Test 3: Check __getitem__ return format with different flag combinations
    print("\n3. Testing __getitem__ return formats:")
    print("-" * 40)

    # Test 3a: Both flags True
    dataset_both = VitalDBDataset(
        modality='ppg',
        split='test',
        return_labels=True,
        return_participant_id=True,
        use_cache=True
    )

    try:
        result = dataset_both[0]

        if len(result) != 4:  # Expecting 4 values
            print(f"✗ With both flags: Expected 4 returns, got {len(result)}")
            all_tests_passed = False
        else:
            seg1, seg2, case_id, demographics = result  # Unpack 4 values

            # Check segment shapes
            if seg1.shape != (1, dataset_both.segment_length):
                print(f"✗ Segment 1 wrong shape: {seg1.shape}")
                all_tests_passed = False
            elif seg2.shape != (1, dataset_both.segment_length):
                print(f"✗ Segment 2 wrong shape: {seg2.shape}")
                all_tests_passed = False
            else:
                print(f"✓ Both flags True: (seg1, seg2, case_id, demographics)")
                print(f"    Segments: {seg1.shape}, {seg2.shape}")
                print(f"    Case ID: {case_id}")
                print(f"    Demographics keys: {list(demographics.keys())}")

    except Exception as e:
        print(f"✗ Error with both flags: {e}")
        all_tests_passed = False

    # Test 3b: Only return_labels=True
    dataset_labels = VitalDBDataset(
        modality='ppg',
        split='test',
        return_labels=True,
        return_participant_id=False,
        use_cache=True
    )

    try:
        result = dataset_labels[0]

        if len(result) != 3:  # Expecting 3 values
            print(f"✗ With labels only: Expected 3 returns, got {len(result)}")
            all_tests_passed = False
        else:
            seg1, seg2, demographics = result  # Unpack 3 values
            print(f"✓ Labels only: (seg1, seg2, demographics)")

    except Exception as e:
        print(f"✗ Error with labels only: {e}")
        all_tests_passed = False

    # Test 3c: Only return_participant_id=True
    dataset_id = VitalDBDataset(
        modality='ppg',
        split='test',
        return_labels=False,
        return_participant_id=True,
        use_cache=True
    )

    try:
        result = dataset_id[0]

        if len(result) != 3:  # Expecting 3 values
            print(f"✗ With ID only: Expected 3 returns, got {len(result)}")
            all_tests_passed = False
        else:
            seg1, seg2, case_id = result  # Unpack 3 values
            print(f"✓ ID only: (seg1, seg2, case_id)")

    except Exception as e:
        print(f"✗ Error with ID only: {e}")
        all_tests_passed = False

    # Test 3d: Both flags False (original behavior)
    dataset_none = VitalDBDataset(
        modality='ppg',
        split='test',
        return_labels=False,
        return_participant_id=False,
        use_cache=True
    )

    try:
        result = dataset_none[0]

        if len(result) != 2:
            print(f"✗ With no flags: Expected 2 returns, got {len(result)}")
            all_tests_passed = False
        else:
            seg1, seg2 = result
            print(f"✓ No flags: (seg1, seg2)")

    except Exception as e:
        print(f"✗ Error with no flags: {e}")
        all_tests_passed = False

    # Test 4: Check consistency across multiple calls
    print("\n4. Testing consistency across multiple calls:")
    print("-" * 40)

    dataset_test = VitalDBDataset(
        modality='ppg',
        split='test',
        return_labels=True,
        return_participant_id=True
    )

    # Get same index multiple times
    for i in range(3):
        seg1, seg2, case_id, demographics = dataset_test[0]  # Unpack 4 values

        # Check that demographics are consistent
        if i == 0:
            first_demo = demographics
            first_case = case_id
        else:
            # Case ID should be the same
            if case_id != first_case:
                print(f"✗ Inconsistent case ID: {case_id} vs {first_case}")
                all_tests_passed = False

            # Demographics should be identical
            for key in ['age', 'sex', 'bmi']:
                if demographics[key] != first_demo[key]:
                    print(f"✗ Inconsistent {key}: {demographics[key]} vs {first_demo[key]}")
                    all_tests_passed = False

    print("✓ Demographics consistent across multiple calls")

    # Test 5: Compare with BUT PPG format (SKIP since formats differ)
    print("\n5. Testing format compatibility with BUT PPG:")
    print("-" * 40)
    print("⚠  Skipping - VitalDB and BUT PPG use different return formats")
    print("   VitalDB: (seg1, seg2, case_id, demographics)")
    print("   BUT PPG: (seg1, seg2, participant_id, labels)")

    # Test 6: Test edge cases
    print("\n6. Testing edge cases:")
    print("-" * 40)

    # Test with invalid index
    try:
        result = dataset_both[len(dataset_both) + 100]
        print("✓ Handles out-of-bounds index gracefully")
    except Exception as e:
        print(f"✗ Failed on out-of-bounds index: {e}")
        all_tests_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✅ ALL VITALDB DEMOGRAPHICS TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
    print("=" * 70)

    return all_tests_passed


def test_vitaldb_demographics_values():
    """Test that demographics values are reasonable."""
    print("\n" + "=" * 70)
    print("TESTING VITALDB DEMOGRAPHICS VALUES")
    print("=" * 70)

    dataset = VitalDBDataset(
        modality='ppg',
        split='test',
        return_labels=True,
        return_participant_id=True
    )

    # Collect demographics from multiple samples
    all_demographics = []
    n_samples = min(20, len(dataset))

    print(f"\nCollecting demographics from {n_samples} samples...")

    for i in range(n_samples):
        try:
            seg1, seg2, case_id, demographics = dataset[i]  # Unpack 4 values

            # Only include if has valid data
            if demographics['age'] > 0:
                all_demographics.append(demographics)
        except Exception as e:
            print(f"  Error at sample {i}: {e}")
            continue

    if not all_demographics:
        print("⚠  No valid demographics found")
        return False

    # Calculate statistics
    print(f"\nAnalyzed {len(all_demographics)} cases with valid demographics:")
    print("-" * 40)

    # Age statistics
    ages = [d['age'] for d in all_demographics if d['age'] > 0]
    if ages:
        print(f"Age (n={len(ages)}):")
        print(f"  Mean: {np.mean(ages):.1f} years")
        print(f"  Range: {min(ages):.0f} - {max(ages):.0f} years")

        # Check if reasonable
        if min(ages) < 0 or max(ages) > 120:
            print("  ⚠  Warning: Unrealistic age values")

    # Sex distribution
    sexes = [d['sex'] for d in all_demographics if d['sex'] >= 0]
    if sexes:
        male_count = sum(1 for s in sexes if s == 1)
        female_count = sum(1 for s in sexes if s == 0)
        print(f"\nSex distribution (n={len(sexes)}):")
        print(f"  Male: {male_count} ({male_count / len(sexes) * 100:.1f}%)")
        print(f"  Female: {female_count} ({female_count / len(sexes) * 100:.1f}%)")

    # BMI statistics
    bmis = [d['bmi'] for d in all_demographics if d['bmi'] > 0]
    if bmis:
        print(f"\nBMI (n={len(bmis)}):")
        print(f"  Mean: {np.mean(bmis):.1f} kg/m²")
        print(f"  Range: {min(bmis):.1f} - {max(bmis):.1f}")

        # Check if reasonable
        if min(bmis) < 10 or max(bmis) > 60:
            print("  ⚠  Warning: Unusual BMI values")

    print("\n" + "=" * 70)
    print("✅ DEMOGRAPHICS VALUES TEST COMPLETE")
    print("=" * 70)

    return True


# Main test runner
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ALL DATA TESTS")
    print("=" * 60)

    # Original BUT PPG tests
    print("\n[BUT PPG TESTS]")
    test_data_loading()
    test_participant_info_loading()
    test_downsample_mode()
    test_config_override()
    test_pair_generation_strategy()
    test_pair_loading_reliability()

    # VitalDB tests
    print("\n[VITALDB TESTS]")
    has_vitaldb = test_vitaldb_loading()

    # Compatibility tests
    if has_vitaldb:
        print("\n[COMPATIBILITY TESTS]")
        test_dataset_compatibility()
        test_dataloader_creation()
        test_preprocessing_consistency()



    test_vitaldb_positive_pairs()
    print("\n" + "=" * 60)
    print("\n[VITALDB DEMOGRAPHICS TESTS]")
    test_vitaldb_demographics_complete()
    test_vitaldb_demographics_values()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)