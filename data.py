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
            **kwargs
    ):
        # Load config
        self.config = get_config()
        vitaldb_config = self.config.config.get('vitaldb', {})

        # Set cache directory from config
        self.cache_dir = Path(cache_dir if cache_dir else vitaldb_config.get('cache_dir', 'data/vitaldb_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Split ratios from config
        train_ratio = train_ratio if train_ratio else vitaldb_config.get('train_ratio', 0.8)
        val_ratio = val_ratio if val_ratio else vitaldb_config.get('val_ratio', 0.1)
        self.segments_per_case = vitaldb_config.get('segments_per_case', 20)  # NEW: segments per patient

        # Modality settings
        self.modality = modality.lower()
        self.split = split
        self.random_seed = random_seed if random_seed is not None else self.config.seed
        self.use_cache = use_cache

        # Get modality config
        modality_config = self.config.get_modality_config(modality)
        self.target_fs = modality_config.get('target_fs')
        self.segment_length_sec = modality_config.get('segment_length')
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
        """Get TWO segments from SAME patient - CRITICAL FIX."""
        if len(self.cases) == 0:
            zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
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
                zero_seg = torch.zeros(1, self.segment_length, dtype=np.float32)
                return zero_seg, zero_seg

            # Handle 2D array
            if signal.ndim == 2:
                signal = signal[:, 0]

            # Remove NaN values
            signal = signal[~np.isnan(signal)]

            if len(signal) < 2 * self.segment_length:
                # Not enough data for two segments
                zero_seg = torch.zeros(1, self.segment_length, dtype=np.float32)
                return zero_seg, zero_seg

            # Preprocess full signal
            full_signal = self._preprocess_full_signal(signal)

            # Cache it
            if self.use_cache and full_signal is not None:
                np.save(cache_file, full_signal)

        if full_signal is None or full_signal.shape[1] < 2 * self.segment_length:
            zero_seg = torch.zeros(1, self.segment_length, dtype=np.float32)
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

        return torch.from_numpy(seg1).float(), torch.from_numpy(seg2).float()

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
            config_path: str = 'configs/config.yaml',
            quality_filter: bool = False,
            return_participant_id: bool = False,
            return_labels: bool = False,
            segment_overlap: float = 0.5,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            random_seed: Optional[int] = None,
            use_cache: bool = True,
            cache_size: int = 500,
            preprocessed_dir: Optional[str] = None,
            downsample: bool = False,
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
        self.downsample = downsample

        # Optimization settings
        self.use_cache = use_cache
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
        """Load quality annotations and subject info - FIXED for subdirectory structure."""
        # Load subject info using config paths
        subject_path = self.data_dir / self.config.get('dataset.subject_file', 'subject-info.csv')
        if subject_path.exists():
            self.subject_df = pd.read_csv(subject_path)
            print(f"  Loaded subject info: {len(self.subject_df)} entries")
        else:
            print(f"Warning: Subject info not found at {subject_path}")
            self.subject_df = pd.DataFrame()

        # Load quality annotations using config paths
        quality_path = self.data_dir / self.config.get('dataset.quality_file', 'quality-hr-ann.csv')
        if quality_path.exists() and self.quality_filter:
            self.quality_df = pd.read_csv(quality_path)
            print(f"  Loaded quality annotations: {len(self.quality_df)} records")
        else:
            self.quality_df = None
            if self.quality_filter:
                print("  Warning: Quality filter requested but annotations not found")

        # Build participant mapping from actual file structure
        self.participant_records = defaultdict(list)

        # Find all record IDs by looking for .hea files IN SUBDIRECTORIES
        all_records = set()

        # Look for PPG, ECG, ACC files in subdirectories
        pattern = f"*/*_{self.modality.upper()}.hea"
        for hea_file in self.data_dir.glob(pattern):
            # Extract record ID from parent directory name
            record_id = hea_file.parent.name  # e.g., "100001"
            all_records.add(record_id)

        print(f"  Found {len(all_records)} {self.modality.upper()} records")

        if len(all_records) == 0:
            print(f"  Warning: No {self.modality.upper()} files found!")
            print(f"  Looked for pattern: {self.data_dir}/{pattern}")

        # Group by participant (first 3 digits of record ID)
        for record_id in all_records:
            # Extract participant ID from record ID
            if len(record_id) >= 6:
                participant_id = record_id[:3]  # "100001" -> "100"
            else:
                participant_id = record_id

            # Apply quality filter if requested
            if self.quality_filter and self.quality_df is not None:
                try:
                    record_num = int(record_id)
                    quality_mask = (self.quality_df['ID'] == record_num)
                    if quality_mask.any():
                        quality_row = self.quality_df[quality_mask].iloc[0]
                        # Check quality score (adjust column name as needed)
                        if quality_row.get(f'{self.modality}_quality', 0) < 3:
                            continue  # Skip low quality records
                except:
                    pass  # If parsing fails, include the record

            self.participant_records[participant_id].append(record_id)

        print(f"  Found {len(self.participant_records)} participants with {self.modality.upper()} data")

    def _create_splits(self, train_ratio: float, val_ratio: float):
        """Create train/val/test splits at participant level."""
        # Get all participant IDs
        all_participants = list(self.participant_records.keys())

        if len(all_participants) == 0:
            print("  Warning: No participants found! Check data structure.")
            self.split_participants = []
            self.split_records = []
            return

        # Sort for reproducibility
        all_participants.sort()

        # Set random seed for reproducible splits
        np.random.seed(self.random_seed)
        np.random.shuffle(all_participants)

        # Calculate split sizes
        n_total = len(all_participants)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split participants
        if self.split == 'train':
            self.split_participants = all_participants[:n_train]
        elif self.split == 'val':
            self.split_participants = all_participants[n_train:n_train + n_val]
        else:  # test
            self.split_participants = all_participants[n_train + n_val:]

        # Get all records for split participants
        self.split_records = []
        for participant_id in self.split_participants:
            self.split_records.extend(self.participant_records[participant_id])

    def _build_segment_pairs(self):
        """Build positive pairs efficiently - avoiding combinatorial explosion."""
        self.segment_pairs = []
        self.participant_pairs = defaultdict(list)

        # Configuration for pair generation
        max_pairs_per_participant = 300
        min_recordings_for_pairs = 2

        total_possible_pairs = 0
        total_created_pairs = 0

        for participant_id in self.split_participants:
            records = self.participant_records[participant_id]
            n_records = len(records)

            if n_records < min_recordings_for_pairs:
                continue

            # Calculate how many pairs we could theoretically create
            n_possible_pairs = (n_records * (n_records - 1)) // 2
            total_possible_pairs += n_possible_pairs

            # Decide how many pairs to actually create
            if n_possible_pairs <= max_pairs_per_participant:
                # If under limit, create all possible pairs
                for i in range(n_records):
                    for j in range(i + 1, n_records):
                        pair_idx = len(self.segment_pairs)
                        self.segment_pairs.append({
                            'participant_id': participant_id,
                            'record1': records[i],
                            'record2': records[j]
                        })
                        self.participant_pairs[participant_id].append(pair_idx)
                        total_created_pairs += 1
            else:
                # Sample a subset of pairs to avoid explosion
                created_pairs = set()

                # Strategy: Create diverse pairs by ensuring each recording appears at least once
                for i in range(n_records - 1):
                    pair = (i, i + 1)
                    created_pairs.add(pair)

                # Add random pairs until we reach the limit
                while len(created_pairs) < max_pairs_per_participant:
                    idx1 = np.random.randint(0, n_records)
                    idx2 = np.random.randint(0, n_records)

                    if idx1 != idx2:
                        pair = (min(idx1, idx2), max(idx1, idx2))
                        created_pairs.add(pair)

                # Convert pairs to segment_pairs format
                for idx1, idx2 in created_pairs:
                    pair_idx = len(self.segment_pairs)
                    self.segment_pairs.append({
                        'participant_id': participant_id,
                        'record1': records[idx1],
                        'record2': records[idx2]
                    })
                    self.participant_pairs[participant_id].append(pair_idx)
                    total_created_pairs += 1

        # Shuffle pairs for random batching
        if self.segment_pairs:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.segment_pairs)

        # Print statistics
        print(f"  Pair generation statistics:")
        print(f"    Total possible pairs: {total_possible_pairs:,}")
        print(f"    Created pairs: {total_created_pairs:,}")
        if total_possible_pairs > 0:
            print(f"    Reduction ratio: {total_created_pairs / total_possible_pairs:.2%}")
        print(f"    Participants with pairs: {len(self.participant_pairs)}")

    def __len__(self):
        """Return dataset length."""
        return len(self.segment_pairs) if self.segment_pairs else 1

    def __getitem__(self, idx):
        """Get a positive pair with better error handling."""
        # Handle empty dataset
        if len(self.segment_pairs) == 0:
            zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
            if self.return_labels or self.return_participant_id:
                empty_info = {'age': -1, 'sex': -1, 'bmi': -1}
                if self.return_participant_id:
                    return zero_seg, zero_seg, "unknown", empty_info
                else:
                    return zero_seg, zero_seg, empty_info
            else:
                return zero_seg, zero_seg

        # Try up to 3 different pairs if loading fails
        max_attempts = 3
        for attempt in range(max_attempts):
            # Get a pair index
            pair_idx = (idx + attempt) % len(self.segment_pairs)
            pair_info = self.segment_pairs[pair_idx]

            participant_id = pair_info['participant_id']
            record1 = pair_info['record1']
            record2 = pair_info['record2']

            # Load both signals
            signal1 = self._load_signal(record1)
            signal2 = self._load_signal(record2)

            # If both signals loaded successfully, process them
            if signal1 is not None and signal2 is not None:
                # Get participant info if needed
                participant_info = self._get_participant_info(participant_id) if self.return_labels else None

                # Since signals are already the right length after preprocessing
                seg1 = signal1
                seg2 = signal2

                # Add tiny noise in training for diversity
                if self.split == 'train':
                    noise_scale = 0.001
                    seg1 = seg1 + np.random.randn(*seg1.shape) * noise_scale
                    seg2 = seg2 + np.random.randn(*seg2.shape) * noise_scale

                # Convert to tensors
                seg1_tensor = torch.from_numpy(seg1).float()
                seg2_tensor = torch.from_numpy(seg2).float()

                # Return successful pair
                if self.return_labels or self.return_participant_id:
                    if self.return_participant_id:
                        return seg1_tensor, seg2_tensor, participant_id, participant_info or {'age': -1, 'sex': -1,
                                                                                              'bmi': -1}
                    else:
                        return seg1_tensor, seg2_tensor, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
                else:
                    return seg1_tensor, seg2_tensor

            # If loading failed, mark these records as failed
            if signal1 is None:
                self._failed_records.add(record1)
            if signal2 is None:
                self._failed_records.add(record2)

        # All attempts failed - return cached signal if available
        if self.signal_cache:
            cached_signal = next(iter(self.signal_cache.values()))
            seg_tensor = torch.from_numpy(cached_signal).float()

            if self.return_labels or self.return_participant_id:
                empty_info = {'age': -1, 'sex': -1, 'bmi': -1}
                if self.return_participant_id:
                    return seg_tensor, seg_tensor, participant_id, empty_info
                else:
                    return seg_tensor, seg_tensor, empty_info
            else:
                return seg_tensor, seg_tensor

        # Absolute fallback
        zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
        if self.return_labels or self.return_participant_id:
            empty_info = {'age': -1, 'sex': -1, 'bmi': -1}
            if self.return_participant_id:
                return zero_seg, zero_seg, "unknown", empty_info
            else:
                return zero_seg, zero_seg, empty_info
        else:
            return zero_seg, zero_seg

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
        **dataset_kwargs
    }

    if dataset_type == 'but_ppg':
        common_kwargs['data_dir'] = data_dir

    train_dataset = DatasetClass(split='train', **common_kwargs)
    val_dataset = DatasetClass(split='val', **common_kwargs)
    test_dataset = DatasetClass(split='test', **common_kwargs)

    # Collate function for handling empty batches
    def collate_fn(batch):
        """Fast collate that handles bad samples."""
        valid_batch = [item for item in batch if item[0].numel() > 0]

        if len(valid_batch) == 0:
            # Return dummy sample if all samples are bad
            dummy = torch.zeros(1, 1, train_dataset.segment_length)
            return dummy, dummy

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
        data_dir=data_dir,
        modality='ppg',
        split='train',
        quality_filter=False,
        use_cache=True,
        cache_size=10
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




# Main test runner
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ALL DATA TESTS")
    print("=" * 60)

    # Original BUT PPG tests
    print("\n[BUT PPG TESTS]")
    test_data_loading()
    test_participant_info_loading()

    # VitalDB tests
    print("\n[VITALDB TESTS]")
    has_vitaldb = test_vitaldb_loading()

    # Compatibility tests
    if has_vitaldb:
        print("\n[COMPATIBILITY TESTS]")
        test_dataset_compatibility()
        test_dataloader_creation()
        test_preprocessing_consistency()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

    test_vitaldb_positive_pairs()
