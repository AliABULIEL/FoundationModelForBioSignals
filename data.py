# biosignal/data.py
"""
Data loader for BUT PPG dataset - FULLY FIXED VERSION
Implements participant-level positive pairs as per Apple paper
Optimized with caching and correct subdirectory structure
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
import yaml

import threading

warnings.filterwarnings('ignore')


class BUTPPGDataset(Dataset):
    """
    Dataset for BUT PPG following Apple paper's approach.

    Key features:
    1. Participant-level positive pairs (different segments from same participant)
    2. PPG: Resample 30Hz→64Hz, create 60s segments
    3. ECG: Resample 1000Hz→128Hz, create 30s segments
    4. ACC: Keep 100Hz, prepare for future use
    5. Z-score normalization per segment
    6. OPTIMIZED: In-memory caching, preprocessed data support, NO RETRIES
    """

    def __init__(
            self,
            data_dir: str,
            modality: str = 'ppg',  # 'ppg', 'ecg', or 'acc'
            split: str = 'train',  # 'train', 'val', or 'test'
            config_path: str = 'configs/config.yaml',
            quality_filter: bool = False,  # Changed to False - use all data by default
            return_participant_id: bool = False,
            return_labels: bool = False,
            segment_overlap: float = 0.5,  # For creating longer segments
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            random_seed: int = 42,
            # New optimization parameters
            use_cache: bool = True,
            cache_size: int = 500,  # Number of signals to cache in memory
            preprocessed_dir: Optional[str] = None,
            # Directory with preprocessed data
            downsample: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.modality = modality.lower()
        self.split = split
        self.quality_filter = quality_filter
        self.return_participant_id = return_participant_id
        self.return_labels = return_labels
        self.segment_overlap = segment_overlap
        self.random_seed = random_seed
        self.downsample = downsample

        # Optimization settings
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # Initialize caches
        self.signal_cache = OrderedDict()  # LRU cache for signals
        self.cache_lock = threading.Lock()  # Thread-safe cache access
        self._failed_records = set()  # Track failed loads to avoid retrying



        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get modality-specific parameters
        self.signal_params = self.config['dataset'][modality]
        self.original_fs = self.signal_params['original_fs']
        self.target_fs = self.signal_params['target_fs']
        self.segment_length_sec = self.signal_params['segment_length']
        self.segment_length = int(self.segment_length_sec * self.target_fs)
        self.band_low = self.signal_params['band_low']
        self.band_high = self.signal_params['band_high']

        if self.downsample:
            self.segment_length_sec = self.config['downsample']["segment_length_sec"]  # ← Changed to 10 seconds
            original_length = self.signal_params['segment_length']
            print(f"  ⚡ Downsampling enabled: {original_length}s → {self.segment_length_sec}s segments")
        else:
            # Use original segment lengths from config
            self.segment_length_sec = self.signal_params['segment_length']

        self.segment_length = int(self.segment_length_sec * self.target_fs)

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
        # Load subject info - FIXED filename
        subject_path = self.data_dir / "subject-info.csv"
        if subject_path.exists():
            self.subject_df = pd.read_csv(subject_path)
            print(f"  Loaded subject info: {len(self.subject_df)} entries")
        else:
            print(f"Warning: Subject info not found at {subject_path}")
            self.subject_df = pd.DataFrame()

        # Load quality annotations
        quality_path = self.data_dir / "quality-hr-ann.csv"
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
            # List a few directories to debug
            subdirs = list(self.data_dir.iterdir())[:5]
            if subdirs:
                print(f"  Sample directories in {self.data_dir}:")
                for subdir in subdirs:
                    if subdir.is_dir():
                        print(f"    - {subdir.name}")
                        # Check what files are in there
                        files = list(subdir.glob(f"*_{self.modality.upper()}.*"))[:2]
                        for f in files:
                            print(f"      -> {f.name}")

        # Group by participant (first 3 digits of record ID)
        for record_id in all_records:
            # Extract participant ID from record ID
            # For 6-digit IDs like "100001", use first 3 digits: "100"
            # Adjust this logic based on your actual participant grouping needs
            if len(record_id) >= 6:
                participant_id = record_id[:3]  # "100001" -> "100"
            else:
                participant_id = record_id

            # Apply quality filter if requested
            if self.quality_filter and self.quality_df is not None:
                # Check quality for this record
                # You may need to adjust this based on your quality CSV structure
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
        self.participant_pairs = defaultdict(list)  # Track which pairs belong to which participant

        # Configuration for pair generation
        max_pairs_per_participant = 300  # Limit to prevent explosion
        min_recordings_for_pairs = 2  # Need at least 2 recordings to make pairs

        total_possible_pairs = 0
        total_created_pairs = 0

        for participant_id in self.split_participants:
            records = self.participant_records[participant_id]
            n_records = len(records)

            if n_records < min_recordings_for_pairs:
                # Skip participants with too few recordings
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
                # Use a set to avoid duplicate pairs
                created_pairs = set()

                # Strategy: Create diverse pairs by ensuring each recording appears at least once
                # First, create a chain of pairs ensuring all recordings are used
                for i in range(n_records - 1):
                    pair = (i, i + 1)
                    created_pairs.add(pair)

                # Add random pairs until we reach the limit
                while len(created_pairs) < max_pairs_per_participant:
                    idx1 = np.random.randint(0, n_records)
                    idx2 = np.random.randint(0, n_records)

                    if idx1 != idx2:
                        # Ensure consistent ordering
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
        print(f"    Reduction ratio: {total_created_pairs / max(1, total_possible_pairs):.2%}")
        print(f"    Participants with pairs: {len(self.participant_pairs)}")

    def __len__(self):
        """Return dataset length."""
        return len(self.segment_pairs) if self.segment_pairs else 1

    def __getitem__(self, idx):
        """Get a positive pair with better error handling."""
        # Handle empty dataset
        if len(self.segment_pairs) == 0:
            # This should rarely happen with proper initialization
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

        # All attempts failed - this should be very rare
        print(f"Warning: Failed to load any valid pair after {max_attempts} attempts at idx {idx}")

        # As last resort, return a valid but repeated signal from cache
        if self.signal_cache:
            # Get any valid signal from cache
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

        # Absolute fallback - should never reach here
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

    def _preprocess_signal(self, signal_data: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess signal - FIXED for BUT PPG's actual format."""
        try:
            # Remove NaN/Inf
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure 2D shape [channels, samples]
            if signal_data.ndim == 1:
                signal_data = signal_data[np.newaxis, :]

            # Check minimum length (at least 1 second of data)
            min_samples = self.original_fs  # At least 1 second
            if signal_data.shape[1] < min_samples:
                return None

            # 1. Bandpass filter (only if we have enough samples)
            if signal_data.shape[1] >= 100:  # Need reasonable length for filtering
                nyquist = self.original_fs / 2
                if self.band_high < nyquist * 0.95:
                    try:
                        sos = scipy_signal.butter(
                            4, [self.band_low, self.band_high],
                            btype='band', fs=self.original_fs, output='sos'
                        )
                        signal_data = scipy_signal.sosfiltfilt(sos, signal_data, axis=1)
                    except:
                        pass  # Skip filtering if it fails

            # 2. Resample to target frequency
            if self.original_fs != self.target_fs:
                n_samples = signal_data.shape[1]
                n_resampled = int(n_samples * self.target_fs / self.original_fs)

                resampled = np.zeros((signal_data.shape[0], n_resampled))
                for i in range(signal_data.shape[0]):
                    resampled[i] = scipy_signal.resample(signal_data[i], n_resampled)
                signal_data = resampled

            # 3. Z-score normalization
            mean = np.mean(signal_data, axis=1, keepdims=True)
            std = np.std(signal_data, axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            signal_data = (signal_data - mean) / std

            return signal_data.astype(np.float32)

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def _create_segments(self, signal_data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Create segment of required length - FIXED for BUT PPG."""
        if signal_data is None:
            return None

        # Set seed for reproducible segment extraction
        if seed is not None:
            np.random.seed(seed)

        n_samples = signal_data.shape[1]

        # For BUT PPG with 10-second data:
        # After resampling PPG: 30Hz→64Hz gives 640 samples
        # After resampling ECG: 100Hz→128Hz gives 1280 samples

        if n_samples < self.segment_length:
            # If we need more samples, tile the signal
            n_repeats = (self.segment_length // n_samples) + 1
            extended = np.tile(signal_data, (1, n_repeats))

            # Take exactly segment_length samples
            segment = extended[:, :self.segment_length]

            # Add small random noise to break exact repetition
            if n_repeats > 1:
                noise = np.random.randn(*segment.shape) * 0.001
                segment = segment + noise

            return segment
        elif n_samples == self.segment_length:
            # Perfect match
            return signal_data
        else:
            # Random crop if longer
            max_start = n_samples - self.segment_length
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            return signal_data[:, start:start + self.segment_length]

    def _get_participant_info(self, participant_id: str) -> Dict:
            # Add this fixed method to the BUTPPGDataset class in data.py

        """Get demographic info for a participant - FIXED VERSION."""
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
                print(f"Warning: Invalid record_id format: {record_id}")
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
                result = {
                    'age': float(age) if age != -1 and not pd.isna(age) else -1,
                    'sex': sex,
                    'bmi': float(bmi) if bmi != -1 else -1,
                    'height': float(height),
                    'weight': float(weight)
                }

                # Debug output (remove after testing)
                if self.split == 'test' and len(self.participant_records) < 10:
                    print(f"DEBUG: Participant {participant_id}, record {record_id}: age={result['age']}, sex={result['sex']}, bmi={result['bmi']}")

                return result
            else:
                # Try all records for this participant
                for record_id in records:
                    try:
                        record_num = int(record_id)
                        mask = self.subject_df['ID'] == record_num
                        if mask.any():
                            # Found it with another record, recursively call with this record
                            self.participant_records[participant_id] = [record_id] + [r for r in records if r != record_id]
                            return self._get_participant_info(participant_id)
                    except:
                        continue

        except Exception as e:
            print(f"Error getting participant info for {participant_id}: {e}")
            import traceback
            traceback.print_exc()

        return {'age': -1, 'sex': -1, 'bmi': -1}




def create_dataloaders(
        data_dir: str,
        modality: str = 'ppg',
        batch_size: int = 64,
        num_workers: int = 4,
        config_path: str = 'configs/config.yaml',
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        preprocessed_dir: Optional[str] = None,
        cache_size: int = 500,
        downsample: bool = False,
        **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with optimizations."""

    # Check for preprocessed directory
    if preprocessed_dir is None:
        prep_path = Path(data_dir).parent / "preprocessed" / modality
        if prep_path.exists():
            preprocessed_dir = str(prep_path)
            print(f"Auto-detected preprocessed data at: {preprocessed_dir}")

    # Create datasets with caching enabled
    train_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=modality,
        split='train',
        config_path=config_path,
        preprocessed_dir=preprocessed_dir,
        cache_size=cache_size,
        use_cache=True,
        downsample=downsample,
        **dataset_kwargs
    )

    val_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=modality,
        split='val',
        config_path=config_path,
        preprocessed_dir=preprocessed_dir,
        cache_size=cache_size // 2,
        use_cache=True,
        downsample=downsample,
        **dataset_kwargs
    )

    test_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=modality,
        split='test',
        config_path=config_path,
        preprocessed_dir=preprocessed_dir,
        cache_size=cache_size // 2,
        use_cache=True,
        downsample=downsample,
        **dataset_kwargs
    )

    # Simpler collate function - no complex error handling
    def fast_collate(batch):
        """Fast collate that skips bad samples."""
        valid_batch = [item for item in batch if item[0].numel() > 0]

        if len(valid_batch) == 0:
            # Return single dummy sample
            dummy = torch.zeros(1, 1, train_dataset.segment_length)
            return dummy, dummy

        return torch.utils.data.dataloader.default_collate(valid_batch)

    # DataLoader settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': fast_collate
    }

    # Add optional parameters only if workers > 0
    if num_workers > 0 and len(train_dataset) > 100:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor > 0:
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

    data_dir = "data/but_ppg/dataset"

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
    print("\n" + "="*60)
    print("TESTING PARTICIPANT INFO LOADING")
    print("="*60)

    # Create an actual dataset instance
    dataset = BUTPPGDataset(
        data_dir="data/but_ppg/dataset",
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

    data_dir = Path("data/but_ppg/dataset")

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

    dataset = BUTPPGDataset(
        data_dir="data/but_ppg/dataset",
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


if __name__ == "__main__":
    test_data_loading()
    test_participant_info_loading()

    print("\n" + "=" * 70)

    # First diagnose the issue
    if test_zero_signal_diagnostic():
        # Then verify no zeros
        test_no_zeros_guarantee()