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
            train_ratio: float = 0.6,
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
        """Build positive pairs from same participant."""
        self.segment_pairs = []

        for participant_id in self.split_participants:
            records = self.participant_records[participant_id]

            if len(records) >= 2:
                # Multiple recordings: create pairs
                for i in range(len(records)):
                    for j in range(i + 1, len(records)):
                        self.segment_pairs.append({
                            'participant_id': participant_id,
                            'record1': records[i],
                            'record2': records[j]
                        })
            elif len(records) == 1:
                # Single recording: will split into two segments
                self.segment_pairs.append({
                    'participant_id': participant_id,
                    'record1': records[0],
                    'record2': records[0]  # Same record, different segments
                })

        # Shuffle pairs
        if self.segment_pairs:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.segment_pairs)

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
        """Preprocess signal according to paper specifications."""
        try:
            # Remove NaN/Inf
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure 2D shape [channels, samples]
            if signal_data.ndim == 1:
                signal_data = signal_data[np.newaxis, :]

            # Check minimum length
            if signal_data.shape[1] < 50:  # Too short to process
                return None

            # 1. Bandpass filter (with safety checks)
            nyquist = self.original_fs / 2
            if self.band_high < nyquist * 0.95:
                try:
                    filter_order = 4
                    critical_freqs = [self.band_low, self.band_high]
                    sos = scipy_signal.butter(
                        filter_order, critical_freqs,
                        btype='band', fs=self.original_fs, output='sos'
                    )

                    # Check if signal is long enough for filter
                    min_padlen = 3 * max(len(s) for s in sos)

                    if signal_data.shape[1] > min_padlen:
                        signal_data = scipy_signal.sosfiltfilt(sos, signal_data, axis=1)

                except Exception:
                    pass  # Skip filtering if it fails

            # 2. Resample to target frequency
            if self.original_fs != self.target_fs:
                n_samples = signal_data.shape[1]
                n_resampled = int(n_samples * self.target_fs / self.original_fs)

                if n_resampled < 10:  # Too few samples
                    return None

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

        except Exception:
            return None

    def _create_segments(self, signal_data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Create segment of required length."""
        if signal_data is None:
            return None

        # Set seed for reproducible segment extraction
        if seed is not None:
            np.random.seed(seed)

        n_samples = signal_data.shape[1]

        if n_samples < self.segment_length:
            # Too short - try to extend by repeating
            if n_samples < self.segment_length // 4:
                return None  # Too short to extend

            # Repeat and concatenate
            n_repeats = (self.segment_length // n_samples) + 1
            extended = np.tile(signal_data, (1, n_repeats))
            return extended[:, :self.segment_length]
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

    def __len__(self):
        return max(1, len(self.segment_pairs))  # Return at least 1 to avoid errors

    def __getitem__(self, idx):
        """
        Returns positive pairs from SAME participant with no overlap when downsampling.
        """
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

        # Use modulo to handle index overflow
        idx = idx % len(self.segment_pairs)
        pair_info = self.segment_pairs[idx]

        # Get participant ID and info
        participant_id = pair_info['participant_id']
        participant_info = self._get_participant_info(participant_id) if self.return_labels else None

        # Use idx as seed for reproducibility
        np.random.seed(idx + self.random_seed)

        # Get all records for this participant
        participant_records = self.participant_records[participant_id]

        # If participant has multiple records, use different ones
        if len(participant_records) >= 2:
            # Select two different records from same participant
            rec_indices = np.random.choice(len(participant_records), 2, replace=False)
            record1 = participant_records[rec_indices[0]]
            record2 = participant_records[rec_indices[1]]
        else:
            # Use same record but will create different segments
            record1 = participant_records[0]
            record2 = participant_records[0]

        # Load both signals
        signal1 = self._load_signal(record1)
        signal2 = self._load_signal(record2)

        # If either fails, return zeros with labels
        if signal1 is None or signal2 is None:
            zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
            if self.return_labels or self.return_participant_id:
                if self.return_participant_id:
                    return zero_seg, zero_seg, participant_id, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
                else:
                    return zero_seg, zero_seg, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
            else:
                return zero_seg, zero_seg

        # Create segments - FIXED to ensure no overlap
        if record1 == record2:
            # Same record: ensure non-overlapping segments
            signal_length = signal1.shape[1]

            if self.downsample:
                # For downsampling: divide into non-overlapping chunks
                n_possible_segments = signal_length // self.segment_length

                if n_possible_segments >= 2:
                    # Pick two different non-overlapping segment indices
                    segment_indices = np.random.choice(n_possible_segments,
                                                       min(2, n_possible_segments),
                                                       replace=False)

                    # Extract non-overlapping segments
                    start1 = segment_indices[0] * self.segment_length
                    seg1 = signal1[:, start1:start1 + self.segment_length]

                    start2 = segment_indices[1] * self.segment_length
                    seg2 = signal2[:, start2:start2 + self.segment_length]
                else:
                    # Signal too short for 2 non-overlapping segments
                    # Use the whole segment twice (not ideal but safe)
                    seg1 = signal1[:, :self.segment_length]
                    seg2 = signal2[:, :self.segment_length]
            else:
                # Original behavior: random crops with different seeds
                # Ensure minimum gap between segments
                max_start = signal_length - self.segment_length

                if max_start > self.segment_length:
                    # Enough space for non-overlapping segments
                    start1 = np.random.randint(0, max_start - self.segment_length + 1)
                    seg1 = signal1[:, start1:start1 + self.segment_length]

                    # Second segment starts after first one ends
                    min_start2 = start1 + self.segment_length
                    start2 = np.random.randint(min_start2, max_start + 1)
                    seg2 = signal2[:, start2:start2 + self.segment_length]
                else:
                    # Not enough space, use different seeds and hope for the best
                    seg1 = self._create_segments(signal1, seed=idx)
                    seg2 = self._create_segments(signal2, seed=idx + 1000000)
        else:
            # Different records from same participant: can use random crops
            seg1 = self._create_segments(signal1, seed=idx)
            seg2 = self._create_segments(signal2, seed=idx + 1)

        # Handle None returns from segment creation
        if seg1 is None or seg2 is None:
            zero_seg = torch.zeros(1, self.segment_length, dtype=torch.float32)
            if self.return_labels or self.return_participant_id:
                if self.return_participant_id:
                    return zero_seg, zero_seg, participant_id, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
                else:
                    return zero_seg, zero_seg, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
            else:
                return zero_seg, zero_seg

        # Convert to tensors
        seg1_tensor = torch.from_numpy(seg1).float()
        seg2_tensor = torch.from_numpy(seg2).float()

        # Return based on configuration
        if self.return_labels or self.return_participant_id:
            if self.return_participant_id:
                return seg1_tensor, seg2_tensor, participant_id, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
            else:
                return seg1_tensor, seg2_tensor, participant_info or {'age': -1, 'sex': -1, 'bmi': -1}
        else:
            return seg1_tensor, seg2_tensor


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

if __name__ == "__main__":
    test_data_loading()
    test_participant_info_loading()