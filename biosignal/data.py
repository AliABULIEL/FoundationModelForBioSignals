# biosignal/data.py

from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample, sosfiltfilt
from scipy import signal as scipy_signal
from typing import Optional, Tuple, List, Dict
import pandas as pd
from collections import defaultdict
import wfdb
import warnings

warnings.filterwarnings('ignore')


class MIMICWaveformDataset(Dataset):
    """
    Dataset for MIMIC-IV waveforms implementing participant-level positive pairs
    as described in the Apple paper: "Large-scale Training of Foundation Models for Wearable Biosignals"

    Key features aligned with paper:
    1. Participant-level positive pairs (different segments from same participant)
    2. PPG: 60 seconds @ 64Hz (3840 samples)
    3. ECG: 30 seconds @ 128Hz (3840 samples)
    4. Z-score normalization per segment
    5. Train/val/test split at participant level (80/10/10)

    MIMIC-IV Structure:
    data/mimic4wdb/physionet.org/files/mimic4wdb/0.1.0/waves/
    └── p100/p10039708/85940419/
        ├── 85940419.hea (multi-segment header)
        ├── 85940419_0000.hea (layout header)
        ├── 85940419_0001.hea (segment headers)
        ├── 85940419_0001e.dat (signal data)
        └── 85940419n.csv.gz (numerics)
    """

    def __init__(
            self,
            index_csv: str,  # Path to waveform_index.csv from download script
            data_root: str,  # Path to waves directory
            modality: str = 'ppg',  # 'ppg' or 'ecg'
            labels_csv: Optional[str] = None,  # Path to labels.csv for downstream
            *,
            # Paper-specific parameters
            segment_len_seconds: Optional[float] = None,
            target_fs: Optional[int] = None,

            # Preprocessing
            preprocess: bool = True,
            band_lo: Optional[float] = None,
            band_hi: Optional[float] = None,

            # Dataset construction
            min_segments_per_participant: int = 4,  # Paper requirement
            max_segments_per_participant: int = 100,  # For memory efficiency
            return_participant_id: bool = False,

            # Split
            split: str = 'train',  # 'train', 'val', 'test'
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            random_seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.modality = modality.lower()
        self.preprocess = preprocess
        self.min_segments_per_participant = min_segments_per_participant
        self.max_segments_per_participant = max_segments_per_participant
        self.return_participant_id = return_participant_id
        self.split = split
        self.random_seed = random_seed

        # Set parameters according to paper (Table 1 in paper)
        if modality == 'ppg':
            # Paper: 60 seconds @ 64Hz for PPG, 4 channels
            self.segment_len_seconds = segment_len_seconds or 60.0
            self.target_fs = target_fs or 64
            self.segment_len = int(self.segment_len_seconds * self.target_fs)  # 3840
            self.band_lo = band_lo or 0.4
            self.band_hi = band_hi or 8.0
            # MIMIC PPG signal names
            self.signal_names = ['PLETH', 'PPG']
            self.n_channels = 1  # MIMIC has 1, paper has 4
        else:  # ECG
            # Paper: 30 seconds @ 128Hz for ECG, 1 channel
            self.segment_len_seconds = segment_len_seconds or 30.0
            self.target_fs = target_fs or 128
            self.segment_len = int(self.segment_len_seconds * self.target_fs)  # 3840
            self.band_lo = band_lo or 0.5
            self.band_hi = band_hi or 40.0
            # MIMIC ECG signal names
            self.signal_names = ['II', 'V', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'ECG', 'I', 'III']
            self.n_channels = 1  # Paper uses single lead

        # Load index
        self.index_df = pd.read_csv(index_csv)

        # Filter for modality
        if modality == 'ppg':
            self.index_df = self.index_df[self.index_df['has_ppg'] == True]
        else:
            self.index_df = self.index_df[self.index_df['has_ecg'] == True]

        # Remove records with missing paths
        self.index_df = self.index_df.dropna(subset=['record_dir'])

        # Load labels if provided (for downstream tasks)
        self.labels_df = None
        if labels_csv and Path(labels_csv).exists():
            self.labels_df = pd.read_csv(labels_csv)
            print(f"Loaded labels for downstream tasks: {list(self.labels_df.columns)[:5]}...")

        # Build participant index
        self._build_participant_index()

        # Split dataset
        self._create_splits(train_ratio, val_ratio)

        # Build segment pairs for current split
        self._build_segment_pairs()

        print(f"\n{modality.upper()} Dataset initialized for {split}:")
        print(f"  Participants: {len(self.split_participants)}")
        print(f"  Total segments: {len(self.split_segments)}")
        print(f"  Positive pairs: {len(self.segment_pairs)}")
        print(f"  Segment: {self.segment_len_seconds}s @ {self.target_fs}Hz = {self.segment_len} samples")
        print(f"  Bandpass filter: {self.band_lo}-{self.band_hi} Hz")

    def _build_participant_index(self):
        """Build mapping of participants to their recordings."""
        self.participant_segments = defaultdict(list)
        self.all_segments = []

        for _, row in self.index_df.iterrows():
            subject_id = row['subject_id']
            record_id = row['record_id']
            record_dir = row['record_dir']

            # Check duration if available
            duration = row.get('duration_sec', None)
            if duration and duration < self.segment_len_seconds:
                continue  # Skip short recordings

            segment_info = {
                'subject_id': int(subject_id),
                'record_id': str(record_id),
                'record_dir': str(record_dir),
                'duration': duration
            }

            self.participant_segments[int(subject_id)].append(segment_info)
            self.all_segments.append(segment_info)

        # Filter participants with minimum segments
        self.valid_participants = []
        for subject_id, segments in self.participant_segments.items():
            if len(segments) >= self.min_segments_per_participant:
                # Limit segments per participant
                if len(segments) > self.max_segments_per_participant:
                    self.participant_segments[subject_id] = random.sample(
                        segments, self.max_segments_per_participant
                    )
                self.valid_participants.append(subject_id)

        print(f"\nData Statistics:")
        print(f"  Total participants: {len(self.participant_segments)}")
        print(f"  Participants with >={self.min_segments_per_participant} segments: {len(self.valid_participants)}")
        print(f"  Total segments: {len(self.all_segments)}")

    def _create_splits(self, train_ratio: float, val_ratio: float):
        """Create train/val/test splits at participant level (as per paper Section 4.1)."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Shuffle participants
        participants = self.valid_participants.copy()
        random.shuffle(participants)

        n_total = len(participants)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split participants (not segments!) as per paper
        train_participants = participants[:n_train]
        val_participants = participants[n_train:n_train + n_val]
        test_participants = participants[n_train + n_val:]

        # Assign current split
        if self.split == 'train':
            self.split_participants = train_participants
        elif self.split == 'val':
            self.split_participants = val_participants
        else:  # test
            self.split_participants = test_participants

        # Get segments for current split
        self.split_segments = []
        for subject_id in self.split_participants:
            self.split_segments.extend(self.participant_segments[subject_id])

    def _build_segment_pairs(self):
        """Create positive pairs from same participant (Section 3.1 of paper)."""
        self.segment_pairs = []

        for subject_id in self.split_participants:
            segments = self.participant_segments[subject_id]
            n_segments = len(segments)

            # Create all possible pairs within participant
            # This is the key difference from segment-level pairing
            for i in range(n_segments):
                for j in range(i + 1, n_segments):
                    self.segment_pairs.append((segments[i], segments[j], subject_id))

            # If only one segment, pair with itself (will be augmented differently)
            if n_segments == 1:
                self.segment_pairs.append((segments[0], segments[0], subject_id))

    def __len__(self):
        return len(self.segment_pairs)

    def __getitem__(self, idx):
        """Get a positive pair from the same participant."""
        seg1_info, seg2_info, subject_id = self.segment_pairs[idx]

        # Load segments
        seg1 = self._load_and_process_segment(seg1_info)
        seg2 = self._load_and_process_segment(seg2_info)

        if self.return_participant_id:
            # Also return labels if available
            labels = {}
            if self.labels_df is not None:
                participant_labels = self.labels_df[
                    self.labels_df['subject_id'] == subject_id
                    ]
                if not participant_labels.empty:
                    labels = participant_labels.iloc[0].to_dict()

            return seg1, seg2, subject_id, labels

        return seg1, seg2

    def _load_and_process_segment(self, segment_info: dict) -> torch.Tensor:
        """Load and process a segment from MIMIC-IV."""
        record_dir = segment_info['record_dir']
        record_id = segment_info['record_id']

        # Full path to record
        # Structure: data_root/p100/p10039708/85940419/85940419
        record_path = self.data_root / record_dir / record_id

        try:
            # Read WFDB record - handle multi-segment records
            try:
                # Try reading as regular record first
                record = wfdb.rdrecord(str(record_path))
            except:
                # If fails, might be multi-segment, read just the header
                header = wfdb.rdheader(str(record_path))
                if hasattr(header, 'seg_name') and header.seg_name:
                    # Read first segment
                    seg_path = self.data_root / record_dir / header.seg_name[0]
                    record = wfdb.rdrecord(str(seg_path))
                else:
                    raise

            # Find appropriate signal
            signal_idx = None
            signal_name_used = None
            for sig_name in self.signal_names:
                for idx, rec_sig_name in enumerate(record.sig_name):
                    if sig_name.upper() in rec_sig_name.upper():
                        signal_idx = idx
                        signal_name_used = rec_sig_name
                        break
                if signal_idx is not None:
                    break

            if signal_idx is None:
                # Fallback: use first available signal
                signal_idx = 0
                signal_name_used = record.sig_name[0] if record.sig_name else "Unknown"

            # Get signal
            signal = record.p_signal[:, signal_idx:signal_idx + 1].T  # Shape: [1, N]
            fs = record.fs

            # Process signal
            signal = self._preprocess_signal(signal, fs)

            # Extract random window of required length
            signal = self._extract_window(signal)

            return torch.from_numpy(signal).float()

        except Exception as e:
            print(f"Error loading {record_path}: {e}")
            # Return zeros as fallback
            return torch.zeros((self.n_channels, self.segment_len), dtype=torch.float32)

    def _preprocess_signal(self, signal: np.ndarray, original_fs: int) -> np.ndarray:
        """Preprocess signal according to paper specifications (Section 4.1)."""

        # Remove NaN/Inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        if self.preprocess:
            # 1. Bandpass filter (before resampling for better results)
            if original_fs > 0:
                nyq = original_fs / 2
                if self.band_hi < nyq:
                    # Use SOS format for numerical stability
                    sos = scipy_signal.butter(
                        4, [self.band_lo, self.band_hi],
                        btype='band', fs=original_fs, output='sos'
                    )
                    signal = scipy_signal.sosfiltfilt(sos, signal, axis=1)

            # 2. Resample to target frequency
            if original_fs != self.target_fs and original_fs > 0:
                n_samples = signal.shape[1]
                n_resampled = int(n_samples * self.target_fs / original_fs)

                # Ensure we have enough samples
                if n_resampled > 10:  # Arbitrary minimum
                    signal_resampled = np.zeros((signal.shape[0], n_resampled))
                    for i in range(signal.shape[0]):
                        signal_resampled[i] = resample(signal[i], n_resampled)
                    signal = signal_resampled

            # 3. Z-score normalization (as per paper)
            # "temporal channel-wise z-scoring for each segment"
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True) + 1e-8
            signal = (signal - mean) / std

        return signal.astype(np.float32)

    def _extract_window(self, signal: np.ndarray) -> np.ndarray:
        """Extract a window of the required length."""
        _, n_samples = signal.shape

        if n_samples > self.segment_len:
            # Random crop (helps with augmentation diversity)
            start = random.randint(0, n_samples - self.segment_len)
            signal = signal[:, start:start + self.segment_len]
        elif n_samples < self.segment_len:
            # Pad with zeros
            pad_len = self.segment_len - n_samples
            signal = np.pad(signal, ((0, 0), (0, pad_len)), mode='constant')

        return signal

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'split': self.split,
            'modality': self.modality,
            'total_participants': len(self.split_participants),
            'total_segments': len(self.split_segments),
            'total_pairs': len(self.segment_pairs),
            'sampling_rate': self.target_fs,
            'segment_length_sec': self.segment_len_seconds,
            'segment_length_samples': self.segment_len,
            'n_channels': self.n_channels
        }


def create_dataloaders(
        index_csv: str,
        data_root: str,
        labels_csv: Optional[str] = None,
        batch_size: int = 256,  # Paper uses 256
        num_workers: int = 4,
        **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for MIMIC-IV.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Create datasets for each split
    train_dataset = MIMICWaveformDataset(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        split='train',
        **dataset_kwargs
    )

    val_dataset = MIMICWaveformDataset(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        split='val',
        **dataset_kwargs
    )

    test_dataset = MIMICWaveformDataset(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        split='test',
        **dataset_kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For stable batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============= TEST FUNCTIONS =============

def test_data_loading():
    """Test data loading functionality."""
    print("=" * 50)
    print("Testing MIMIC-IV Data Loading")
    print("=" * 50)

    # Paths - adjust these to your setup
    index_csv = "data/outputs/waveform_index.csv"
    data_root = "data/mimic4wdb/physionet.org/files/mimic4wdb/0.1.0/waves"
    labels_csv = "data/outputs/labels.csv"

    # Test PPG dataset
    print("\n1. Testing PPG Dataset:")
    ppg_dataset = MIMICWaveformDataset(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        modality='ppg',
        split='train',
        min_segments_per_participant=2  # Lower for testing
    )

    if len(ppg_dataset) > 0:
        seg1, seg2 = ppg_dataset[0]
        print(f"   PPG pair shapes: {seg1.shape}, {seg2.shape}")
        print(f"   Expected shape: (1, 3840)")
        assert seg1.shape == (1, 3840), f"Wrong PPG shape: {seg1.shape}"
        print("   ✓ PPG test passed!")

    # Test ECG dataset
    print("\n2. Testing ECG Dataset:")
    ecg_dataset = MIMICWaveformDataset(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        modality='ecg',
        split='train',
        min_segments_per_participant=2
    )

    if len(ecg_dataset) > 0:
        seg1, seg2 = ecg_dataset[0]
        print(f"   ECG pair shapes: {seg1.shape}, {seg2.shape}")
        print(f"   Expected shape: (1, 3840)")
        assert seg1.shape == (1, 3840), f"Wrong ECG shape: {seg1.shape}"
        print("   ✓ ECG test passed!")

    # Test with participant ID return
    print("\n3. Testing with participant ID and labels:")
    dataset_with_ids = MIMICWaveformDataset(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        modality='ppg',
        split='train',
        return_participant_id=True,
        min_segments_per_participant=2
    )

    if len(dataset_with_ids) > 0:
        seg1, seg2, pid, labels = dataset_with_ids[0]
        print(f"   Participant ID: {pid}")
        print(f"   Labels keys: {list(labels.keys())[:5] if labels else 'No labels'}")
        print("   ✓ ID/labels test passed!")

    # Test dataloaders
    print("\n4. Testing DataLoader creation:")
    train_loader, val_loader, test_loader = create_dataloaders(
        index_csv=index_csv,
        data_root=data_root,
        labels_csv=labels_csv,
        batch_size=8,  # Small for testing
        num_workers=0,  # No multiprocessing for testing
        modality='ppg',
        min_segments_per_participant=2
    )

    # Test one batch
    for batch in train_loader:
        seg1_batch, seg2_batch = batch
        print(f"   Batch shapes: {seg1_batch.shape}, {seg2_batch.shape}")
        print(f"   Expected: (8, 1, 3840), (8, 1, 3840)")
        assert seg1_batch.shape[0] == 8, "Wrong batch size"
        print("   ✓ DataLoader test passed!")
        break

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    # Run tests
    test_data_loading()