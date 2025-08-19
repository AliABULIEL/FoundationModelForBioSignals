# biosignal/data.py

from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from typing import Optional, Tuple, List, Dict
import pandas as pd
from collections import defaultdict


class FolderPerParticipant(Dataset):
    """
    Dataset adapter for BUT PPG database structure.

    BUT PPG structure:
    - Folders named: 100001, 100002, etc.
    - First 3 digits = participant ID (100, 101, 102...)
    - Last 3 digits = recording number (001, 002, 003...)

    Each folder contains:
    - {ID}_PPG.npy: PPG signal
    - {ID}_ECG.npy: ECG signal
    """

    def __init__(
            self,
            root: str,
            segment_len: Optional[int] = None,
            modality: str = 'ppg',  # 'ppg' or 'ecg'
            dataset_type: str = 'but_ppg',  # 'but_ppg' or 'paper'
            *,
            preprocess: bool = True,
            native_fs: Optional[int] = None,
            target_fs: Optional[int] = None,
            band_lo: Optional[float] = None,
            band_hi: Optional[float] = None,
            min_recordings_per_participant: int = 2,  # BUT PPG has fewer recordings per participant
            return_participant_id: bool = False,
            quality_csv: Optional[str] = None,  # Path to quality-hr-ann.csv
            use_only_good_quality: bool = True
    ):
        self.root = Path(root)
        self.modality = modality.lower()
        self.dataset_type = dataset_type.lower()
        self.preprocess = preprocess
        self.min_recordings_per_participant = min_recordings_per_participant
        self.return_participant_id = return_participant_id
        self.use_only_good_quality = use_only_good_quality

        # Set dataset-specific parameters
        if dataset_type == 'but_ppg':
            if modality == 'ppg':
                # BUT PPG: 30Hz, 10 seconds
                self.native_fs = native_fs or 30
                self.target_fs = target_fs or 30  # Keep original
                self.segment_len = segment_len or 300  # 10s * 30Hz
                self.band_lo = band_lo or 0.5
                self.band_hi = band_hi or 5.0
            else:  # ECG
                # BUT ECG: 1000Hz, 10 seconds
                self.native_fs = native_fs or 1000
                self.target_fs = target_fs or 128  # Downsample for efficiency
                self.segment_len = segment_len or 1280  # 10s * 128Hz after resampling
                self.band_lo = band_lo or 0.5
                self.band_hi = band_hi or 40.0
        else:  # paper settings
            if modality == 'ppg':
                self.native_fs = native_fs or 256
                self.target_fs = target_fs or 64
                self.segment_len = segment_len or 3840
                self.band_lo = band_lo or 0.4
                self.band_hi = band_hi or 8.0
            else:  # ECG
                self.native_fs = native_fs or 512
                self.target_fs = target_fs or 128
                self.segment_len = segment_len or 3840
                self.band_lo = band_lo or 0.5
                self.band_hi = band_hi or 40.0

        # Load quality annotations if provided
        self.quality_labels = {}
        if quality_csv and Path(quality_csv).exists():
            self._load_quality_labels(quality_csv)

        # Initialize filter
        self._init_filter()

        # Build participant-recording mapping
        self._build_participant_index()

        # Create segment pairs
        self._build_segment_pairs()

    def _load_quality_labels(self, csv_path: str):
        """Load quality labels from CSV file."""
        df = pd.read_csv(csv_path, header=None, names=['recording_id', 'quality', 'hr'])
        for _, row in df.iterrows():
            # Ensure recording_id is string and padded
            rec_id = str(int(row['recording_id'])).zfill(6)
            self.quality_labels[rec_id] = {
                'quality': int(row['quality']),
                'hr': float(row['hr'])
            }
        print(f"Loaded quality labels for {len(self.quality_labels)} recordings")

    def _init_filter(self):
        """Initialize bandpass filter coefficients."""
        if self.preprocess and self.band_lo and self.band_hi:
            nyq = self.native_fs / 2

            # Adjust frequencies if they exceed Nyquist
            if self.band_lo >= nyq or self.band_hi >= nyq:
                print(f"Adjusting filter frequencies for Nyquist limit ({nyq}Hz)")
                self.band_hi = min(self.band_hi, nyq * 0.9)
                self.band_lo = min(self.band_lo, self.band_hi * 0.01)

            self.filter_b, self.filter_a = butter(
                N=4,
                Wn=[self.band_lo / nyq, self.band_hi / nyq],
                btype='band'
            )
            self.use_bandpass = True
        else:
            self.use_bandpass = False

    def _build_participant_index(self):
        """
        Parse BUT PPG structure to map participants to their recordings.
        """
        self.participant_recordings = defaultdict(list)
        self.all_recordings = []
        self.participants = []

        # Scan all directories
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue

            recording_id = folder.name

            # Extract participant ID (first 3 digits)
            try:
                participant_id = recording_id[:3]
            except:
                continue

            # Check if required files exist
            if self.modality == 'ppg':
                signal_file = folder / f"{recording_id}_PPG.npy"
            else:  # ECG
                signal_file = folder / f"{recording_id}_ECG.npy"

            if not signal_file.exists():
                continue

            # Filter by quality if labels are available
            if self.use_only_good_quality and self.quality_labels:
                if recording_id not in self.quality_labels:
                    continue
                if self.quality_labels[recording_id]['quality'] == 0:
                    continue  # Skip poor quality

            self.participant_recordings[participant_id].append(recording_id)
            self.all_recordings.append(recording_id)

        # Filter participants with minimum recordings
        for pid, recordings in self.participant_recordings.items():
            if len(recordings) >= self.min_recordings_per_participant:
                self.participants.append(pid)

        print(f"BUT PPG Dataset Summary:")
        print(f"  Total recordings: {len(self.all_recordings)}")
        print(f"  Total participants: {len(self.participant_recordings)}")
        print(f"  Participants with >= {self.min_recordings_per_participant} recordings: {len(self.participants)}")

        # Show distribution
        recordings_counts = [len(recs) for recs in self.participant_recordings.values()]
        if recordings_counts:
            print(f"  Recordings per participant: min={min(recordings_counts)}, "
                  f"max={max(recordings_counts)}, avg={np.mean(recordings_counts):.1f}")

    def _build_segment_pairs(self):
        """Create all possible pairs from same participant."""
        self.segment_pairs = []

        for pid in self.participants:
            recordings = self.participant_recordings[pid]
            n_recordings = len(recordings)

            # Create all possible pairs
            for i in range(n_recordings):
                for j in range(i + 1, n_recordings):
                    self.segment_pairs.append((pid, recordings[i], recordings[j]))

        print(f"  Created {len(self.segment_pairs)} positive pairs")

        # If too few pairs, allow same recording with different augmentations
        if len(self.segment_pairs) < 100:
            print(f"  WARNING: Only {len(self.segment_pairs)} pairs available!")
            print(f"  Consider using segment-level augmentation or reducing min_recordings_per_participant")

    def __len__(self):
        # If no valid pairs, return number of recordings for fallback mode
        if len(self.segment_pairs) == 0:
            return len(self.all_recordings)
        return len(self.segment_pairs)

    def __getitem__(self, idx):
        """Get a positive pair or single recording with augmentation."""

        # If we have valid pairs
        if len(self.segment_pairs) > 0:
            pid, rec1_id, rec2_id = self.segment_pairs[idx % len(self.segment_pairs)]

            # Load both recordings
            seg1 = self._load_recording(rec1_id)
            seg2 = self._load_recording(rec2_id)

            if self.return_participant_id:
                return seg1, seg2, pid
            return seg1, seg2

        # Fallback: return same recording twice (will be augmented differently)
        else:
            rec_id = self.all_recordings[idx % len(self.all_recordings)]
            seg = self._load_recording(rec_id)

            if self.return_participant_id:
                pid = rec_id[:3]
                return seg, seg.clone(), pid
            return seg, seg.clone()

    def _load_recording(self, recording_id: str) -> torch.Tensor:
        """Load and preprocess a single recording."""
        folder = self.root / recording_id

        # Load appropriate signal
        if self.modality == 'ppg':
            signal_file = folder / f"{recording_id}_PPG.npy"
        else:  # ECG
            signal_file = folder / f"{recording_id}_ECG.npy"

        signal = np.load(signal_file).astype(np.float32)

        # Ensure correct shape [C, L]
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        # Preprocess if needed
        if self.preprocess:
            signal = self._apply_preprocessing(signal)

        # Adjust length
        signal = self._crop_or_pad(signal, self.segment_len)

        return torch.from_numpy(signal).float()

    def _apply_preprocessing(self, signal: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline."""
        # Bandpass filter
        if self.use_bandpass:
            signal = filtfilt(self.filter_b, self.filter_a, signal, axis=-1)

        # Resample if needed
        if self.target_fs != self.native_fs:
            C, L = signal.shape
            new_length = int(L * self.target_fs / self.native_fs)

            resampled = np.zeros((C, new_length), dtype=np.float32)
            for c in range(C):
                resampled[c] = resample(signal[c], new_length)
            signal = resampled

        # Z-score normalization
        eps = 1e-8
        for c in range(signal.shape[0]):
            mean = np.mean(signal[c])
            std = np.std(signal[c]) + eps
            signal[c] = (signal[c] - mean) / std

        return signal

    def _crop_or_pad(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Crop or pad to target length."""
        C, L = signal.shape

        if L > target_length:
            # Center crop for consistency
            start = (L - target_length) // 2
            signal = signal[:, start:start + target_length]
        elif L < target_length:
            # Pad equally on both sides
            pad_total = target_length - L
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            signal = np.pad(signal, ((0, 0), (pad_left, pad_right)), mode='constant')

        return signal

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'total_participants': len(self.participant_recordings),
            'valid_participants': len(self.participants),
            'total_recordings': len(self.all_recordings),
            'total_pairs': len(self.segment_pairs),
            'modality': self.modality,
            'sampling_rate': self.target_fs,
            'segment_length': self.segment_len
        }


# Utility function for BUT PPG
def create_but_ppg_dataset(
        root: str,
        modality: str = 'ppg',
        quality_csv: Optional[str] = None,
        min_recordings: int = 2,
        **kwargs
) -> FolderPerParticipant:
    """
    Create dataset specifically for BUT PPG database.

    Args:
        root: Path to PPG folder containing recording directories
        modality: 'ppg' or 'ecg'
        quality_csv: Path to quality-hr-ann.csv file
        min_recordings: Minimum recordings per participant
    """
    return FolderPerParticipant(
        root=root,
        modality=modality,
        dataset_type='but_ppg',
        min_recordings_per_participant=min_recordings,
        quality_csv=quality_csv,
        **kwargs
    )


# Test function
if __name__ == "__main__":
    # Test with mock data structure
    dataset = create_but_ppg_dataset(
        root="data/PPG",  # Adjust to your path
        modality='ppg',
        quality_csv='quality-hr-ann.csv',  # Optional
        min_recordings=2
    )

    print(f"\nDataset created successfully!")
    print(f"Statistics: {dataset.get_statistics()}")

    if len(dataset) > 0:
        # Test loading a pair
        seg1, seg2 = dataset[0]
        print(f"\nSample pair shapes: {seg1.shape}, {seg2.shape}")