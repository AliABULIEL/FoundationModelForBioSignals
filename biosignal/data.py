# biosignal/data.py
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample  # <-- NEW

class FolderPerParticipant(Dataset):
    """
    Generic folder-per-participant loader.

    root/
        P0001/seg_000.npy
        P0001/seg_001.npy
        …

    Each .npy file is float32 [C,L] or [L].
    Returns two segments from the same participant (positive pair).
    """

    def __init__(
        self,
        root: str,
        segment_len: int,
        *,
        preprocess: bool = False,          # NEW flag
        native_fs: int = 256,              # sampling rate of stored files
        target_fs: int = 64,               # desired rate after resample
        band_lo: float = 0.4,              # band-pass lower edge (Hz)
        band_hi: float = 8.0               # band-pass upper edge (Hz)
    ):
        self.root = Path(root)
        self.segment_len = segment_len
        self.preprocess = preprocess

        # prepare band-pass filter only if we’ll use it
        if preprocess and (band_lo or band_hi):
            b, a = butter(
                N=4,
                Wn=[band_lo / (native_fs / 2), band_hi / (native_fs / 2)],
                btype="band",
            )
            self._bandpass = lambda sig: filtfilt(b, a, sig, axis=-1)
        else:
            self._bandpass = lambda sig: sig

        self.native_fs = native_fs
        self.target_fs = target_fs

        # Build index:  {participant_id: [file1, file2, …]}
        self.by_pid = {
            d.name: list(d.glob("*.npy"))
            for d in self.root.iterdir()
            if d.is_dir() and len(list(d.glob("*.npy"))) >= 2
        }
        self.pids = list(self.by_pid.keys())

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        f1, f2 = random.sample(self.by_pid[pid], 2)
        return self._load_segment(f1), self._load_segment(f2)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _load_segment(self, fpath: Path) -> torch.Tensor:
        """Load ONE segment, apply optional pre-processing, then crop/pad."""
        x = np.load(fpath).astype(np.float32)
        if x.ndim == 1:               # → [C,L] where C=1
            x = x[None, :]

        if self.preprocess:
            x = self._preprocess_segment(x)  # <-- NEW

        # crop / pad to fixed length
        C, L = x.shape
        if L > self.segment_len:
            start = random.randint(0, L - self.segment_len)
            x = x[:, start : start + self.segment_len]
        elif L < self.segment_len:
            pad = self.segment_len - L
            x = np.pad(x, ((0, 0), (0, pad)))

        return torch.from_numpy(x)

    # -------------- NEW ------------------------------------------------ #
    def _preprocess_segment(self, x: np.ndarray) -> np.ndarray:
        """
        Band-pass filter → resample to target_fs → per-segment z-score.
        x is [C,L] NumPy float32.
        """
        # 1) band-pass
        x = self._bandpass(x)

        # 2) resample (only if target_fs differs)
        if self.target_fs != self.native_fs:
            new_len = int(x.shape[1] * self.target_fs / self.native_fs)
            x = resample(x, new_len, axis=1)

        # 3) per-segment z-score   (avoid division by ~0)
        mean = x.mean(axis=1, keepdims=True)
        std  = x.std(axis=1,  keepdims=True) + 1e-6
        x = (x - mean) / std
        return x
