# biosignal/augment.py

import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import CubicSpline


class BiosignalAugment:
    """
    Augmentation module adapted for BUT PPG dataset.
    BUT PPG: 10-second signals at 30Hz (PPG) or 1000Hz (ECG)
    """

    def __init__(self, modality='ppg', signal_type='but_ppg'):
        """
        Initialize augmentation module with dataset-specific parameters.

        Args:
            modality: 'ppg' or 'ecg'
            signal_type: 'but_ppg' for BUT dataset, 'paper' for paper specs
        """
        self.modality = modality.lower()
        self.signal_type = signal_type.lower()

        if signal_type == 'but_ppg':
            if modality.lower() == 'ppg':
                # BUT PPG specific: 30Hz, 10 seconds, smartphone data
                # Slightly different probabilities due to shorter signals
                self.cutout_p = 0.3  # Lower for shorter signals
                self.magnitude_warp_p = 0.3  # Higher for smartphone variability
                self.gaussian_noise_p = 0.3  # Higher for smartphone noise
                self.channel_permute_p = 0.0  # BUT PPG is single channel
                self.time_warp_p = 0.2  # Moderate time warping

                # BUT PPG specific parameters
                self.sampling_rate = 30  # Hz
                self.signal_duration = 10  # seconds
                self.expected_length = 300  # samples

            else:  # ECG
                # BUT ECG: 1000Hz, 10 seconds
                self.cutout_p = 0.4
                self.magnitude_warp_p = 0.3
                self.gaussian_noise_p = 0.3
                self.channel_permute_p = 0.0  # Single channel
                self.time_warp_p = 0.2

                # BUT ECG specific parameters
                self.sampling_rate = 1000  # Hz
                self.signal_duration = 10  # seconds
                self.expected_length = 10000  # samples
        else:
            # Original paper settings
            if modality.lower() == 'ppg':
                self.cutout_p = 0.4
                self.magnitude_warp_p = 0.25
                self.gaussian_noise_p = 0.25
                self.channel_permute_p = 0.25
                self.time_warp_p = 0.15
                self.sampling_rate = 64
                self.expected_length = 3840
            else:  # ECG
                self.cutout_p = 0.8
                self.magnitude_warp_p = 0.5
                self.gaussian_noise_p = 0.5
                self.channel_permute_p = 0.0
                self.time_warp_p = 0.3
                self.sampling_rate = 128
                self.expected_length = 3840

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout augmentation adapted for signal length.
        """
        C, L = x.shape

        # Adapt cutout length based on signal duration
        if self.signal_type == 'but_ppg':
            # For 10-second signals, use 5-15% cutout
            min_cutout = int(0.05 * L)
            max_cutout = int(0.15 * L)
        else:
            # Original: 10-20% for 60-second signals
            min_cutout = int(0.1 * L)
            max_cutout = int(0.2 * L)

        # Ensure we have valid range
        if min_cutout >= max_cutout:
            max_cutout = min_cutout + 1
        if max_cutout > L:
            max_cutout = L // 4
            min_cutout = L // 8

        cutout_length = random.randint(min_cutout, max_cutout)

        # Random starting position
        if L > cutout_length:
            start_idx = random.randint(0, L - cutout_length)
        else:
            start_idx = 0
            cutout_length = L // 2

        # Create a copy to avoid in-place modification
        x_aug = x.clone()

        # Mask the segment (set to zero)
        x_aug[:, start_idx:start_idx + cutout_length] = 0.0

        return x_aug

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping with adaptive parameters for signal length.
        """
        C, L = x.shape

        # Fewer knots for shorter signals
        if self.signal_type == 'but_ppg' and L < 500:
            n_knots = 3  # Fewer knots for short signals
        else:
            n_knots = 4

        # Generate random knots
        knot_xs = np.linspace(0, L - 1, n_knots)

        # Adapt warping strength based on modality
        if self.modality == 'ppg' and self.signal_type == 'but_ppg':
            # Smartphone PPG might have more variability
            knot_ys = np.random.uniform(0.7, 1.3, n_knots)
        else:
            knot_ys = np.random.uniform(0.5, 2.0, n_knots)

        # Ensure endpoints are less extreme
        knot_ys[0] = np.clip(knot_ys[0], 0.8, 1.2)
        knot_ys[-1] = np.clip(knot_ys[-1], 0.8, 1.2)

        # Create smooth curve using cubic spline
        if n_knots >= 4:
            spline = CubicSpline(knot_xs, knot_ys)
        else:
            # Use linear interpolation for fewer knots
            from scipy.interpolate import interp1d
            spline = interp1d(knot_xs, knot_ys, kind='quadratic', fill_value='extrapolate')

        warp_curve = spline(np.arange(L))

        # Convert to tensor
        warp_tensor = torch.from_numpy(warp_curve.astype(np.float32)).to(x.device)

        # Apply channel-wise multiplication
        x_warped = x * warp_tensor.unsqueeze(0)

        return x_warped

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping adapted for signal characteristics.
        """
        C, L = x.shape

        # Adapt number of knots based on signal length
        if L < 500:  # Short signals (like BUT PPG at 30Hz)
            n_knots = 3
        else:
            n_knots = 4

        # Original time points
        orig_times = np.linspace(0, L - 1, n_knots)

        # Warped time points (with constraints to avoid extreme warping)
        cumulative_shift = 0
        warped_times = [0]  # First point stays fixed

        for i in range(1, n_knots - 1):
            # Smaller shifts for shorter signals
            if self.signal_type == 'but_ppg':
                max_shift = int(0.05 * L / n_knots)  # Smaller shifts
            else:
                max_shift = int(0.1 * L / n_knots)

            shift = random.randint(-max_shift, max_shift)
            cumulative_shift += shift
            warped_times.append(orig_times[i] + cumulative_shift)

        warped_times.append(L - 1)  # Last point stays fixed
        warped_times = np.array(warped_times)

        # Create warping function
        if n_knots >= 4:
            warp_func = CubicSpline(warped_times, orig_times)
        else:
            from scipy.interpolate import interp1d
            warp_func = interp1d(warped_times, orig_times, kind='linear', fill_value='extrapolate')

        # Generate new time indices
        new_indices = warp_func(np.arange(L))
        new_indices = np.clip(new_indices, 0, L - 1)

        # Apply warping
        x_np = x.cpu().numpy()
        x_warped = np.zeros_like(x_np)

        for c in range(C):
            x_warped[c] = np.interp(new_indices, np.arange(L), x_np[c])

        return torch.from_numpy(x_warped).float().to(x.device)

    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise adapted for signal characteristics.
        """
        # Estimate signal std for adaptive noise
        signal_std = torch.std(x)

        # Adapt noise level based on signal type
        if self.signal_type == 'but_ppg' and self.modality == 'ppg':
            # Smartphone PPG might need different noise levels
            noise_level = random.uniform(0.02, 0.08)  # Slightly higher for smartphone
        else:
            noise_level = random.uniform(0.01, 0.05)

        noise = torch.randn_like(x) * signal_std * noise_level

        return x + noise

    def add_motion_artifact(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add motion artifact simulation (relevant for BUT PPG which has motion labels).
        Only applied occasionally.
        """
        if random.random() > 0.2:  # Only 20% chance
            return x

        C, L = x.shape

        # Create low-frequency motion artifact
        t = np.linspace(0, 2 * np.pi, L)

        # Random frequency between 0.5-2 Hz
        freq = random.uniform(0.5, 2.0)
        artifact = np.sin(freq * t) * random.uniform(0.1, 0.3)

        # Convert to tensor and add
        artifact_tensor = torch.from_numpy(artifact.astype(np.float32)).to(x.device)

        # Add to all channels
        x_with_artifact = x + artifact_tensor.unsqueeze(0) * torch.std(x)

        return x_with_artifact

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic sequence of augmentations.
        """
        # Ensure tensor is float32
        x = x.float()

        # Validate input shape
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Add channel dimension

        # Store original shape
        original_shape = x.shape

        # Apply augmentations stochastically
        if random.random() < self.cutout_p:
            x = self.cutout(x)

        if random.random() < self.magnitude_warp_p:
            x = self.magnitude_warp(x)

        if random.random() < self.gaussian_noise_p:
            x = self.add_gaussian_noise(x)

        if random.random() < self.time_warp_p:
            x = self.time_warp(x)

        # BUT PPG specific: add motion artifacts occasionally
        if self.signal_type == 'but_ppg' and self.modality == 'ppg':
            x = self.add_motion_artifact(x)

        # Channel permutation (not used for BUT PPG - single channel)
        if self.channel_permute_p > 0 and x.shape[0] > 1:
            if random.random() < self.channel_permute_p:
                x = self.channel_permute(x)

        # Ensure output maintains original length
        if x.shape[-1] != original_shape[-1]:
            x = F.interpolate(
                x.unsqueeze(0),
                size=original_shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(0)

        return x

    def channel_permute(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly permute channels (not used for BUT PPG)."""
        if x.shape[0] <= 1:
            return x
        C = x.shape[0]
        perm = torch.randperm(C)
        return x[perm]


class DualAugment:
    """
    Wrapper to create two different augmented views for contrastive learning.
    """

    def __init__(self, modality='ppg', signal_type='but_ppg'):
        self.aug1 = BiosignalAugment(modality=modality, signal_type=signal_type)
        self.aug2 = BiosignalAugment(modality=modality, signal_type=signal_type)

    def __call__(self, x: torch.Tensor):
        """Returns two differently augmented versions of input."""
        return self.aug1(x.clone()), self.aug2(x.clone())


# Factory function
def get_augmentation_module(modality='ppg', dataset='but_ppg'):
    """
    Factory function to get the appropriate augmentation module.

    Args:
        modality: 'ppg' or 'ecg'
        dataset: 'but_ppg' or 'paper'

    Returns:
        BiosignalAugment instance configured for the dataset
    """
    return BiosignalAugment(modality=modality, signal_type=dataset)

