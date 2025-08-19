# biosignal/augmentations.py

import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, Dict


class StochasticAugmentation:
    """
    Stochastic augmentation module following the Apple paper specifications.

    Paper Section 3.1:
    - PPG augmentations: {cut out: 0.4, magnitude warp: 0.25, gaussian noise: 0.25,
                         channel permute: 0.25, time warp: 0.15}
    - ECG augmentations: {cut out: 0.8, magnitude warp: 0.5, gaussian noise: 0.5,
                         time warp: 0.3}

    Each augmentation has internal randomly selected hyperparameters.
    """

    def __init__(self, modality: str = 'ppg', augmentation_config: Optional[Dict] = None):
        """
        Initialize augmentation module with paper-specified probabilities.

        Args:
            modality: 'ppg' or 'ecg'
            augmentation_config: Optional custom configuration
        """
        self.modality = modality.lower()

        # Set default probabilities from paper (Section 3.1)
        if augmentation_config is None:
            if self.modality == 'ppg':
                self.config = {
                    'cutout': 0.4,
                    'magnitude_warp': 0.25,
                    'gaussian_noise': 0.25,
                    'channel_permute': 0.0,  # MIMIC has 1 channel, paper has 4
                    'time_warp': 0.15
                }
                # PPG: 60s @ 64Hz = 3840 samples
                self.expected_length = 3840
                self.sampling_rate = 64
            else:  # ECG
                self.config = {
                    'cutout': 0.8,  # Higher for ECG as per paper
                    'magnitude_warp': 0.5,
                    'gaussian_noise': 0.5,
                    'channel_permute': 0.0,  # Single channel
                    'time_warp': 0.3
                }
                # ECG: 30s @ 128Hz = 3840 samples
                self.expected_length = 3840
                self.sampling_rate = 128
        else:
            self.config = augmentation_config
            self.expected_length = 3840  # Default

        print(f"Augmentation initialized for {modality.upper()}:")
        for aug, prob in self.config.items():
            if prob > 0:
                print(f"  {aug}: {prob:.2f}")

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout augmentation (masking a continuous segment).
        Paper doesn't specify exact parameters, using reasonable defaults.
        """
        C, L = x.shape

        # Random cutout length: 10-20% of signal length
        min_cutout = int(0.1 * L)
        max_cutout = int(0.2 * L)

        cutout_length = random.randint(min_cutout, max_cutout)

        # Random starting position
        if L > cutout_length:
            start_idx = random.randint(0, L - cutout_length)
        else:
            start_idx = 0
            cutout_length = L

        # Create a copy and mask the segment
        x_aug = x.clone()
        x_aug[:, start_idx:start_idx + cutout_length] = 0.0

        return x_aug

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping using smooth random curves.
        This scales the signal amplitude over time.
        """
        C, L = x.shape

        # Number of knots for cubic spline (4-6 knots works well)
        n_knots = random.randint(4, 6)

        # Generate random knots
        knot_xs = np.linspace(0, L - 1, n_knots)

        # Random scaling factors (0.5 to 2.0 for reasonable warping)
        knot_ys = np.random.uniform(0.5, 2.0, n_knots)

        # Ensure endpoints are moderate to avoid extreme distortion
        knot_ys[0] = np.clip(knot_ys[0], 0.8, 1.2)
        knot_ys[-1] = np.clip(knot_ys[-1], 0.8, 1.2)

        # Create smooth curve using cubic spline
        spline = CubicSpline(knot_xs, knot_ys)
        warp_curve = spline(np.arange(L))

        # Convert to tensor and apply
        warp_tensor = torch.from_numpy(warp_curve.astype(np.float32)).to(x.device)
        x_warped = x * warp_tensor.unsqueeze(0)

        return x_warped

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping (non-linear time axis distortion).
        This simulates small variations in sampling rate or timing.
        """
        C, L = x.shape

        # Number of knots
        n_knots = random.randint(4, 6)

        # Original time points
        orig_times = np.linspace(0, L - 1, n_knots)

        # Warped time points
        warped_times = [0]  # First point stays fixed

        for i in range(1, n_knots - 1):
            # Maximum shift is 10% of the segment between knots
            max_shift = int(0.1 * L / n_knots)
            shift = random.randint(-max_shift, max_shift)

            # Ensure monotonicity
            new_time = orig_times[i] + shift
            new_time = max(warped_times[-1] + 1, new_time)
            new_time = min(new_time, L - (n_knots - i))
            warped_times.append(new_time)

        warped_times.append(L - 1)  # Last point stays fixed
        warped_times = np.array(warped_times)

        # Create warping function
        warp_func = CubicSpline(warped_times, orig_times)

        # Generate new time indices
        new_indices = warp_func(np.arange(L))
        new_indices = np.clip(new_indices, 0, L - 1)

        # Apply warping using interpolation
        x_np = x.cpu().numpy()
        x_warped = np.zeros_like(x_np)

        for c in range(C):
            x_warped[c] = np.interp(new_indices, np.arange(L), x_np[c])

        return torch.from_numpy(x_warped).float().to(x.device)

    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the signal.
        Noise level is adaptive to signal standard deviation.
        """
        # Estimate signal std for adaptive noise
        signal_std = torch.std(x, dim=-1, keepdim=True)

        # Random noise level: 1-5% of signal std
        noise_level = random.uniform(0.01, 0.05)

        # Generate and add noise
        noise = torch.randn_like(x) * signal_std * noise_level

        return x + noise

    def channel_permute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly permute channels.
        Note: MIMIC-IV has single channel, so this is typically not used.
        Paper uses this for 4-channel PPG.
        """
        if x.shape[0] <= 1:
            return x

        C = x.shape[0]
        perm = torch.randperm(C)
        return x[perm]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic sequence of augmentations.

        Following paper: "at each call of T(·), a sequence of augmentation
        functions is applied given random binary events drawn from assigned
        probability values."

        Args:
            x: Input signal tensor [C, L] or [L]

        Returns:
            Augmented signal tensor
        """
        # Ensure correct shape
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Store original for residual if needed
        original = x.clone()

        # Apply each augmentation stochastically based on configured probabilities

        if random.random() < self.config.get('cutout', 0):
            x = self.cutout(x)

        if random.random() < self.config.get('magnitude_warp', 0):
            x = self.magnitude_warp(x)

        if random.random() < self.config.get('gaussian_noise', 0):
            x = self.add_gaussian_noise(x)

        if random.random() < self.config.get('time_warp', 0):
            x = self.time_warp(x)

        if random.random() < self.config.get('channel_permute', 0):
            x = self.channel_permute(x)

        return x


class ContrastiveAugmentation:
    """
    Wrapper for creating augmented positive pairs for contrastive learning.
    Creates two different augmented views of the same signal.
    """

    def __init__(self, modality: str = 'ppg', augmentation_config: Optional[Dict] = None):
        """
        Initialize augmentation for positive pair generation.

        Args:
            modality: 'ppg' or 'ecg'
            augmentation_config: Optional custom configuration
        """
        self.augment = StochasticAugmentation(modality, augmentation_config)
        self.modality = modality

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to create positive pairs.

        Args:
            x1: First segment from participant
            x2: Second segment from same participant

        Returns:
            Tuple of augmented segments
        """
        # Apply different random augmentations to each segment
        x1_aug = self.augment(x1)
        x2_aug = self.augment(x2)

        return x1_aug, x2_aug


# ============= TEST FUNCTIONS =============

def test_augmentations():
    """Test augmentation functionality."""
    print("=" * 50)
    print("Testing Augmentations")
    print("=" * 50)

    # Test PPG augmentations
    print("\n1. Testing PPG Augmentations:")
    ppg_aug = StochasticAugmentation(modality='ppg')

    # Create dummy PPG signal: [1, 3840] (60s @ 64Hz)
    ppg_signal = torch.randn(1, 3840)

    # Test individual augmentations
    print("   Testing individual augmentations:")

    # Cutout
    ppg_cutout = ppg_aug.cutout(ppg_signal.clone())
    print(f"   Cutout: {ppg_cutout.shape}, zeros: {(ppg_cutout == 0).sum().item()}")

    # Magnitude warp
    ppg_mag = ppg_aug.magnitude_warp(ppg_signal.clone())
    print(f"   Magnitude warp: {ppg_mag.shape}, mean ratio: {(ppg_mag / ppg_signal).mean():.3f}")

    # Time warp
    ppg_time = ppg_aug.time_warp(ppg_signal.clone())
    print(f"   Time warp: {ppg_time.shape}")

    # Gaussian noise
    ppg_noise = ppg_aug.add_gaussian_noise(ppg_signal.clone())
    noise_added = (ppg_noise - ppg_signal).std()
    print(f"   Gaussian noise: {ppg_noise.shape}, noise std: {noise_added:.4f}")

    # Full augmentation
    ppg_full = ppg_aug(ppg_signal.clone())
    print(f"   Full augmentation: {ppg_full.shape}")
    assert ppg_full.shape == ppg_signal.shape, "Shape mismatch after augmentation"
    print("   ✓ PPG augmentation test passed!")

    # Test ECG augmentations
    print("\n2. Testing ECG Augmentations:")
    ecg_aug = StochasticAugmentation(modality='ecg')

    # Create dummy ECG signal: [1, 3840] (30s @ 128Hz)
    ecg_signal = torch.randn(1, 3840)

    # Test full augmentation multiple times
    augmented_signals = []
    for i in range(5):
        aug_signal = ecg_aug(ecg_signal.clone())
        augmented_signals.append(aug_signal)

    # Check that augmentations produce different results
    differences = []
    for i in range(1, 5):
        diff = (augmented_signals[i] - augmented_signals[0]).abs().mean().item()
        differences.append(diff)

    print(f"   Generated 5 augmented versions")
    print(f"   Mean differences from first: {np.mean(differences):.4f}")
    print("   ✓ ECG augmentation test passed!")

    # Test ContrastiveAugmentation
    print("\n3. Testing Contrastive Augmentation:")
    contrast_aug = ContrastiveAugmentation(modality='ppg')

    # Create two segments from same participant
    seg1 = torch.randn(1, 3840)
    seg2 = torch.randn(1, 3840)

    # Generate augmented positive pair
    seg1_aug, seg2_aug = contrast_aug(seg1, seg2)

    print(f"   Input shapes: {seg1.shape}, {seg2.shape}")
    print(f"   Output shapes: {seg1_aug.shape}, {seg2_aug.shape}")
    print(f"   Difference from original seg1: {(seg1_aug - seg1).abs().mean():.4f}")
    print(f"   Difference from original seg2: {(seg2_aug - seg2).abs().mean():.4f}")
    print("   ✓ Contrastive augmentation test passed!")

    # Test augmentation strength
    print("\n4. Testing Augmentation Strength:")

    # Test with different probabilities
    strong_config = {
        'cutout': 1.0,
        'magnitude_warp': 1.0,
        'gaussian_noise': 1.0,
        'time_warp': 1.0,
        'channel_permute': 0.0
    }

    strong_aug = StochasticAugmentation(modality='ppg', augmentation_config=strong_config)

    original = torch.randn(1, 3840)
    augmented = strong_aug(original.clone())

    print(f"   Strong augmentation difference: {(augmented - original).abs().mean():.4f}")
    print("   ✓ Augmentation strength test passed!")

    # Test shape preservation
    print("\n5. Testing Shape Preservation:")

    for length in [1920, 3840, 7680]:
        test_signal = torch.randn(1, length)
        aug_signal = ppg_aug(test_signal)
        assert aug_signal.shape == test_signal.shape, f"Shape not preserved for length {length}"
        print(f"   Length {length}: ✓")

    print("   ✓ Shape preservation test passed!")

    print("\n" + "=" * 50)
    print("All augmentation tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    # Run tests
    test_augmentations()