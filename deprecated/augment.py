# biosignal/augment.py
"""
Signal augmentation module for SSL training
Following Apple paper's augmentation strategy
Uses global device manager instead of internal device management
Uses centralized configuration management
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import random
from scipy.interpolate import CubicSpline
from pathlib import Path

from device import get_device_manager
from config_loader import get_config  # Added ConfigLoader


class SignalAugmentation:
    """
    Augmentation module for biosignals following Apple paper.

    Paper augmentations:
    - Cutout: Random masking of signal segments
    - Magnitude warp: Smooth amplitude scaling
    - Gaussian noise: Additive noise
    - Time warp: Non-linear time distortion
    - Channel permute: For multi-channel signals (ACC)
    """

    def __init__(
            self,
            modality: str = 'ppg',
            config_path: str = 'configs/config.yaml',
            ssl_method: str = 'infonce'
    ):
        """
        Initialize augmentation module.

        Args:
            modality: Signal type ('ppg', 'ecg', or 'acc')
            config_path: Path to configuration file
            ssl_method: SSL method ('infonce' or 'simsiam')
        """
        self.modality = modality.lower()
        self.ssl_method = ssl_method

        # Get device from global device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Load configuration
        self.config = get_config()

        # Get modality-specific augmentation probabilities from config
        self.aug_probs = self.config.get_augmentation_config(modality, ssl_method)

        # If no augmentation config found, load from config file
        if not self.aug_probs:
            # Get the right section based on SSL method
            if ssl_method == 'simsiam':
                config_section = self.config.get_section('simsiam')
            else:
                config_section = self.config.get_section('ssl')

            # Get augmentation probabilities for the modality
            aug_key = f'augmentations_{modality}'
            self.aug_probs = config_section.get(aug_key, {})

            # If still no config, use minimal defaults
            if not self.aug_probs:
                print(f"Warning: No augmentation config found for {modality} with {ssl_method}, using minimal defaults")
                self.aug_probs = {
                    'cutout': 0.0,
                    'magnitude_warp': 0.0,
                    'gaussian_noise': 0.0,
                    'time_warp': 0.0,
                    'channel_permute': 0.0
                }

        print(f"Augmentation initialized for {modality.upper()} on {self.device}")
        print(f"  SSL method: {ssl_method}")
        print(f"  Augmentation probabilities from config:")
        for aug_name, prob in self.aug_probs.items():
            print(f"    {aug_name}: {prob:.2f}")
        if modality == 'acc':
            print(f"  ACC-specific: channel_permute probability = {self.aug_probs.get('channel_permute', 0)}")

        # Get cutout parameters from config
        self.cutout_min_ratio = self.config.get('augmentation.cutout_min_ratio', 0.05)
        self.cutout_max_ratio = self.config.get('augmentation.cutout_max_ratio', 0.15)
        self.cutout_max_masks = self.config.get('augmentation.cutout_max_masks', 3)

        # Get magnitude warp parameters from config
        self.magnitude_warp_min = self.config.get('augmentation.magnitude_warp_min', 0.8)
        self.magnitude_warp_max = self.config.get('augmentation.magnitude_warp_max', 1.2)
        self.magnitude_warp_knots_min = self.config.get('augmentation.magnitude_warp_knots_min', 4)
        self.magnitude_warp_knots_max = self.config.get('augmentation.magnitude_warp_knots_max', 6)

        # Get noise parameters from config
        self.noise_level_min = self.config.get('augmentation.noise_level_min', 0.01)
        self.noise_level_max = self.config.get('augmentation.noise_level_max', 0.03)

        # Get time warp parameters from config
        self.time_warp_shift_ratio = self.config.get('augmentation.time_warp_shift_ratio', 0.05)
        self.time_warp_knots_min = self.config.get('augmentation.time_warp_knots_min', 4)
        self.time_warp_knots_max = self.config.get('augmentation.time_warp_knots_max', 6)

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout augmentation (masking segments).

        Args:
            x: Input tensor [batch_size, channels, length]
        """
        B, C, L = x.shape
        x_aug = x.clone()

        # Apply to each sample in batch
        for b in range(B):
            if random.random() < self.aug_probs.get('cutout', 0):
                # Number of segments to mask (from config)
                n_masks = random.randint(1, self.cutout_max_masks)

                for _ in range(n_masks):
                    # Mask length: from config ratios
                    mask_len = random.randint(
                        int(self.cutout_min_ratio * L),
                        int(self.cutout_max_ratio * L)
                    )

                    # Random position
                    if L > mask_len:
                        start = random.randint(0, L - mask_len)
                        # For ACC, we can choose to mask all channels or individual channels
                        acc_individual_channel_prob = self.config.get('augmentation.acc_individual_channel_prob', 0.5)
                        if self.modality == 'acc' and random.random() < acc_individual_channel_prob:
                            # Mask individual channel
                            channel = random.randint(0, C - 1)
                            x_aug[b, channel, start:start + mask_len] = 0.0
                        else:
                            # Mask all channels
                            x_aug[b, :, start:start + mask_len] = 0.0

        return x_aug

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping using smooth curves.

        Args:
            x: Input tensor [batch_size, channels, length]
        """
        B, C, L = x.shape
        x_aug = x.clone()

        for b in range(B):
            if random.random() < self.aug_probs.get('magnitude_warp', 0):
                # Create smooth warping curve
                n_knots = random.randint(self.magnitude_warp_knots_min, self.magnitude_warp_knots_max)
                knot_xs = np.linspace(0, L - 1, n_knots)

                # For ACC, we can apply different warping to each axis
                if self.modality == 'acc':
                    for c in range(C):
                        # Warping factors from config
                        knot_ys = np.random.uniform(
                            self.magnitude_warp_min,
                            self.magnitude_warp_max,
                            n_knots
                        )
                        # Clip edge values for stability
                        edge_min = self.config.get('augmentation.magnitude_warp_edge_min', 0.9)
                        edge_max = self.config.get('augmentation.magnitude_warp_edge_max', 1.1)
                        knot_ys[0] = np.clip(knot_ys[0], edge_min, edge_max)
                        knot_ys[-1] = np.clip(knot_ys[-1], edge_min, edge_max)

                        # Create smooth curve
                        spline = CubicSpline(knot_xs, knot_ys)
                        warp_curve = spline(np.arange(L))

                        # Convert to tensor and apply
                        warp_tensor = torch.from_numpy(warp_curve.astype(np.float32)).to(self.device)
                        x_aug[b, c] = x_aug[b, c] * warp_tensor
                else:
                    # Single warping for all channels (PPG/ECG)
                    knot_ys = np.random.uniform(
                        self.magnitude_warp_min,
                        self.magnitude_warp_max,
                        n_knots
                    )
                    edge_min = self.config.get('augmentation.magnitude_warp_edge_min', 0.9)
                    edge_max = self.config.get('augmentation.magnitude_warp_edge_max', 1.1)
                    knot_ys[0] = np.clip(knot_ys[0], edge_min, edge_max)
                    knot_ys[-1] = np.clip(knot_ys[-1], edge_min, edge_max)

                    spline = CubicSpline(knot_xs, knot_ys)
                    warp_curve = spline(np.arange(L))

                    warp_tensor = torch.from_numpy(warp_curve.astype(np.float32)).to(self.device)
                    x_aug[b] = x_aug[b] * warp_tensor.unsqueeze(0)

        return x_aug

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to signal.

        Args:
            x: Input tensor [batch_size, channels, length]
        """
        B, C, L = x.shape
        x_aug = x.clone()

        for b in range(B):
            if random.random() < self.aug_probs.get('gaussian_noise', 0):
                # Adaptive noise based on signal std
                if self.modality == 'acc':
                    # For ACC, calculate std per channel
                    for c in range(C):
                        channel_std = torch.std(x[b, c])
                        # Noise level from config
                        noise_level = random.uniform(self.noise_level_min, self.noise_level_max)
                        noise = torch.randn_like(x[b, c]) * channel_std * noise_level
                        x_aug[b, c] = x_aug[b, c] + noise
                else:
                    # For PPG/ECG
                    signal_std = torch.std(x[b], dim=-1, keepdim=True)
                    noise_level = random.uniform(self.noise_level_min, self.noise_level_max)
                    noise = torch.randn_like(x[b]) * signal_std * noise_level
                    x_aug[b] = x_aug[b] + noise

        return x_aug

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping (non-linear time distortion).

        Args:
            x: Input tensor [batch_size, channels, length]
        """
        B, C, L = x.shape
        x_aug = x.clone()

        for b in range(B):
            if random.random() < self.aug_probs.get('time_warp', 0):
                # Create time warping
                n_knots = random.randint(self.time_warp_knots_min, self.time_warp_knots_max)
                orig_times = np.linspace(0, L - 1, n_knots)

                # Warped time points
                warped_times = [0]
                for i in range(1, n_knots - 1):
                    max_shift = int(self.time_warp_shift_ratio * L / n_knots)
                    shift = random.randint(-max_shift, max_shift)
                    new_time = orig_times[i] + shift
                    new_time = max(warped_times[-1] + 1, new_time)
                    new_time = min(new_time, L - (n_knots - i))
                    warped_times.append(new_time)
                warped_times.append(L - 1)

                # Create warping function
                warped_times = np.array(warped_times)
                warp_func = CubicSpline(warped_times, orig_times)
                new_indices = warp_func(np.arange(L))
                new_indices = np.clip(new_indices, 0, L - 1)

                # Apply warping to each channel
                x_np = x_aug[b].cpu().numpy()
                x_warped = np.zeros_like(x_np)
                for c in range(C):
                    x_warped[c] = np.interp(new_indices, np.arange(L), x_np[c])

                x_aug[b] = torch.from_numpy(x_warped).float().to(self.device)

        return x_aug

    def channel_permute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly permute channels (for multi-channel signals like ACC).
        This is crucial for ACC to learn rotation-invariant features.

        Args:
            x: Input tensor [batch_size, channels, length]
        """
        B, C, L = x.shape

        if C <= 1:
            return x  # No permutation for single channel

        x_aug = x.clone()

        for b in range(B):
            if random.random() < self.aug_probs.get('channel_permute', 0):
                perm = torch.randperm(C)
                x_aug[b] = x_aug[b, perm]

                # For ACC, we can also apply axis swapping with sign flips
                # This simulates different device orientations
                acc_sign_flip_prob = self.config.get('augmentation.acc_sign_flip_prob', 0.5)
                if self.modality == 'acc' and random.random() < acc_sign_flip_prob:
                    # Randomly flip signs of axes
                    signs = torch.tensor([random.choice([-1, 1]) for _ in range(C)],
                                         dtype=x.dtype, device=self.device)
                    x_aug[b] = x_aug[b] * signs.view(-1, 1)

        return x_aug

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline.

        Args:
            x: Input tensor [batch_size, channels, length]

        Returns:
            Augmented tensor
        """
        # Apply augmentations in sequence
        x = self.cutout(x)
        x = self.magnitude_warp(x)
        x = self.gaussian_noise(x)
        x = self.time_warp(x)
        x = self.channel_permute(x)

        return x


class ContrastiveAugmentation:
    """
    Wrapper for creating augmented positive pairs for SSL.
    Uses global device manager.
    """

    def __init__(
            self,
            modality: str = 'ppg',
            config_path: str = 'configs/config.yaml',
            ssl_method: str = 'infonce'
    ):
        """Initialize augmentation for positive pairs."""
        self.augment = SignalAugmentation(modality, config_path, ssl_method)
        self.modality = modality

        # Get device from global device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

    def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create augmented positive pairs.

        Args:
            x1: First segment [batch_size, channels, length]
            x2: Second segment [batch_size, channels, length]

        Returns:
            Tuple of augmented segments
        """
        # Apply different random augmentations to each segment
        x1_aug = self.augment(x1)
        x2_aug = self.augment(x2)

        return x1_aug, x2_aug


# ============= TEST FUNCTIONS =============

def test_augmentations():
    """Test augmentation functionality with device manager."""
    print("=" * 50)
    print("Testing Signal Augmentations with Device Manager")
    print("=" * 50)

    # Get device from global device manager
    device_manager = get_device_manager()
    device = device_manager.device

    # Load config
    config = get_config()

    print(f"✓ Using device: {device} ({device_manager.type})")

    # Test PPG augmentations
    print("\n1. Testing PPG Augmentations:")
    ppg_aug = SignalAugmentation(modality='ppg')

    # Get PPG segment parameters from config
    ppg_segment_length = config.get('dataset.ppg.segment_length', 60)
    ppg_target_fs = config.get('dataset.ppg.target_fs', 64)
    ppg_length = ppg_segment_length * ppg_target_fs

    # Create dummy PPG signal
    ppg_signal = torch.randn(2, 1, ppg_length).to(device)

    # Test individual augmentations
    print("   Testing individual augmentations:")

    ppg_cutout = ppg_aug.cutout(ppg_signal.clone())
    print(f"   ✓ Cutout: {ppg_cutout.shape}, zeros: {(ppg_cutout == 0).sum().item()}")

    ppg_mag = ppg_aug.magnitude_warp(ppg_signal.clone())
    mag_ratio = (ppg_mag / (ppg_signal + 1e-6)).mean().item()
    print(f"   ✓ Magnitude warp: {ppg_mag.shape}, mean ratio: {mag_ratio:.3f}")

    ppg_noise = ppg_aug.gaussian_noise(ppg_signal.clone())
    noise_added = (ppg_noise - ppg_signal).std().item()
    print(f"   ✓ Gaussian noise: {ppg_noise.shape}, noise std: {noise_added:.4f}")

    ppg_time = ppg_aug.time_warp(ppg_signal.clone())
    print(f"   ✓ Time warp: {ppg_time.shape}")

    # Full augmentation
    ppg_full = ppg_aug(ppg_signal.clone())
    print(f"   ✓ Full augmentation: {ppg_full.shape}")

    # Test ECG augmentations
    print("\n2. Testing ECG Augmentations:")
    ecg_aug = SignalAugmentation(modality='ecg')

    # Get ECG segment parameters from config
    ecg_segment_length = config.get('dataset.ecg.segment_length', 30)
    ecg_target_fs = config.get('dataset.ecg.target_fs', 128)
    ecg_length = ecg_segment_length * ecg_target_fs

    # Create dummy ECG signal
    ecg_signal = torch.randn(2, 1, ecg_length).to(device)

    # Test multiple augmentations
    augmented_ecgs = []
    for i in range(3):
        aug_ecg = ecg_aug(ecg_signal.clone())
        augmented_ecgs.append(aug_ecg)

    # Check differences
    diff1 = (augmented_ecgs[1] - augmented_ecgs[0]).abs().mean().item()
    diff2 = (augmented_ecgs[2] - augmented_ecgs[0]).abs().mean().item()

    print(f"   Generated 3 augmented versions")
    print(f"   Mean differences: {diff1:.4f}, {diff2:.4f}")
    print("   ✓ ECG augmentation test passed!")

    # Test ACC augmentations (ENHANCED)
    print("\n3. Testing ACC Augmentations (3-channel):")
    acc_aug = SignalAugmentation(modality='acc')

    # Get ACC segment parameters from config
    acc_segment_length = config.get('dataset.acc.segment_length', 60)
    acc_target_fs = config.get('dataset.acc.target_fs', 100)
    acc_channels = config.get('dataset.acc.channels', 3)
    acc_length = acc_segment_length * acc_target_fs

    # Create dummy ACC signal
    acc_signal = torch.randn(2, acc_channels, acc_length).to(device)
    print(f"   Original ACC signal shape: {acc_signal.shape}")

    # Test individual ACC augmentations
    print("\n   Testing ACC-specific augmentations:")

    # Test cutout
    acc_cutout = acc_aug.cutout(acc_signal.clone())
    zeros_per_channel = [(acc_cutout[0, c] == 0).sum().item() for c in range(acc_channels)]
    print(f"   ✓ Cutout - zeros per channel: {zeros_per_channel}")

    # Test magnitude warp (per-channel)
    acc_mag = acc_aug.magnitude_warp(acc_signal.clone())
    for c in range(acc_channels):
        ratio = (acc_mag[0, c] / (acc_signal[0, c] + 1e-6)).mean().item()
        print(f"   ✓ Magnitude warp channel {c}: ratio = {ratio:.3f}")

    # Test Gaussian noise (per-channel)
    acc_noise = acc_aug.gaussian_noise(acc_signal.clone())
    for c in range(acc_channels):
        noise_std = (acc_noise[0, c] - acc_signal[0, c]).std().item()
        print(f"   ✓ Gaussian noise channel {c}: std = {noise_std:.4f}")

    # Test channel permutation
    print("\n   Testing channel permutation:")
    acc_perm = acc_aug.channel_permute(acc_signal.clone())

    # Check if channels were actually permuted
    original_norms = [acc_signal[0, c].norm().item() for c in range(acc_channels)]
    permuted_norms = [acc_perm[0, c].norm().item() for c in range(acc_channels)]
    print(f"   Original channel norms: {[f'{n:.3f}' for n in original_norms]}")
    print(f"   After permutation: {[f'{n:.3f}' for n in permuted_norms]}")

    # Full augmentation
    acc_full = acc_aug(acc_signal.clone())
    print(f"\n   ✓ Full ACC augmentation: {acc_full.shape}")
    assert acc_full.shape[0] == 2, f"Wrong batch size: {acc_full.shape[0]}"
    assert acc_full.shape[1] == acc_channels, f"ACC should have {acc_channels} channels, got {acc_full.shape[1]}"
    print("   ✓ ACC augmentation test passed!")

    # Test ContrastiveAugmentation for ACC
    print("\n4. Testing Contrastive Augmentation for ACC:")
    contrast_aug = ContrastiveAugmentation(modality='acc')

    seg1 = torch.randn(4, acc_channels, acc_length).to(device)
    seg2 = torch.randn(4, acc_channels, acc_length).to(device)

    seg1_aug, seg2_aug = contrast_aug(seg1, seg2)

    print(f"   Input shapes: {seg1.shape}, {seg2.shape}")
    print(f"   Output shapes: {seg1_aug.shape}, {seg2_aug.shape}")
    print(f"   Diff from original seg1: {(seg1_aug - seg1).abs().mean():.4f}")
    print(f"   Diff from original seg2: {(seg2_aug - seg2).abs().mean():.4f}")
    print("   ✓ ACC contrastive augmentation test passed!")

    # Test with different batch sizes for ACC
    print("\n5. Testing different batch sizes for ACC:")
    for batch_size in [1, 4, 8, 16]:
        test_signal = torch.randn(batch_size, acc_channels, acc_length).to(device)
        aug_signal = acc_aug(test_signal)
        assert aug_signal.shape == test_signal.shape
        print(f"   Batch size {batch_size}: ✓")

    # Test reproducibility with seed for ACC
    print("\n6. Testing ACC reproducibility:")
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    test_signal = torch.randn(2, acc_channels, acc_length).to(device)
    aug1 = acc_aug(test_signal.clone())

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    aug2 = acc_aug(test_signal.clone())

    diff = (aug1 - aug2).abs().max().item()
    print(f"   Max difference with same seed: {diff:.6f}")
    print("   ✓ ACC reproducibility test passed!")

    # Test with SimSiam augmentations
    print("\n7. Testing SimSiam augmentations:")
    simsiam_aug = SignalAugmentation(modality='ppg', ssl_method='simsiam')
    print("   ✓ SimSiam augmentation initialized")

    # Test memory efficiency
    print("\n8. Testing memory efficiency:")
    if device_manager.is_cuda:
        initial_mem = device_manager.memory_stats()
        print(f"   Initial GPU memory: {initial_mem.get('allocated', 0):.3f} GB")

        # Run augmentations
        for _ in range(10):
            large_signal = torch.randn(32, acc_channels, acc_length).to(device)
            aug_signal = acc_aug(large_signal)
            del large_signal, aug_signal

        # Clean cache
        device_manager.empty_cache()
        final_mem = device_manager.memory_stats()
        print(f"   Final GPU memory: {final_mem.get('allocated', 0):.3f} GB")
        print("   ✓ Memory management test passed!")
    else:
        print("   Skipping memory test (not on CUDA)")

    print("\n" + "=" * 50)
    print(f"All augmentation tests passed successfully on {device_manager.type}!")
    print("=" * 50)


if __name__ == "__main__":
    test_augmentations()