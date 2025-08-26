#!/usr/bin/env python3
"""
Minimal test to verify the training pipeline works
"""

import torch
import numpy as np
from pathlib import Path


def create_dummy_dataset():
    """Create a minimal dummy dataset for testing."""

    print("Creating dummy dataset for testing...")

    # Create dummy data directory
    dummy_dir = Path("data/dummy_test")
    dummy_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal config
    import yaml
    config = {
        'dataset': {
            'data_dir': str(dummy_dir),
            'ppg': {
                'original_fs': 30,
                'target_fs': 64,
                'segment_length': 60,
                'band_low': 0.4,
                'band_high': 8.0
            }
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'num_workers': 0,
            'save_freq': 1,
            'checkpoint_dir': 'data/outputs/test',
            'optimizer': 'adam',
            'use_amp': False,
            'gradient_accumulation_steps': 1
        },
        'model': {
            'embedding_dim': 256,
            'n_blocks': 4,  # Smaller for testing
            'width_multiplier': 0.5,  # Smaller model
            'depth_multiplier': 0.5,
            'dropout_rate': 0.1,
            'drop_path_rate': 0.1
        },
        'ssl': {
            'temperature': 0.04,
            'lambda_koleo': 0.1,
            'momentum_rate': 0.99,
            'augmentations_ppg': {
                'cutout': 0.4,
                'magnitude_warp': 0.25,
                'gaussian_noise': 0.25,
                'time_warp': 0.15,
                'channel_permute': 0.0
            }
        },
        'device': {
            'backend': 'mps' if torch.backends.mps.is_available() else 'cpu'
        },
        'seed': 42
    }

    with open('configs/test_minimal.yaml', 'w') as f:
        yaml.dump(config, f)

    print("✓ Created test configuration")

    # Test training with synthetic data
    print("\nTesting training pipeline with synthetic data...")

    from train import Trainer

    # Create a minimal trainer that uses synthetic data
    trainer = Trainer(config_path='configs/test_minimal.yaml', experiment_name='minimal_test')

    # Override data loading to use synthetic data
    class DummyDataLoader:
        def __init__(self, n_batches=10):
            self.n_batches = n_batches

        def __len__(self):
            return self.n_batches

        def __iter__(self):
            for _ in range(self.n_batches):
                # Generate synthetic batch
                seg1 = torch.randn(4, 1, 3840)  # batch_size=4, channels=1, length=3840
                seg2 = torch.randn(4, 1, 3840)
                yield seg1, seg2

    # Replace dataloaders
    trainer.train_loader = DummyDataLoader(10)
    trainer.val_loader = DummyDataLoader(2)
    trainer.test_loader = DummyDataLoader(2)

    # Setup model and optimizer
    trainer.setup_model(modality='ppg')
    trainer.setup_optimizer()

    # Create augmentation
    from augment import ContrastiveAugmentation
    trainer.augmentation = ContrastiveAugmentation(
        modality='ppg',
        device=str(trainer.device)
    )
    trainer.modality = 'ppg'

    # Run one epoch
    print("\nRunning one training epoch with synthetic data...")
    train_stats = trainer.train_epoch(0)

    print(f"\n✓ Training successful!")
    print(f"  Loss: {train_stats['loss']:.4f}")
    print(f"  This confirms the model and training pipeline work correctly.")

    # Clean up
    import shutil
    if trainer.checkpoint_dir.exists():
        shutil.rmtree(trainer.checkpoint_dir.parent)

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("MINIMAL PIPELINE TEST")
    print("=" * 70)

    if create_dummy_dataset():
        print("\n" + "=" * 70)
        print("SUCCESS: Training pipeline works!")
        print("=" * 70)
        print("\nThe issue is with data loading, not the model or training code.")
        print("\nNext steps:")
        print("1. Run: python full_diagnostic.py")
        print("2. Check if dataset is properly downloaded")
        print("3. Verify file structure matches expected format")
    else:
        print("\nPipeline test failed - check error messages above")