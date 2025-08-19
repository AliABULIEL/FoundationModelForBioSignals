#!/usr/bin/env python3
"""
Test script for SSL components.
Run from project root: python3 test_ssl.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from biosignal.ssl import RegularizedInfoNCE, SSLModel, create_ssl_model


def test_ssl_components():
    """Test SSL loss functionality."""
    print("=" * 50)
    print("Testing SSL Components")
    print("=" * 50)

    # Test RegularizedInfoNCE
    print("\n1. Testing RegularizedInfoNCE:")
    batch_size = 16  # Use larger batch size for stability
    dim = 128

    # Create dummy embeddings
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    z1_m = torch.randn(batch_size, dim)
    z2_m = torch.randn(batch_size, dim)

    # Create loss function
    criterion = RegularizedInfoNCE(temperature=0.04, lambda_koleo=0.1)

    # Calculate loss
    loss, loss_dict = criterion(z1, z2, z1_m, z2_m)

    print(f"   Total loss: {loss:.4f}")
    print(f"   Contrastive loss: {loss_dict['loss_contrastive']:.4f}")
    print(f"   KoLeo loss: {loss_dict['loss_koleo']:.4f}")
    assert loss > 0, "Loss should be positive"
    print("   ✓ RegularizedInfoNCE test passed!")

    # Test SSLModel for both modalities
    print("\n2. Testing SSLModel for both modalities:")

    for modality in ['ppg', 'ecg']:
        print(f"\n   Testing {modality.upper()}:")
        model = create_ssl_model(modality=modality)

        # Create dummy input
        if modality == 'ppg':
            # PPG: 60s @ 64Hz = 3840 samples
            x1 = torch.randn(8, 1, 3840)
            x2 = torch.randn(8, 1, 3840)
        else:
            # ECG: 30s @ 128Hz = 3840 samples
            x1 = torch.randn(8, 1, 3840)
            x2 = torch.randn(8, 1, 3840)

        # Forward pass
        loss, metrics = model(x1, x2)

        print(f"     Loss: {loss:.4f}")
        print(f"     Contrastive: {metrics['loss_contrastive']:.4f}")
        print(f"     KoLeo: {metrics['loss_koleo']:.4f}")

        # Test with embeddings return
        loss, metrics, (h1, h2) = model(x1, x2, return_embeddings=True)
        print(f"     Embedding shape: {h1.shape} (expected: [8, 256])")
        assert h1.shape == (8, 256), f"Wrong embedding shape: {h1.shape}"

    print("\n   ✓ SSLModel tests passed!")

    # Test momentum update
    print("\n3. Testing momentum network update:")

    model = create_ssl_model(modality='ecg')

    # Get initial momentum weight
    initial_weight = model.momentum_encoder.stem[0].weight.data.clone()

    # Modify online encoder weight
    model.encoder.stem[0].weight.data += 0.1

    # Update momentum network
    model.update_momentum_networks()

    # Check momentum weight changed
    updated_weight = model.momentum_encoder.stem[0].weight.data
    weight_diff = (updated_weight - initial_weight).abs().mean().item()

    print(f"   Weight difference: {weight_diff:.6f}")
    print(f"   Expected: ~{0.1 * (1 - 0.99):.6f} (1% of change)")

    assert weight_diff > 0, "Momentum weights should change"
    assert weight_diff < 0.01, "Change should be small due to momentum"
    print("   ✓ Momentum update test passed!")

    # Test batch size recommendations
    print("\n4. Testing batch size effects:")
    print("   Paper recommends batch_size = 256")

    model = create_ssl_model(modality='ppg')
    batch_sizes = [4, 8, 16, 32, 64]

    for bs in batch_sizes:
        x1 = torch.randn(bs, 1, 3840)
        x2 = torch.randn(bs, 1, 3840)

        loss, metrics = model(x1, x2)
        stability = "stable" if loss > 0 else "unstable"

        print(f"   Batch size {bs:3d}: loss = {loss:7.4f} ({stability})")

    print("\n   Note: Larger batch sizes provide more stable training")
    print("   ✓ Batch size test passed!")

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    success = test_ssl_components()
    sys.exit(0 if success else 1)