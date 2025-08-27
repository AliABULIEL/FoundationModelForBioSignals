# simsiam_model.py
"""
SimSiam implementation for small-scale SSL training.
Follows the exact structure of ssl_model.py for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path
from copy import deepcopy
from device import get_device_manager




def create_simsiam_model(
        encoder: nn.Module,
        projection_head: nn.Module,  # Ignored, but kept for interface compatibility
        config_path: str = 'configs/config.yaml'
) -> SimSiamModel:
    """Create SimSiam model with configuration."""
    return SimSiamModel(encoder, projection_head, config_path)


# ============= TEST FUNCTION =============

def test_simsiam():
    """Test SimSiam functionality."""
    print("=" * 50)
    print("Testing SimSiam Model")
    print("=" * 50)

    # Get device from global device manager
    device_manager = get_device_manager()
    device = device_manager.device

    print(f"✓ Using device: {device} ({device_manager.type})")

    # Test SimSiam loss
    print("\n1. Testing SimSiam Loss:")
    criterion = SimSiamLoss()

    batch_size = 16
    dim = 2048
    p1 = torch.randn(batch_size, dim).to(device)
    p2 = torch.randn(batch_size, dim).to(device)
    z1 = torch.randn(batch_size, dim).to(device)
    z2 = torch.randn(batch_size, dim).to(device)

    loss, metrics = criterion(p1, p2, z1, z2)

    print(f"   Total loss: {loss:.4f}")
    print(f"   Loss 1->2: {metrics['loss_12']:.4f}")
    print(f"   Loss 2->1: {metrics['loss_21']:.4f}")
    print("   ✓ SimSiam loss test passed!")

    # Test SimSiam Model
    print("\n2. Testing SimSiam Model:")

    from model import EfficientNet1D, ProjectionHead

    encoder = EfficientNet1D(in_channels=1, embedding_dim=256)
    projection_head = ProjectionHead(input_dim=256, output_dim=128)  # Will be replaced

    simsiam_model = create_simsiam_model(encoder, projection_head)

    # Test forward pass
    x1 = torch.randn(4, 1, 3840).to(device)
    x2 = torch.randn(4, 1, 3840).to(device)

    loss, metrics = simsiam_model(x1, x2)

    print(f"   SimSiam model loss: {loss:.4f}")
    print(f"   Metrics: {list(metrics.keys())}")
    assert loss < 0, "SimSiam loss should be negative (cosine similarity)"
    print("   ✓ SimSiam model test passed!")

    # Test with different batch sizes
    print("\n3. Testing different batch sizes:")
    for batch_size in [2, 4, 8, 16]:
        x1 = torch.randn(batch_size, 1, 3840).to(device)
        x2 = torch.randn(batch_size, 1, 3840).to(device)

        loss, _ = simsiam_model(x1, x2)
        print(f"   Batch size {batch_size}: loss = {loss:.4f}")

    print("   ✓ Batch size test passed!")
    print("   Note: SimSiam works well even with small batches!")

    # Clean up
    if device_manager.is_cuda:
        device_manager.empty_cache()

    print("\n" + "=" * 50)
    print("All SimSiam tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_simsiam()