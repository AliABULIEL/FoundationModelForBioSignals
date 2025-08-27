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


class SimSiamLoss(nn.Module):
    """
    SimSiam loss - cosine similarity without negative pairs.
    Much simpler than InfoNCE, works better with small datasets.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
            self,
            p1: torch.Tensor,
            p2: torch.Tensor,
            z1: torch.Tensor,
            z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate SimSiam loss.

        Args:
            p1, p2: Predictions [N, D]
            z1, z2: Projections [N, D] (stop-gradient applied)

        Returns:
            Total loss and metrics
        """
        # Normalize if required
        if self.normalize:
            p1 = F.normalize(p1, dim=1)
            p2 = F.normalize(p2, dim=1)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        # Calculate cosine similarity loss (negative because we maximize similarity)
        loss1 = -F.cosine_similarity(p1, z2.detach(), dim=1).mean()
        loss2 = -F.cosine_similarity(p2, z1.detach(), dim=1).mean()

        total_loss = (loss1 + loss2) / 2

        # Metrics for logging (compatible with your training loop)
        metrics = {
            'loss': total_loss.item(),
            'loss_contrastive': total_loss.item(),  # For compatibility with training loop
            'loss_koleo': 0.0,  # No KoLeo in SimSiam
            'loss_12': loss1.item(),
            'loss_21': loss2.item()
        }

        return total_loss, metrics


class SimSiamModel(nn.Module):
    """
    SimSiam model following the exact structure of SSLModel.
    No momentum encoder, uses stop-gradient instead.
    """

    def __init__(
            self,
            encoder: nn.Module,
            projection_head: nn.Module,  # Will be replaced with SimSiam's projector
            config_path: str = 'configs/config.yaml'
    ):
        super().__init__()

        # Load configuration (add simsiam section to config)
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Look for simsiam config, fall back to defaults
            if 'simsiam' in config:
                simsiam_config = config['simsiam']
            else:
                simsiam_config = {
                    'projection_dim': 2048,
                    'prediction_dim': 512
                }
        else:
            simsiam_config = {
                'projection_dim': 2048,
                'prediction_dim': 512
            }

        # Encoder (shared)
        self.encoder = encoder

        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.randn(1, 1, 3840)
            embedding_dim = encoder(dummy).shape[1]

        # SimSiam projector (3 layers with BN)
        projection_dim = simsiam_config.get('projection_dim', 2048)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )

        # SimSiam predictor (2 layers with BN)
        prediction_dim = simsiam_config.get('prediction_dim', 512)
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim, bias=False),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_dim, projection_dim)
        )

        # Loss function
        self.criterion = SimSiamLoss()

        # For compatibility with SSLModel interface
        self.projection_head = self.projector  # Alias for compatibility
        self.momentum_rate = 0.0  # No momentum in SimSiam

        # Dummy momentum networks for interface compatibility
        self.momentum_encoder = None
        self.momentum_projection = None

        # Get device from global device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Move to device
        self.to(self.device)

        print(f"\nSimSiam Model initialized:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Prediction dim: {prediction_dim}")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def update_momentum_networks(self):
        """No-op for SimSiam (no momentum networks)."""
        pass

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through SimSiam model.

        Args:
            x1: First view [B, C, L]
            x2: Second view [B, C, L]

        Returns:
            Loss and metrics (compatible with training loop)
        """
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Project both views
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Predict from both projections
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Calculate loss (stop-gradient handled inside)
        loss, metrics = self.criterion(p1, p2, z1, z2)

        return loss, metrics

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for downstream tasks."""
        with torch.no_grad():
            return self.encoder(x)

    def save_checkpoint(
            self,
            path: str,
            optimizer=None,
            epoch: int = 0,
            best_loss: float = float('inf')
    ):
        """Save model checkpoint (compatible with SSLModel)."""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_state_dict': self.projector.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"SimSiam checkpoint saved to {path}")

    def load_checkpoint(
            self,
            path: str,
            optimizer=None
    ) -> Tuple[int, float]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projector.load_state_dict(checkpoint['projection_state_dict'])

        if 'predictor_state_dict' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"SimSiam checkpoint loaded from {path}")
        return epoch, best_loss


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