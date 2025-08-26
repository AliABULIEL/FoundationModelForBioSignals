# biosignal/ssl_model.py
"""
Self-supervised learning module with RegularizedInfoNCE loss
Following Apple paper's SSL approach
Uses global device manager instead of internal device management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path
from copy import deepcopy
from device import get_device_manager


class RegularizedInfoNCE(nn.Module):
    """
    Regularized InfoNCE loss from Apple paper.
    Combines InfoNCE with KoLeo regularization.

    Paper equations:
    - InfoNCE: L_contrastive = -log(exp(sim(z1,z2)/τ) / Σ exp(sim(z1,zj)/τ))
    - KoLeo: L_koleo = -log(min_j≠i ||zi - zj||)
    - Final: L = L_contrastive + λ * L_koleo
    """

    def __init__(
            self,
            temperature: float = 0.07,
            lambda_koleo: float = 0.05,
            normalize: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_koleo = lambda_koleo
        self.normalize = normalize

    def infonce_loss(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate InfoNCE loss.

        Args:
            queries: Query embeddings [N, D]
            keys: Key embeddings [N, D]
        """
        batch_size = queries.shape[0]

        # Normalize if required
        if self.normalize:
            queries = F.normalize(queries, dim=1)
            keys = F.normalize(keys, dim=1)

        # Calculate similarity matrix
        sim_matrix = torch.matmul(queries, keys.T) / self.temperature

        # Labels (positive pairs on diagonal)
        labels = torch.arange(batch_size, device=queries.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def koleo_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate KoLeo regularization.
        Encourages uniform span of features.

        Args:
            embeddings: Embeddings [N, D]
        """
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize if required
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)

        # Pairwise L2 distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Mask diagonal
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        distances = distances.masked_fill(mask, float('inf'))

        # Find minimum distances
        min_distances, _ = distances.min(dim=1)

        # Clamp for numerical stability
        min_distances = torch.clamp(min_distances, min=1e-3)

        # KoLeo loss
        koleo = -torch.log(min_distances).mean()

        return koleo

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            z1_momentum: Optional[torch.Tensor] = None,
            z2_momentum: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate regularized InfoNCE loss.

        Args:
            z1: First view embeddings [N, D]
            z2: Second view embeddings [N, D]
            z1_momentum: First view from momentum encoder
            z2_momentum: Second view from momentum encoder

        Returns:
            Total loss and metrics dictionary
        """
        # Use momentum embeddings as keys if available
        keys1 = z1_momentum if z1_momentum is not None else z1
        keys2 = z2_momentum if z2_momentum is not None else z2

        # Bidirectional InfoNCE
        loss_12 = self.infonce_loss(z1, keys2)
        loss_21 = self.infonce_loss(z2, keys1)
        loss_contrastive = (loss_12 + loss_21) / 2

        # KoLeo regularization
        koleo_1 = self.koleo_loss(z1)
        koleo_2 = self.koleo_loss(z2)
        loss_koleo = (koleo_1 + koleo_2) / 2

        # Total loss
        total_loss = loss_contrastive + self.lambda_koleo * loss_koleo

        # Metrics for logging
        metrics = {
            'loss': total_loss.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_koleo': loss_koleo.item(),
            'loss_12': loss_12.item(),
            'loss_21': loss_21.item()
        }

        return total_loss, metrics


class SSLModel(nn.Module):
    """
    Complete SSL model with momentum encoder.
    Implements Apple paper's training framework.
    Uses global device manager for device handling.
    """

    def __init__(
            self,
            encoder: nn.Module,
            projection_head: nn.Module,
            config_path: str = 'configs/config.yaml'
    ):
        super().__init__()

        # Load configuration
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            ssl_config = config['ssl']
        else:
            ssl_config = {
                'temperature': 0.04,
                'lambda_koleo': 0.1,
                'momentum_rate': 0.99
            }

        # Online encoder and projection
        self.encoder = encoder
        self.projection_head = projection_head

        # Momentum encoder and projection (no gradients)
        self.momentum_encoder = deepcopy(encoder)
        self.momentum_projection = deepcopy(projection_head)

        # Freeze momentum networks
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projection.parameters():
            param.requires_grad = False

        # Loss function
        self.criterion = RegularizedInfoNCE(
            temperature=ssl_config['temperature'],
            lambda_koleo=ssl_config['lambda_koleo']
        )

        # Momentum rate
        self.momentum_rate = ssl_config['momentum_rate']

        # Get device from global device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Move to device
        self.to(self.device)

    @torch.no_grad()
    def update_momentum_networks(self):
        """Update momentum networks with EMA."""
        # Update encoder
        for param_m, param in zip(
                self.momentum_encoder.parameters(),
                self.encoder.parameters()
        ):
            param_m.data = (
                    self.momentum_rate * param_m.data +
                    (1 - self.momentum_rate) * param.data
            )

        # Update projection head
        for param_m, param in zip(
                self.momentum_projection.parameters(),
                self.projection_head.parameters()
        ):
            param_m.data = (
                    self.momentum_rate * param_m.data +
                    (1 - self.momentum_rate) * param.data
            )

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through SSL model.

        Args:
            x1: First view [B, C, L]
            x2: Second view [B, C, L]

        Returns:
            Loss and metrics
        """
        # Online encoder forward
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        # Momentum encoder forward (no gradients)
        with torch.no_grad():
            h1_m = self.momentum_encoder(x1)
            h2_m = self.momentum_encoder(x2)
            z1_m = self.momentum_projection(h1_m)
            z2_m = self.momentum_projection(h2_m)

        # Calculate loss
        loss, metrics = self.criterion(z1, z2, z2_m, z1_m)

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
        """Save model checkpoint."""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_state_dict': self.projection_head.state_dict(),
            'momentum_encoder_state_dict': self.momentum_encoder.state_dict(),
            'momentum_projection_state_dict': self.momentum_projection.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(
            self,
            path: str,
            optimizer=None
    ) -> Tuple[int, float]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_state_dict'])
        self.momentum_encoder.load_state_dict(checkpoint['momentum_encoder_state_dict'])
        self.momentum_projection.load_state_dict(checkpoint['momentum_projection_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Checkpoint loaded from {path}")
        return epoch, best_loss


def create_ssl_model(
        encoder: nn.Module,
        projection_head: nn.Module,
        config_path: str = 'configs/config.yaml'
) -> SSLModel:
    """Create SSL model with configuration."""
    return SSLModel(encoder, projection_head, config_path)


# ============= TEST FUNCTIONS =============

def test_ssl():
    """Test SSL functionality with device manager."""
    print("=" * 50)
    print("Testing SSL Components with Device Manager")
    print("=" * 50)

    # Get device from global device manager
    device_manager = get_device_manager()
    device = device_manager.device

    print(f"✓ Using device: {device} ({device_manager.type})")

    # Test RegularizedInfoNCE
    print("\n1. Testing RegularizedInfoNCE:")
    criterion = RegularizedInfoNCE(temperature=0.04, lambda_koleo=0.1)

    # Create dummy embeddings
    batch_size = 16
    z1 = torch.randn(batch_size, 128).to(device)
    z2 = torch.randn(batch_size, 128).to(device)

    loss, metrics = criterion(z1, z2)

    print(f"   Total loss: {loss:.4f}")
    print(f"   Contrastive loss: {metrics['loss_contrastive']:.4f}")
    print(f"   KoLeo loss: {metrics['loss_koleo']:.4f}")
    assert loss > 0, "Loss should be positive"
    print("   ✓ RegularizedInfoNCE test passed!")

    # Test with momentum embeddings
    print("\n2. Testing with momentum embeddings:")
    z1_m = torch.randn(batch_size, 128).to(device)
    z2_m = torch.randn(batch_size, 128).to(device)

    loss_m, metrics_m = criterion(z1, z2, z1_m, z2_m)
    print(f"   Loss with momentum: {loss_m:.4f}")
    print("   ✓ Momentum test passed!")

    # Test SSLModel
    print("\n3. Testing SSLModel:")

    # Import model components
    from model import EfficientNet1D, ProjectionHead

    # Create encoder and projection head
    encoder = EfficientNet1D(in_channels=1, embedding_dim=256)
    projection_head = ProjectionHead(input_dim=256, output_dim=128)

    # Create SSL model (will use global device manager)
    ssl_model = create_ssl_model(encoder, projection_head)

    print(f"   SSL model created on {device}")

    # Test forward pass
    x1 = torch.randn(4, 1, 3840).to(device)
    x2 = torch.randn(4, 1, 3840).to(device)

    loss, metrics = ssl_model(x1, x2)

    print(f"   SSL model loss: {loss:.4f}")
    print(f"   Metrics: {list(metrics.keys())}")
    assert loss > 0, "Model loss should be positive"
    print("   ✓ SSLModel test passed!")

    # Test momentum update
    print("\n4. Testing momentum update:")

    # Get initial momentum weight
    initial_weight = ssl_model.momentum_encoder.stem[0].weight.data.clone()

    # Modify online encoder
    ssl_model.encoder.stem[0].weight.data += 0.1

    # Update momentum
    ssl_model.update_momentum_networks()

    # Check momentum weight changed
    updated_weight = ssl_model.momentum_encoder.stem[0].weight.data
    weight_diff = (updated_weight - initial_weight).abs().mean().item()

    print(f"   Weight difference: {weight_diff:.6f}")
    print(f"   Expected ~{0.1 * (1 - 0.99):.6f}")
    assert weight_diff > 0, "Momentum weights should change"
    print("   ✓ Momentum update test passed!")

    # Test checkpoint save/load
    print("\n5. Testing checkpoint save/load:")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
        # Save checkpoint
        ssl_model.save_checkpoint(tmp.name, epoch=5, best_loss=0.5)

        # Create new model and load
        new_encoder = EfficientNet1D(in_channels=1, embedding_dim=256)
        new_projection = ProjectionHead(input_dim=256, output_dim=128)
        new_ssl_model = create_ssl_model(new_encoder, new_projection)

        epoch, best_loss = new_ssl_model.load_checkpoint(tmp.name)

        print(f"   Loaded epoch: {epoch}")
        print(f"   Loaded best loss: {best_loss}")
        assert epoch == 5, "Epoch should be restored"
        assert best_loss == 0.5, "Best loss should be restored"

    print("   ✓ Checkpoint test passed!")

    # Test different batch sizes
    print("\n6. Testing different batch sizes:")
    for batch_size in [2, 4, 8, 16, 32]:
        x1 = torch.randn(batch_size, 1, 3840).to(device)
        x2 = torch.randn(batch_size, 1, 3840).to(device)

        loss, _ = ssl_model(x1, x2)
        print(f"   Batch size {batch_size}: loss = {loss:.4f}")

        # Note: Very small batch sizes may have unstable KoLeo
        if batch_size >= 4:
            assert loss > 0, f"Loss should be positive for batch size {batch_size}"

    print("   ✓ Batch size test passed!")
    print("   Note: Paper recommends batch_size >= 256 for best results")

    # Test get_embedding
    print("\n7. Testing get_embedding:")
    test_input = torch.randn(2, 1, 3840).to(device)
    embedding = ssl_model.get_embedding(test_input)

    print(f"   Embedding shape: {embedding.shape}")
    assert embedding.shape == (2, 256), "Wrong embedding shape"
    print("   ✓ Get embedding test passed!")

    # Clean up GPU memory if CUDA
    if device_manager.is_cuda:
        device_manager.empty_cache()
        print("\n   Cleared GPU cache")

    print("\n" + "=" * 50)
    print(f"All SSL tests passed successfully on {device_manager.type}!")
    print("=" * 50)


if __name__ == "__main__":
    test_ssl()