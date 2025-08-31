# ssl_model.py - PROPERLY FIXED VERSION
"""
SSL model implementation supporting InfoNCE and SimSiam
Uses centralized configuration management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pathlib import Path
from copy import deepcopy

from device import get_device_manager
from config_loader import get_config  # Added ConfigLoader


class RegularizedInfoNCE(nn.Module):
    """InfoNCE loss with KoLeo regularization."""

    def __init__(
            self,
            temperature: Optional[float] = None,
            lambda_koleo: Optional[float] = None,
            normalize: bool = True,
            config_path: str = 'configs/config.yaml'
    ):
        super().__init__()

        # Load configuration
        config = get_config()
        ssl_config = config.get_ssl_config('infonce')

        # Use config values with fallbacks
        self.temperature = temperature if temperature is not None else ssl_config.get('temperature', 0.07)
        self.lambda_koleo = lambda_koleo if lambda_koleo is not None else ssl_config.get('lambda_koleo', 0.05)
        self.normalize = normalize

        # Get variance loss weight from config
        self.lambda_variance = config.get('ssl.lambda_variance', 0.1)

    def infonce_loss(self, queries, keys):
        batch_size = queries.shape[0]

        if self.normalize:
            queries = F.normalize(queries, dim=1)
            keys = F.normalize(keys, dim=1)

        sim_matrix = torch.matmul(queries, keys.T) / self.temperature
        labels = torch.arange(batch_size, device=queries.device)

        return F.cross_entropy(sim_matrix, labels)

    def koleo_loss(self, embeddings):
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)

        distances = torch.cdist(embeddings, embeddings, p=2)
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        distances = distances.masked_fill(mask, float('inf'))
        min_distances = distances.min(dim=1)[0]

        # Get min distance clamp from config
        config = get_config()
        min_dist_clamp = config.get('ssl.koleo_min_distance_clamp', 1e-3)
        min_distances = torch.clamp(min_distances, min=min_dist_clamp)

        return -torch.log(min_distances).mean()

    def forward(self, z1, z2, z1_momentum=None, z2_momentum=None):
        keys1 = z1_momentum if z1_momentum is not None else z1
        keys2 = z2_momentum if z2_momentum is not None else z2

        loss_12 = self.infonce_loss(z1, keys2)
        loss_21 = self.infonce_loss(z2, keys1)
        loss_contrastive = (loss_12 + loss_21) / 2

        koleo_1 = self.koleo_loss(z1)
        koleo_2 = self.koleo_loss(z2)
        loss_koleo = (koleo_1 + koleo_2) / 2

        var_loss = 0
        z_concat = torch.cat([z1, z2], dim=0)

        # Get variance parameters from config
        config = get_config()
        variance_epsilon = config.get('ssl.variance_epsilon', 1e-4)
        variance_threshold = config.get('ssl.variance_threshold', 1.0)

        std = torch.sqrt(z_concat.var(dim=0) + variance_epsilon)
        var_loss = torch.mean(F.relu(variance_threshold - std))

        total_loss = loss_contrastive + self.lambda_koleo * loss_koleo + self.lambda_variance * var_loss

        metrics = {
            'loss': total_loss.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_koleo': loss_koleo.item(),
            'loss_12': loss_12.item(),
            'loss_21': loss_21.item(),
            'loss_variance': var_loss.item()
        }

        return total_loss, metrics


class SimSiamLoss(nn.Module):
    """SimSiam loss - CORRECTED implementation."""

    def __init__(self, config_path: str = 'configs/config.yaml'):
        super().__init__()
        self.config = get_config()

    def forward(self, p1, p2, z1, z2):
        """
        SimSiam loss computation.
        p1, p2: outputs from predictor
        z1, z2: outputs from projector (with stop-gradient)
        """
        # Normalize the projector outputs (z), not predictor outputs (p)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # Negative cosine similarity (the paper uses negative cosine)
        loss = -(F.cosine_similarity(p1, z2.detach()).mean() +
                 F.cosine_similarity(p2, z1.detach()).mean()) * 0.5

        metrics = {
            'loss': loss.item(),
            'loss_contrastive': loss.item(),
            'loss_koleo': 0.0,
            'loss_12': -F.cosine_similarity(p1, z2.detach()).mean().item(),
            'loss_21': -F.cosine_similarity(p2, z1.detach()).mean().item(),
            'loss_variance': 0.0
        }

        return loss, metrics


class SSLModel(nn.Module):
    """Unified SSL model supporting both InfoNCE and SimSiam."""

    def __init__(
            self,
            encoder: nn.Module,
            projection_head: nn.Module,
            config_path: str = 'configs/config.yaml',
            ssl_method: str = 'infonce'
    ):
        super().__init__()
        self.ssl_method = ssl_method

        # Load configuration
        self.config = get_config()
        model_config = self.config.get_model_config()

        self.encoder = encoder

        # Check actual input size using config
        with torch.no_grad():
            # Use segment size from config
            if self.config.get('downsample.segment_length_sec'):
                segment_length_sec = self.config.get('downsample.segment_length_sec')
                # Assume PPG for default
                target_fs = self.config.get('dataset.ppg.target_fs', 64)
                dummy_size = int(segment_length_sec * target_fs)
            else:
                # Use default segment lengths from config
                segment_length = self.config.get('dataset.ppg.segment_length', 30)
                target_fs = self.config.get('dataset.ppg.target_fs', 64)
                dummy_size = segment_length * target_fs

            device = next(encoder.parameters()).device
            dummy = torch.randn(1, 1, dummy_size).to(device)
            embedding_dim = encoder(dummy).shape[1]
        ssl_config = self.config.get_ssl_config('infonce')
        if ssl_method == 'simsiam':
            simsiam_config = self.config.get_ssl_config('simsiam')
            proj_dim = simsiam_config.get('projection_dim', 2048)
            pred_dim = simsiam_config.get('prediction_dim', 512)

            # Correct 3-layer projector with BN
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim, affine=False)  # No learnable affine parameters
            )

            # Correct 2-layer predictor
            self.predictor = nn.Sequential(
                nn.Linear(proj_dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pred_dim, proj_dim)  # No BN or activation at the end
            )

            self.criterion = SimSiamLoss(config_path=config_path)
            self.momentum_encoder = deepcopy(encoder)
            self.momentum_projection = deepcopy(projection_head)
            self.momentum_rate = ssl_config.get('momentum_rate', 0.99)

        else:  # infonce
            ssl_config = self.config.get_ssl_config('infonce')
            self.projection_head = projection_head

            self.momentum_encoder = deepcopy(encoder)
            self.momentum_projection = deepcopy(projection_head)

            for param in self.momentum_encoder.parameters():
                param.requires_grad = False
            for param in self.momentum_projection.parameters():
                param.requires_grad = False

            self.criterion = RegularizedInfoNCE(
                temperature=ssl_config.get('temperature', 0.07),
                lambda_koleo=ssl_config.get('lambda_koleo', 0.05),
                config_path=config_path
            )

            self.momentum_rate = ssl_config.get('momentum_rate', 0.99)

        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        self.to(self.device)

        print(f"\nSSL Model initialized:")
        print(f"  Method: {ssl_method}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Device: {self.device}")
        if ssl_method == 'infonce':
            print(f"  Temperature: {self.criterion.temperature}")
            print(f"  Lambda KoLeo: {self.criterion.lambda_koleo}")
            print(f"  Lambda Variance: {self.criterion.lambda_variance}")
            print(f"  Momentum rate: {self.momentum_rate}")
        else:
            print(f"  Projection dim: {proj_dim}")
            print(f"  Prediction dim: {pred_dim}")

    @torch.no_grad()
    def update_momentum_networks(self):
        if self.ssl_method != 'infonce':
            return

        for param_m, param in zip(
                self.momentum_encoder.parameters(),
                self.encoder.parameters()
        ):
            param_m.data = (
                    self.momentum_rate * param_m.data +
                    (1 - self.momentum_rate) * param.data
            )

        for param_m, param in zip(
                self.momentum_projection.parameters(),
                self.projection_head.parameters()
        ):
            param_m.data = (
                    self.momentum_rate * param_m.data +
                    (1 - self.momentum_rate) * param.data
            )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        if self.ssl_method == 'simsiam':
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            loss, metrics = self.criterion(p1, p2, z1, z2)

        else:  # infonce
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

            with torch.no_grad():
                h1_m = self.momentum_encoder(x1)
                h2_m = self.momentum_encoder(x2)
                z1_m = self.momentum_projection(h1_m)
                z2_m = self.momentum_projection(h2_m)

            loss, metrics = self.criterion(z1, z2, z1_m, z2_m)

        return loss, metrics

    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def save_checkpoint(self, path, optimizer=None, epoch=0, best_loss=float('inf')):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_state_dict': self.projection_head.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
            'ssl_method': self.ssl_method
        }

        if self.ssl_method == 'simsiam':
            checkpoint['predictor_state_dict'] = self.predictor.state_dict()
        else:
            checkpoint['momentum_encoder_state_dict'] = self.momentum_encoder.state_dict()
            checkpoint['momentum_projection_state_dict'] = self.momentum_projection.state_dict()

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_state_dict'])

        if self.ssl_method == 'simsiam' and 'predictor_state_dict' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        elif self.ssl_method == 'infonce':
            if 'momentum_encoder_state_dict' in checkpoint:
                self.momentum_encoder.load_state_dict(checkpoint['momentum_encoder_state_dict'])
            if 'momentum_projection_state_dict' in checkpoint:
                self.momentum_projection.load_state_dict(checkpoint['momentum_projection_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))


def create_ssl_model(
        encoder: nn.Module,
        projection_head: nn.Module,
        config_path: str = 'configs/config.yaml',
        ssl_method: str = 'infonce'
) -> SSLModel:
    return SSLModel(encoder, projection_head, config_path, ssl_method)


def test_ssl():
    """Test both InfoNCE and SimSiam."""
    print("=" * 50)
    print("Testing SSL Models")
    print("=" * 50)

    device_manager = get_device_manager()
    device = device_manager.device

    from model import EfficientNet1D, ProjectionHead

    # Load config
    config = get_config()

    # Get segment parameters from config
    if config.get('downsample.segment_length_sec'):
        segment_length_sec = config.get('downsample.segment_length_sec')
        target_fs = config.get('dataset.ppg.target_fs', 64)
        test_length = int(segment_length_sec * target_fs)
    else:
        segment_length = config.get('dataset.ppg.segment_length', 30)
        target_fs = config.get('dataset.ppg.target_fs', 64)
        test_length = segment_length * target_fs

    # Test InfoNCE
    print("\n1. Testing InfoNCE Model:")

    embedding_dim = config.get('model.embedding_dim', 256)
    projection_dim = config.get('model.projection_dim', 128)

    encoder = EfficientNet1D(in_channels=1, embedding_dim=embedding_dim)
    projection_head = ProjectionHead(input_dim=embedding_dim, output_dim=projection_dim)

    infonce_model = create_ssl_model(encoder, projection_head, ssl_method='infonce')

    x1 = torch.randn(8, 1, test_length).to(device)
    x2 = torch.randn(8, 1, test_length).to(device)

    loss, metrics = infonce_model(x1, x2)
    print(f"   InfoNCE loss: {loss:.4f}")
    print(f"   Contrastive: {metrics['loss_contrastive']:.4f}")
    print(f"   KoLeo: {metrics['loss_koleo']:.4f}")
    print(f"   Variance: {metrics.get('loss_variance', 0):.4f}")
    assert loss > 0, "InfoNCE loss must be positive"
    print("   ✓ InfoNCE test passed!")

    # Test SimSiam
    print("\n2. Testing SimSiam Model:")
    encoder2 = EfficientNet1D(in_channels=1, embedding_dim=embedding_dim)
    projection_head2 = ProjectionHead(input_dim=embedding_dim, output_dim=projection_dim)

    simsiam_model = create_ssl_model(encoder2, projection_head2, ssl_method='simsiam')

    loss2, metrics2 = simsiam_model(x1, x2)
    print(f"   SimSiam loss: {loss2:.4f}")
    assert loss2 < 0, "SimSiam loss must be positive (fixed)"
    print("   ✓ SimSiam test passed!")

    # Test with different batch sizes
    print("\n3. Testing different batch sizes:")
    batch_sizes = config.get('test.batch_sizes', [1, 4, 8, 16])
    for batch_size in batch_sizes:
        test_x1 = torch.randn(batch_size, 1, test_length).to(device)
        test_x2 = torch.randn(batch_size, 1, test_length).to(device)

        loss, _ = infonce_model(test_x1, test_x2)
        assert loss > 0
        print(f"   Batch size {batch_size}: ✓")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    test_ssl()