import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from .model import EfficientNet1D, ProjectionHead, BiosignalFoundationModel


class RegularizedInfoNCE(nn.Module):
    """
    Regularized InfoNCE loss exactly as described in the paper.
    Combines InfoNCE (NT-Xent) with KoLeo regularization.

    Paper equations:
    - InfoNCE: L_contrastive = -1/N Σ log(exp(sim(h_i^1, h_i^2)/τ) / Σ_j exp(sim(h_i^1, h_j^2)/τ))
    - KoLeo: L_KoLeo = -1/N Σ log(min_j≠i ||h_i - h_j||_2)
    - Final: L = (L_contrastive^(1,2) + L_contrastive^(2,1))/2 + λ(L_KoLeo^(1) + L_KoLeo^(2))/2
    """

    def __init__(
            self,
            temperature: float = 0.04,  # Paper uses 0.04 for both PPG and ECG
            lambda_koleo: float = 0.1,  # Paper uses 0.1 weight for KoLeo
            normalize_embeddings: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_koleo = lambda_koleo
        self.normalize_embeddings = normalize_embeddings

    def infonce_loss(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            temperature: float
    ) -> torch.Tensor:
        """
        Calculate InfoNCE loss (also known as NT-Xent).

        Args:
            queries: [N, D] tensor of query embeddings
            keys: [N, D] tensor of key embeddings
            temperature: Temperature parameter

        Returns:
            InfoNCE loss value
        """
        batch_size = queries.shape[0]

        # Calculate similarity matrix
        sim_matrix = torch.matmul(queries, keys.T) / temperature

        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=queries.device)

        # Calculate cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def koleo_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate Kozachenko-Leonenko (KoLeo) differential entropy estimator.
        Paper uses this to encourage uniform span of features within batch.

        Args:
            embeddings: [N, D] tensor of embeddings

        Returns:
            KoLeo regularization loss
        """
        batch_size = embeddings.shape[0]

        # Calculate pairwise L2 distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Mask diagonal (distance to self) with large value
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        distances = distances.masked_fill(mask, float('inf'))

        # Find minimum distance for each embedding
        min_distances, _ = distances.min(dim=1)

        # KoLeo loss: -log of minimum distances
        # Add small epsilon to avoid log(0)
        koleo = -torch.log(min_distances + 1e-9).mean()

        return koleo

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            z1_momentum: Optional[torch.Tensor] = None,
            z2_momentum: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate regularized InfoNCE loss following the paper.

        Args:
            z1: First view embeddings from online encoder [N, D]
            z2: Second view embeddings from online encoder [N, D]
            z1_momentum: First view from momentum encoder (if using momentum)
            z2_momentum: Second view from momentum encoder (if using momentum)

        Returns:
            Total loss and dictionary of individual loss components
        """
        # L2 normalize embeddings (paper mentions this)
        if self.normalize_embeddings:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            if z1_momentum is not None:
                z1_momentum = F.normalize(z1_momentum, dim=1)
            if z2_momentum is not None:
                z2_momentum = F.normalize(z2_momentum, dim=1)

        # Use momentum embeddings as keys if available (paper's approach)
        keys1 = z1_momentum if z1_momentum is not None else z1
        keys2 = z2_momentum if z2_momentum is not None else z2

        # Calculate bidirectional InfoNCE loss
        # L^(1,2)_contrastive: z1 queries, z2 keys
        loss_12 = self.infonce_loss(z1, keys2, self.temperature)

        # L^(2,1)_contrastive: z2 queries, z1 keys
        loss_21 = self.infonce_loss(z2, keys1, self.temperature)

        # Average bidirectional losses (paper equation 3)
        loss_contrastive = (loss_12 + loss_21) / 2

        # Calculate KoLeo regularization for both views
        koleo_1 = self.koleo_loss(z1)
        koleo_2 = self.koleo_loss(z2)

        # Average KoLeo losses
        loss_koleo = (koleo_1 + koleo_2) / 2

        # Total loss (paper equation 3)
        total_loss = loss_contrastive + self.lambda_koleo * loss_koleo

        # Return total loss and components for logging
        loss_dict = {
            'loss': total_loss.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_koleo': loss_koleo.item(),
            'loss_12': loss_12.item(),
            'loss_21': loss_21.item(),
            'koleo_1': koleo_1.item(),
            'koleo_2': koleo_2.item()
        }

        return total_loss, loss_dict


class SSLModel(nn.Module):
    """
    Self-supervised learning model with momentum encoder.
    Implements the complete training framework from the paper.
    """

    def __init__(
            self,
            modality: str = 'ecg',
            embedding_dim: int = 256,  # Paper: 256-D embeddings
            projection_dim: int = 128,  # Paper: 128-D projection space
            momentum_rate: float = 0.99,  # Paper: τ = 0.99
            temperature: float = 0.04,  # Paper: 0.04 for InfoNCE
            lambda_koleo: float = 0.1,  # Paper: λ = 0.1
            width_multiplier: float = 1.0,
            depth_multiplier: float = 1.0
    ):
        super().__init__()

        self.modality = modality.lower()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.momentum_rate = momentum_rate

        # Determine input channels based on modality
        in_channels = 4 if self.modality == 'ppg' else 1

        # Online encoder and projection head
        self.encoder = EfficientNet1D(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            modality=modality
        )

        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=1024,  # Paper specifies 1024
            output_dim=projection_dim,
            use_bn=True
        )

        # Momentum encoder and projection head
        self.momentum_encoder = EfficientNet1D(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            modality=modality
        )

        self.momentum_projection = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=1024,
            output_dim=projection_dim,
            use_bn=True
        )

        # Initialize momentum networks
        self._initialize_momentum_networks()

        # Loss function
        self.criterion = RegularizedInfoNCE(
            temperature=temperature,
            lambda_koleo=lambda_koleo
        )

    def _initialize_momentum_networks(self):
        """
        Initialize momentum networks with online network weights.
        Momentum networks are not updated by gradients.
        """
        # Copy encoder weights
        for param_m, param in zip(self.momentum_encoder.parameters(),
                                  self.encoder.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False

        # Copy projection head weights
        for param_m, param in zip(self.momentum_projection.parameters(),
                                  self.projection_head.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def update_momentum_networks(self):
        """
        Update momentum networks using exponential moving average.
        Paper: ξ ← τξ + (1-τ)θ where τ = 0.99
        """
        # Update encoder
        for param_m, param in zip(self.momentum_encoder.parameters(),
                                  self.encoder.parameters()):
            param_m.data = (self.momentum_rate * param_m.data +
                            (1 - self.momentum_rate) * param.data)

        # Update projection head
        for param_m, param in zip(self.momentum_projection.parameters(),
                                  self.projection_head.parameters()):
            param_m.data = (self.momentum_rate * param_m.data +
                            (1 - self.momentum_rate) * param.data)

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the SSL model.

        Args:
            x1: First segment from participant [B, C, L]
            x2: Second segment from same participant [B, C, L]
            return_embeddings: If True, also return embeddings

        Returns:
            Loss value and dictionary of metrics
        """
        # Online encoder forward pass
        # Get embeddings
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Project to contrastive space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        # Momentum encoder forward pass (no gradients)
        with torch.no_grad():
            h1_m = self.momentum_encoder(x1)
            h2_m = self.momentum_encoder(x2)
            z1_m = self.momentum_projection(h1_m)
            z2_m = self.momentum_projection(h2_m)

        # Calculate loss
        # Paper uses cross-predictions: online from view1 → momentum from view2
        loss, loss_dict = self.criterion(z1, z2, z2_m, z1_m)

        if return_embeddings:
            return loss, loss_dict, (h1, h2)

        return loss, loss_dict

    def get_encoder(self) -> nn.Module:
        """Get the encoder for downstream tasks."""
        return self.encoder

    def save_encoder(self, path: str):
        """Save only the encoder weights for downstream evaluation."""
        torch.save(self.encoder.state_dict(), path)

    def save_checkpoint(self, path: str, optimizer=None, epoch=None):
        """Save complete training checkpoint."""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_state_dict': self.projection_head.state_dict(),
            'momentum_encoder_state_dict': self.momentum_encoder.state_dict(),
            'momentum_projection_state_dict': self.momentum_projection.state_dict(),
            'epoch': epoch
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer=None):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_state_dict'])
        self.momentum_encoder.load_state_dict(checkpoint['momentum_encoder_state_dict'])
        self.momentum_projection.load_state_dict(checkpoint['momentum_projection_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0)


# Utility function for creating SSL model
def create_ssl_model(modality: str = 'ecg', **kwargs) -> SSLModel:
    """
    Create SSL model with paper-specified parameters.

    Args:
        modality: 'ecg' or 'ppg'
        **kwargs: Override default parameters

    Returns:
        Configured SSLModel
    """
    defaults = {
        'embedding_dim': 256,
        'projection_dim': 128,
        'momentum_rate': 0.99,
        'temperature': 0.04,
        'lambda_koleo': 0.1
    }

    # Merge with user overrides
    config = {**defaults, **kwargs}

    return SSLModel(modality=modality, **config)