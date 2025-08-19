import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SqueezeExcitation1D(nn.Module):
    """
    Squeeze-and-Excitation module for 1D signals.
    Paper mentions using SE blocks in their MBConv blocks.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv1D(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution for 1D signals.
    Following the paper's EfficientNet-style architecture with SE blocks.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            expansion_ratio: int = 6,  # Paper uses 6 for EfficientNet
            se_ratio: float = 0.25,  # SE reduction ratio
            drop_rate: float = 0.0
    ):
        super().__init__()

        # Expansion phase
        expanded_channels = in_channels * expansion_ratio

        # Check if we use residual connection
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Build the block
        layers = []

        # Expand with 1x1 conv if needed
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                nn.SiLU(inplace=True)
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv1d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU(inplace=True)
        ])

        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(
                SqueezeExcitation1D(expanded_channels, int(1 / se_ratio))
            )

        # Project back to output channels
        layers.extend([
            nn.Conv1d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

        # Stochastic depth (drop path) for training
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            if self.drop_rate > 0 and self.training:
                # Stochastic depth
                survival_prob = 1.0 - self.drop_rate
                if torch.rand(1).item() < survival_prob:
                    out = out / survival_prob + x
                else:
                    out = x
            else:
                out = out + x

        return out


class EfficientNet1D(nn.Module):
    """
    1D EfficientNet encoder following the paper's specifications.
    Paper mentions: 16 MBConv blocks with SE for PPG/ECG encoding.
    """

    def __init__(
            self,
            in_channels: int = 1,  # 1 for ECG, 4 for PPG
            embedding_dim: int = 256,  # Paper uses 256-D embeddings
            width_multiplier: float = 1.0,
            depth_multiplier: float = 1.0,
            modality: str = 'ecg',
            dropout_rate: float = 0.2,
            drop_path_rate: float = 0.2  # Stochastic depth
    ):
        super().__init__()

        self.modality = modality.lower()

        # Adjust channels based on modality
        if self.modality == 'ppg':
            in_channels = 4  # PPG has 4 channels in the paper
        else:
            in_channels = 1  # ECG is single channel

        # Stem: Initial convolution
        stem_channels = self._make_divisible(32 * width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True)
        )

        # Build MBConv blocks following paper's 16-block architecture
        # Configuration: (num_blocks, out_channels, kernel_size, stride, expansion)
        # Paper mentions 16 blocks total
        block_configs = [
            # Stage 1
            (2, 16, 3, 1, 1),  # No expansion for first stage
            # Stage 2
            (2, 24, 3, 2, 6),  # Downsample
            # Stage 3
            (3, 40, 5, 2, 6),  # Downsample
            # Stage 4
            (3, 80, 3, 2, 6),  # Downsample
            # Stage 5
            (4, 112, 5, 1, 6),
            # Stage 6
            (2, 192, 5, 2, 6),  # Downsample
        ]

        # Apply depth multiplier
        total_blocks = sum([cfg[0] for cfg in block_configs])

        # Build blocks
        blocks = []
        in_ch = stem_channels
        block_idx = 0
        total_blocks_built = 0

        for num_blocks, out_ch, kernel, stride, expansion in block_configs:
            out_ch = self._make_divisible(out_ch * width_multiplier)
            num_blocks = self._round_repeats(num_blocks, depth_multiplier)

            for i in range(num_blocks):
                # Calculate drop path rate for this block
                drop_rate = drop_path_rate * block_idx / total_blocks

                blocks.append(
                    MBConv1D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        expansion_ratio=expansion,
                        se_ratio=0.25,  # Paper uses SE blocks
                        drop_rate=drop_rate
                    )
                )
                in_ch = out_ch
                block_idx += 1
                total_blocks_built += 1

        self.blocks = nn.Sequential(*blocks)

        # Head: Final layers
        head_channels = self._make_divisible(1280 * width_multiplier)
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm1d(head_channels),
            nn.SiLU(inplace=True)
        )

        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        # Final FC to get 256-D embedding
        self.fc = nn.Linear(head_channels, embedding_dim)

        # Initialize weights
        self._initialize_weights()

        # Print model statistics to match paper
        self._print_model_stats(total_blocks_built)

    def _make_divisible(self, channels: float, divisor: int = 8) -> int:
        """Ensure channel count is divisible by divisor."""
        return int(channels + divisor / 2) // divisor * divisor

    def _round_repeats(self, repeats: int, multiplier: float) -> int:
        """Round number of block repeats based on depth multiplier."""
        if multiplier == 1.0:
            return repeats
        return int(np.ceil(multiplier * repeats))

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _print_model_stats(self, num_blocks: int):
        """Print model statistics to verify it matches the paper."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"EfficientNet1D for {self.modality.upper()}:")
        print(f"  Total blocks: {num_blocks} (paper specifies 16)")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        if self.modality == 'ppg':
            print(f"  Expected: ~3.3M parameters")
        else:
            print(f"  Expected: ~2.5M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor [batch_size, channels, length]

        Returns:
            Embedding tensor [batch_size, embedding_dim]
        """
        # Stem
        x = self.stem(x)

        # Main blocks
        x = self.blocks(x)

        # Head
        x = self.head(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Dropout and final projection
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Paper specifies: MLP with one hidden layer of 1024 units,
    projecting 256-D embedding to 128-D for loss computation.
    """

    def __init__(
            self,
            input_dim: int = 256,  # Embedding dimension
            hidden_dim: int = 1024,  # Paper specifies 1024
            output_dim: int = 128,  # Paper specifies 128
            use_bn: bool = True
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]

        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings to contrastive space."""
        return self.projection(x)


class BiosignalFoundationModel(nn.Module):
    """
    Complete foundation model combining encoder and projection head.
    Matches the paper's architecture specifications.
    """

    def __init__(
            self,
            modality: str = 'ecg',
            embedding_dim: int = 256,
            projection_dim: int = 128,
            width_multiplier: float = 1.0,
            depth_multiplier: float = 1.0
    ):
        super().__init__()

        # Encoder (EfficientNet1D)
        self.encoder = EfficientNet1D(
            embedding_dim=embedding_dim,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            modality=modality
        )

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=1024,  # Paper specification
            output_dim=projection_dim
        )

        # For momentum encoder (will be copied, not trained directly)
        self.momentum_encoder = None
        self.momentum_projection = None

    def forward(
            self,
            x: torch.Tensor,
            return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor [batch_size, channels, length]
            return_embedding: If True, return embedding before projection

        Returns:
            If return_embedding: 256-D embedding
            Else: 128-D projection for contrastive loss
        """
        # Get embedding from encoder
        embedding = self.encoder(x)

        if return_embedding:
            return embedding

        # Project for contrastive learning
        projection = self.projection_head(embedding)

        return projection

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding without projection (for downstream tasks)."""
        return self.encoder(x)


def create_model_pair(modality: str = 'ecg', **kwargs) -> Tuple[BiosignalFoundationModel, BiosignalFoundationModel]:
    """
    Create online and momentum model pair for training.

    Returns:
        Tuple of (online_model, momentum_model)
    """
    # Create online model
    online_model = BiosignalFoundationModel(modality=modality, **kwargs)

    # Create momentum model (exact copy)
    momentum_model = BiosignalFoundationModel(modality=modality, **kwargs)

    # Copy weights
    momentum_model.load_state_dict(online_model.state_dict())

    # Freeze momentum model
    for param in momentum_model.parameters():
        param.requires_grad = False

    return online_model, momentum_model


# Backward compatibility
def create_encoder(modality: str = 'ecg', **kwargs) -> EfficientNet1D:
    """Create just the encoder for evaluation."""
    return EfficientNet1D(modality=modality, **kwargs)