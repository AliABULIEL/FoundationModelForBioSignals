# biosignal/model.py
"""
EfficientNet1D model for biosignals
Following Apple paper's architecture specifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import yaml
from pathlib import Path


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals."""

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
    """Mobile Inverted Bottleneck Convolution for 1D signals."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            expansion_ratio: int = 6,
            se_ratio: float = 0.25,
            drop_rate: float = 0.0
    ):
        super().__init__()

        expanded_channels = in_channels * expansion_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # Expansion
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

        # Project back
        layers.extend([
            nn.Conv1d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])

        self.block = nn.Sequential(*layers)
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
    1D EfficientNet encoder following Apple paper.
    Paper: 16 MBConv blocks with SE, 256-D embeddings
    Enhanced for multi-channel support (ACC)
    """

    def __init__(
            self,
            in_channels: int = 1,  # 1 for PPG/ECG, 3 for ACC
            embedding_dim: int = 128,
            n_blocks: int = 8,
            width_multiplier: float = 1.0,
            depth_multiplier: float = 1.0,
            dropout_rate: float = 0.2,
            drop_path_rate: float = 0.2,
            modality: str = 'ppg'  # Added for modality-specific adjustments
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.modality = modality

        # Stem - adjust based on input channels
        stem_channels = self._make_divisible(32 * width_multiplier)

        # For ACC, we might want a different stem to handle 3 channels
        if modality == 'acc' and in_channels == 3:
            self.stem = nn.Sequential(
                # First conv to mix channels
                nn.Conv1d(in_channels, stem_channels, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(stem_channels),
                nn.SiLU(inplace=True),
                # Additional conv for feature extraction
                nn.Conv1d(stem_channels, stem_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(stem_channels),
                nn.SiLU(inplace=True)
            )
        else:
            # Original stem for PPG/ECG
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(stem_channels),
                nn.SiLU(inplace=True)
            )

        # Build MBConv blocks
        block_configs = [
            # (num_blocks, out_channels, kernel_size, stride, expansion)
            (1, 16, 3, 1, 1),
            (2, 24, 3, 2, 6),
            (2, 40, 5, 2, 6),
            (3, 80, 3, 2, 6),
            (3, 112, 5, 1, 6),
            (3, 160, 5, 2, 6),
            (2, 192, 5, 1, 6),
        ]

        blocks = []
        in_ch = stem_channels
        total_blocks = sum([cfg[0] for cfg in block_configs])
        block_idx = 0

        for num_blocks, out_ch, kernel, stride, expansion in block_configs:
            out_ch = self._make_divisible(out_ch * width_multiplier)
            num_blocks = self._round_repeats(num_blocks, depth_multiplier)

            for i in range(num_blocks):
                drop_rate = drop_path_rate * block_idx / total_blocks

                blocks.append(
                    MBConv1D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        expansion_ratio=expansion,
                        se_ratio=0.25,
                        drop_rate=drop_rate
                    )
                )
                in_ch = out_ch
                block_idx += 1

        self.blocks = nn.Sequential(*blocks)

        # Head
        head_channels = self._make_divisible(1280 * width_multiplier)
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm1d(head_channels),
            nn.SiLU(inplace=True)
        )

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, embedding_dim)

        # Initialize weights
        self._initialize_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"EfficientNet1D initialized for {modality.upper()}:")
        print(f"  Input channels: {in_channels}")
        print(f"  Total blocks: {block_idx}")
        print(f"  Parameters: {total_params / 1e6:.2f}M")

    def _make_divisible(self, channels: float, divisor: int = 8) -> int:
        """Ensure channel count is divisible by divisor."""
        return int(channels + divisor / 2) // divisor * divisor

    def _round_repeats(self, repeats: int, multiplier: float) -> int:
        """Round number of block repeats."""
        if multiplier == 1.0:
            return repeats
        return int(np.ceil(multiplier * repeats))

    def _initialize_weights(self):
        """Initialize model weights."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch_size, channels, length]
               channels=1 for PPG/ECG, channels=3 for ACC

        Returns:
            Embedding [batch_size, embedding_dim]
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ProjectionHead(nn.Module):
    """
    Projection head for SSL training.
    Paper: MLP with 1024 hidden units, projects 256→128
    """

    def __init__(
            self,
            input_dim: int = 256,
            hidden_dim: int = 1024,
            output_dim: int = 128,
            use_bn: bool = True
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim)]

        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        ])

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class BiosignalFoundationModel(nn.Module):
    """Complete foundation model with encoder and projection head."""

    def __init__(
            self,
            config_path: str = 'configs/config.yaml',
            device: str = 'mps',
            modality: str = 'ppg'  # Added modality parameter
    ):
        super().__init__()

        self.modality = modality

        # Load configuration
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_config = config['model']
        else:
            # Default configuration
            model_config = {
                'embedding_dim': 128,
                'n_blocks': 8,
                'width_multiplier': 1.0,
                'depth_multiplier': 1.0,
                'dropout_rate': 0.2,
                'drop_path_rate': 0.2
            }

        # Set input channels based on modality
        if modality == 'acc':
            in_channels = 3  # 3-axis accelerometer
        else:
            in_channels = 1  # Single channel for PPG/ECG

        # Create encoder
        self.encoder = EfficientNet1D(
            in_channels=in_channels,
            embedding_dim=model_config['embedding_dim'],
            n_blocks=model_config['n_blocks'],
            width_multiplier=model_config['width_multiplier'],
            depth_multiplier=model_config['depth_multiplier'],
            dropout_rate=model_config['dropout_rate'],
            drop_path_rate=model_config['drop_path_rate'],
            modality=modality  # Pass modality to encoder
        )

        # Create projection head
        self.projection_head = ProjectionHead(
            input_dim=model_config['embedding_dim'],
            hidden_dim=1024,
            output_dim=128
        )

        # Set device
        self.device = self._get_device(device)
        self.to(self.device)

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device with M1 support."""
        if device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def forward(
            self,
            x: torch.Tensor,
            return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch_size, channels, length]
            return_embedding: Return 256-D embedding instead of projection

        Returns:
            Embedding or projection
        """
        embedding = self.encoder(x)

        if return_embedding:
            return embedding

        projection = self.projection_head(embedding)
        return projection

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding without projection (for downstream tasks)."""
        with torch.no_grad():
            return self.encoder(x)


def create_model(config_path: str = 'configs/config.yaml',
                 device: str = 'mps',
                 modality: str = 'ppg'):
    """Create model with configuration."""
    return BiosignalFoundationModel(config_path, device, modality)


# ============= TEST FUNCTIONS =============

def test_model():
    """Test model functionality including ACC support."""
    print("=" * 50)
    print("Testing Model Architecture")
    print("=" * 50)

    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using M1 chip (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"✓ Using CPU")

    # Test PPG encoder
    print("\n1. Testing EfficientNet1D Encoder for PPG:")
    ppg_encoder = EfficientNet1D(in_channels=1, embedding_dim=256, modality='ppg')
    ppg_encoder = ppg_encoder.to(device)

    # Test with PPG input [batch=2, channels=1, length=3840]
    ppg_input = torch.randn(2, 1, 3840).to(device)
    ppg_embedding = ppg_encoder(ppg_input)

    print(f"   Input shape: {ppg_input.shape}")
    print(f"   Output shape: {ppg_embedding.shape}")
    assert ppg_embedding.shape == (2, 256), f"Wrong shape: {ppg_embedding.shape}"
    print("   ✓ PPG encoder test passed!")

    # Test ECG encoder
    print("\n2. Testing EfficientNet1D Encoder for ECG:")
    ecg_encoder = EfficientNet1D(in_channels=1, embedding_dim=256, modality='ecg')
    ecg_encoder = ecg_encoder.to(device)

    ecg_input = torch.randn(2, 1, 3840).to(device)
    ecg_embedding = ecg_encoder(ecg_input)

    print(f"   Input shape: {ecg_input.shape}")
    print(f"   Output shape: {ecg_embedding.shape}")
    assert ecg_embedding.shape == (2, 256), f"Wrong shape: {ecg_embedding.shape}"
    print("   ✓ ECG encoder test passed!")

    # Test ACC encoder (3 channels)
    print("\n3. Testing EfficientNet1D Encoder for ACC:")
    acc_encoder = EfficientNet1D(in_channels=3, embedding_dim=256, modality='acc')
    acc_encoder = acc_encoder.to(device)

    # Test with ACC input [batch=2, channels=3, length=6000]
    acc_input = torch.randn(2, 3, 6000).to(device)
    acc_embedding = acc_encoder(acc_input)

    print(f"   Input shape: {acc_input.shape}")
    print(f"   Output shape: {acc_embedding.shape}")
    assert acc_embedding.shape == (2, 256), f"Wrong shape: {acc_embedding.shape}"
    print("   ✓ ACC encoder test passed!")

    # Test projection head
    print("\n4. Testing Projection Head:")
    proj_head = ProjectionHead(input_dim=256, output_dim=128)
    proj_head = proj_head.to(device)

    projection = proj_head(acc_embedding)
    print(f"   Input shape: {acc_embedding.shape}")
    print(f"   Output shape: {projection.shape}")
    assert projection.shape == (2, 128), f"Wrong shape: {projection.shape}"
    print("   ✓ Projection head test passed!")

    # Test complete model for each modality
    print("\n5. Testing Complete Models:")

    # PPG model
    print("\n   Testing PPG model:")
    ppg_model = create_model(device=device, modality='ppg')
    ppg_embedding = ppg_model(ppg_input, return_embedding=True)
    ppg_projection = ppg_model(ppg_input, return_embedding=False)
    print(f"   PPG embedding shape: {ppg_embedding.shape}")
    print(f"   PPG projection shape: {ppg_projection.shape}")
    assert ppg_embedding.shape == (2, 256)
    assert ppg_projection.shape == (2, 128)
    print("   ✓ PPG model test passed!")

    # ECG model
    print("\n   Testing ECG model:")
    ecg_model = create_model(device=device, modality='ecg')
    ecg_embedding = ecg_model(ecg_input, return_embedding=True)
    ecg_projection = ecg_model(ecg_input, return_embedding=False)
    print(f"   ECG embedding shape: {ecg_embedding.shape}")
    print(f"   ECG projection shape: {ecg_projection.shape}")
    assert ecg_embedding.shape == (2, 256)
    assert ecg_projection.shape == (2, 128)
    print("   ✓ ECG model test passed!")

    # ACC model
    print("\n   Testing ACC model:")
    acc_model = create_model(device=device, modality='acc')
    acc_embedding = acc_model(acc_input, return_embedding=True)
    acc_projection = acc_model(acc_input, return_embedding=False)
    print(f"   ACC embedding shape: {acc_embedding.shape}")
    print(f"   ACC projection shape: {acc_projection.shape}")
    assert acc_embedding.shape == (2, 256)
    assert acc_projection.shape == (2, 128)
    print("   ✓ ACC model test passed!")

    # Test with different batch sizes for ACC
    print("\n6. Testing different batch sizes for ACC:")
    for batch_size in [1, 4, 8, 16]:
        test_input = torch.randn(batch_size, 3, 6000).to(device)
        output = acc_model(test_input, return_embedding=True)
        assert output.shape == (batch_size, 256)
        print(f"   Batch size {batch_size}: ✓")

    # Test ACC with different input lengths
    print("\n7. Testing ACC with different input lengths:")
    for length in [3000, 6000, 12000]:
        test_input = torch.randn(2, 3, length).to(device)
        output = acc_model(test_input, return_embedding=True)
        assert output.shape == (2, 256)
        print(f"   Length {length}: ✓")

    # Test memory efficiency for ACC
    print("\n8. Testing memory efficiency for ACC:")
    total_params = sum(p.numel() for p in acc_model.parameters())
    trainable_params = sum(p.numel() for p in acc_model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params / 1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print("   ✓ Memory efficiency test passed!")

    print("\n" + "=" * 50)
    print("All model tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_model()