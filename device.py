# biosignal/device.py
"""
Centralized device management for the biosignal foundation model.
Supports CUDA, MPS (Apple Silicon), and CPU with automatic detection.
"""

import torch
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Singleton device manager that handles device selection and management
    across the entire application.
    """

    _instance = None
    _device = None
    _device_type = None
    _device_properties = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._auto_select_device()

    def _auto_select_device(self):
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            self._setup_cuda()
        elif torch.backends.mps.is_available():
            self._setup_mps()
        else:
            self._setup_cpu()

    def _setup_cuda(self):
        """Setup CUDA device with properties."""
        self._device = torch.device('cuda')
        self._device_type = 'cuda'

        # Get CUDA properties
        self._device_properties = {
            'type': 'cuda',
            'name': torch.cuda.get_device_name(0),
            'capability': torch.cuda.get_device_capability(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'memory_allocated': torch.cuda.memory_allocated(0),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'supports_amp': True,  # CUDA supports automatic mixed precision
            'supports_compile': True,  # CUDA supports torch.compile
        }

        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        logger.info(f"CUDA device initialized: {self._device_properties['name']}")
        logger.info(f"  Memory: {self._device_properties['memory_total'] / 1e9:.2f} GB")
        logger.info(f"  Devices available: {self._device_properties['device_count']}")

    def _setup_mps(self):
        """Setup MPS (Apple Silicon) device."""
        self._device = torch.device('mps')
        self._device_type = 'mps'

        self._device_properties = {
            'type': 'mps',
            'name': 'Apple Silicon (MPS)',
            'supports_amp': False,  # MPS doesn't support AMP yet
            'supports_compile': False,  # Limited torch.compile support on MPS
        }

        logger.info("MPS device initialized (Apple Silicon)")

    def _setup_cpu(self):
        """Setup CPU device."""
        self._device = torch.device('cpu')
        self._device_type = 'cpu'

        import platform
        self._device_properties = {
            'type': 'cpu',
            'name': platform.processor() or 'CPU',
            'threads': torch.get_num_threads(),
            'supports_amp': False,
            'supports_compile': True,
        }

        logger.info(f"CPU device initialized: {self._device_properties['name']}")
        logger.info(f"  Threads: {self._device_properties['threads']}")

    def set_device(self, device: Optional[str] = None, device_id: int = 0):
        """
        Manually set the device.

        Args:
            device: Device string ('cuda', 'mps', 'cpu', or None for auto)
            device_id: CUDA device ID (only used for multi-GPU)
        """
        if device is None:
            self._auto_select_device()
            return

        device = device.lower()

        if device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to auto-select")
                self._auto_select_device()
            else:
                if device_id > 0 and device_id < torch.cuda.device_count():
                    self._device = torch.device(f'cuda:{device_id}')
                    torch.cuda.set_device(device_id)
                self._setup_cuda()

        elif device == 'mps':
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to auto-select")
                self._auto_select_device()
            else:
                self._setup_mps()

        elif device == 'cpu':
            self._setup_cpu()
        else:
            logger.warning(f"Unknown device '{device}', using auto-select")
            self._auto_select_device()

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            self._auto_select_device()
        return self._device

    @property
    def type(self) -> str:
        """Get device type string."""
        return self._device_type

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self._device_type == 'cuda'

    @property
    def is_mps(self) -> bool:
        """Check if using MPS."""
        return self._device_type == 'mps'

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self._device_type == 'cpu'

    @property
    def supports_amp(self) -> bool:
        """Check if device supports automatic mixed precision."""
        return self._device_properties.get('supports_amp', False)

    @property
    def supports_compile(self) -> bool:
        """Check if device supports torch.compile."""
        return self._device_properties.get('supports_compile', False)

    def get_properties(self) -> Dict[str, Any]:
        """Get device properties."""
        return self._device_properties.copy()

    def empty_cache(self):
        """Empty cache if applicable (CUDA/MPS)."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            # MPS cache management
            torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

    def synchronize(self):
        """Synchronize device if applicable."""
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_mps:
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None

    def memory_stats(self) -> Dict[str, float]:
        """Get memory statistics if available."""
        stats = {}

        if self.is_cuda:
            stats = {
                'allocated': torch.cuda.memory_allocated() / 1e9,  # GB
                'reserved': torch.cuda.memory_reserved() / 1e9,  # GB
                'free': (self._device_properties.get('memory_total', 0) -
                         torch.cuda.memory_allocated()) / 1e9,  # GB
            }

        return stats

    def get_optimal_batch_size(self, modality: str = 'ppg') -> int:
        """
        Get recommended batch size based on device and modality.

        Args:
            modality: 'ppg', 'ecg', or 'acc'

        Returns:
            Recommended batch size
        """
        # Base recommendations
        base_sizes = {
            'ppg': {'cuda': 256, 'mps': 64, 'cpu': 16},
            'ecg': {'cuda': 256, 'mps': 64, 'cpu': 16},
            'acc': {'cuda': 128, 'mps': 32, 'cpu': 8},  # ACC has 3 channels
        }

        batch_size = base_sizes.get(modality, base_sizes['ppg']).get(self._device_type, 16)

        # Adjust for available memory (CUDA only)
        if self.is_cuda:
            total_memory = self._device_properties.get('memory_total', 0)
            if total_memory > 0:
                if total_memory < 8e9:  # Less than 8GB
                    batch_size = batch_size // 2
                elif total_memory > 16e9:  # More than 16GB
                    batch_size = min(batch_size * 2, 512)

        return batch_size

    def get_num_workers(self) -> int:
        """Get recommended number of DataLoader workers."""
        if self.is_cuda:
            return min(8, os.cpu_count() or 4)
        elif self.is_mps:
            return min(4, os.cpu_count() or 2)
        else:
            return 0  # CPU: use main thread

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device}, type={self._device_type})"


# Global instance getter
def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    return DeviceManager()


# Convenience functions
def get_device() -> torch.device:
    """Get the current device."""
    return get_device_manager().device


def to_device(tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
    """
    Move tensor to the managed device.

    Args:
        tensor: Input tensor
        non_blocking: Use non-blocking transfer (for CUDA)

    Returns:
        Tensor on device
    """
    manager = get_device_manager()
    if manager.is_cuda:
        return tensor.to(manager.device, non_blocking=non_blocking)
    else:
        return tensor.to(manager.device)


def optimize_for_device(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply device-specific optimizations to model.

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    manager = get_device_manager()

    # Move to device
    model = model.to(manager.device)

    # Apply optimizations
    if manager.is_cuda and manager.supports_compile:
        # Optional: compile model for faster execution (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode='default')
            logger.info("Model compiled with torch.compile")
        except:
            pass

    return model