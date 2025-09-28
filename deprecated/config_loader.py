# biosignal/config_loader.py
"""
Centralized configuration management for the biosignal foundation model.
Provides a singleton ConfigLoader that loads and caches configuration from YAML.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import copy
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Singleton configuration loader that provides centralized access to all configuration.
    Loads configuration once and provides easy access to nested values.
    """

    _instance = None
    _config = None
    _config_path = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = 'configs/tabpfn_vitaldb.yaml'):
        """
        Initialize the ConfigLoader with a config file path.

        Args:
            config_path: Path to the YAML configuration file
        """
        if self._initialized and self._config_path == config_path:
            return

        self._config_path = config_path
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(self._config_path)

        if not config_path.exists():
            # Try relative to current file
            config_path = Path(__file__).parent / self._config_path
        
        if not config_path.exists():
            # Try from parent directory (when running from tests/)
            config_path = Path(__file__).parent.parent / self._config_path
            
        if not config_path.exists():
            # Try absolute path from project root
            possible_roots = [
                Path.cwd(),
                Path.cwd().parent,
                Path('/Users/aliab/Desktop/FoundationModelForBioSignals'),
            ]
            for root in possible_roots:
                potential_path = root / self._config_path
                if potential_path.exists():
                    config_path = potential_path
                    break

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}. Searched in {Path.cwd()} and parent directories")

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            
            # No strict mode validation - let config determine mode
            # self._validate_tabular_mode()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _validate_tabular_mode(self):
        """
        Validate that configuration is set to tabular mode only.
        Raises RuntimeError if SSL mode is detected.
        """
        # Check dataset mode
        dataset_mode = self.get('dataset.mode', default='tabular')
        if dataset_mode != 'tabular':
            raise RuntimeError(
                f"Only tabular mode is supported. Got mode='{dataset_mode}'. "
                "SSL training has been deprecated. "
                "Please set dataset.mode='tabular' in your config. "
                "Legacy SSL code has been moved to deprecated/ssl/"
            )
        
        # Check for SSL-specific configurations that shouldn't be used
        if 'ssl' in self._config and self._config['ssl'].get('enabled', False):
            raise RuntimeError(
                "SSL configuration detected but SSL is deprecated. "
                "Remove or disable SSL settings from config. "
                "Use dataset.mode='tabular' with TabPFN."
            )
        
        # Check for simsiam or infonce configurations
        if any(key in self._config for key in ['simsiam', 'infonce']):
            logger.warning(
                "SSL method configurations (simsiam/infonce) found in config. "
                "These are deprecated and will be ignored. "
                "Using TabPFN with tabular mode instead."
            )

    def reload(self, config_path: Optional[str] = None):
        """
        Reload configuration from file.

        Args:
            config_path: Optional new config path. If None, reloads from current path.
        """
        if config_path:
            self._config_path = config_path
        self._load_config()

    @property
    def data_dir(self) -> str:
        """Get data directory path."""
        return self.get('dataset.data', default='data/but_ppg/dataset')

    @property
    def seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.get('seed', default=42)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('training.batch_size')
            64
            >>> config.get('model.embedding_dim')
            256
            >>> config.get('dataset.ppg.target_fs')
            64
        """
        if self._config is None:
            self._load_config()

        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_nested(self, *keys, default: Any = None) -> Any:
        """
        Get a nested configuration value using multiple keys.

        Args:
            *keys: Sequence of keys to navigate nested dict
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get_nested('training', 'batch_size')
            64
            >>> config.get_nested('dataset', 'ppg', 'target_fs')
            64
        """
        if self._config is None:
            self._load_config()

        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Top-level section name

        Returns:
            Dictionary containing the section configuration

        Examples:
            >>> config.get_section('training')
            {'batch_size': 64, 'num_epochs': 30, ...}
        """
        if self._config is None:
            self._load_config()

        return copy.deepcopy(self._config.get(section, {}))

    def get_modality_config(self, modality: str) -> Dict[str, Any]:
        """
        Get configuration for a specific modality (ppg, ecg, acc).

        Args:
            modality: Signal modality name

        Returns:
            Dictionary containing modality-specific configuration
        """
        return self.get_section('dataset').get(modality, {})

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get complete dataset configuration."""
        return self.get_section('dataset')

    def get_downsample_config(self) -> Dict[str, Any]:
        """Get downsampling configuration."""
        return self.get_section('downsample')

    # Just add this one method to ConfigLoader:

    def get_pair_generation_config(self) -> Dict[str, int]:
        """
        Get pair generation configuration for BUT PPG dataset.

        Returns:
            Dictionary with pair generation parameters
        """
        return {
            'pairs_per_participant': self.get('dataset.pairs_per_participant', default=20),
            'max_pairs_per_participant': self.get('dataset.max_pairs_per_participant', default=100),
            'min_recordings_for_pairs': self.get('dataset.min_recordings_for_pairs', default=2)
        }

    def get_augmentation_config(self, modality: str, ssl_method: str = 'infonce') -> Dict[str, float]:
        """
        Get augmentation configuration for a specific modality and SSL method.

        Args:
            modality: Signal modality ('ppg', 'ecg', 'acc')
            ssl_method: SSL method ('infonce' or 'simsiam')

        Returns:
            Dictionary of augmentation probabilities
        """
        if ssl_method == 'simsiam':
            section = 'simsiam'
        else:
            section = 'ssl'

        aug_key = f'augmentations_{modality}'
        return self.get_nested(section, aug_key, default={})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        return self.get_section('model')

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get_section('training')

    def get_ssl_config(self, ssl_method: str = 'infonce') -> Dict[str, Any]:
        """
        Get SSL-specific configuration.

        Args:
            ssl_method: SSL method ('infonce' or 'simsiam')

        Returns:
            SSL configuration dictionary
        """
        if ssl_method == 'simsiam':
            return self.get_section('simsiam')
        else:
            return self.get_section('ssl')

    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration."""
        return self.get_section('device')

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get_section('evaluation')

    def get_paper_benchmarks(self, modality: str) -> Dict[str, float]:
        """
        Get paper benchmark results for comparison.

        Args:
            modality: Signal modality ('ppg', 'ecg', 'acc')

        Returns:
            Dictionary of benchmark metrics
        """
        return self.get_nested('paper_benchmarks', modality, default={})

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary (read-only copy)."""
        if self._config is None:
            self._load_config()
        return copy.deepcopy(self._config)

    @property
    def seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.get('seed', default=42)

    @property
    def dataset_name(self) -> str:
        """Get dataset name."""
        return self.get('dataset.name', default='BUT_PPG')

    @property
    def data_dir(self) -> str:
        """Get data directory path."""
        return self.get('dataset.data', default='data/but_ppg/dataset')

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self._config_path}')"

    def update_for_testing(self, overrides: Dict[str, Any]):
        """
        Temporarily update configuration for testing purposes.

        Args:
            overrides: Dictionary of configuration overrides

        Note:
            This modifies the in-memory config only, not the file.
            Use with caution and primarily for testing.
        """
        if self._config is None:
            self._load_config()

        def update_nested(config, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    update_nested(config[key], value)
                else:
                    config[key] = value

        update_nested(self._config, overrides)


# Global instance getter
def get_config() -> ConfigLoader:
    """
    Get the global ConfigLoader instance.

    Returns:
        ConfigLoader singleton instance

    Examples:
        >>> config = get_config()
        >>> batch_size = config.get('training.batch_size')
    """
    return ConfigLoader()


# Convenience function for direct access
def load_config(config_path: str = 'configs/tabpfn_vitaldb.yaml') -> ConfigLoader:
    """
    Load configuration from a specific file.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance
    """
    config = ConfigLoader()
    if config._config_path != config_path:
        config.reload(config_path)
    return config
