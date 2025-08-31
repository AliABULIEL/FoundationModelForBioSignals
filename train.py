# biosignal/train.py
"""
Training script for biosignal foundation model
Implements SSL training following Apple paper
Enhanced with ACC support and device manager integration
Uses centralized configuration management
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import random
import warnings
from typing import Dict, Tuple, Optional
import time

warnings.filterwarnings('ignore')

# Import modules
from data import create_dataloaders
from model import BiosignalFoundationModel
from augment import ContrastiveAugmentation
from device import DeviceManager, get_device_manager
from ssl_model import create_ssl_model
from config_loader import get_config  # Added ConfigLoader


class Trainer:
    """Trainer class for SSL training with device manager integration."""

    def __init__(
            self,
            config_path: str = 'configs/config.yaml',
            experiment_name: Optional[str] = None,
            device_manager: Optional[DeviceManager] = None,
            downsample: bool = False,
            ssl_method: str = 'infonce'
    ):
        """
        Initialize trainer with device manager.

        Args:
            config_path: Path to configuration file
            experiment_name: Name for this experiment
            device_manager: Device manager instance (if None, creates new one)
            downsample: Whether to use downsampled segments
            ssl_method: SSL method to use ('infonce' or 'simsiam')
        """
        self.early_stopping_patience = None
        self.downsample = downsample
        self.ssl_method = ssl_method

        # Load configuration
        self.config = get_config()
        self.training_config = self.config.get_training_config()

        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"ssl_{timestamp}"
        else:
            self.experiment_name = experiment_name

        # Use device manager instead of internal device selection
        if device_manager is None:
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager

        # Use device from manager
        self.device = self.device_manager.device

        # Set random seeds from config
        self._set_seeds(self.config.seed)

        # Create output directories from config
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'data/outputs/checkpoints')
        self.checkpoint_dir = Path(checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path('data/outputs/logs') / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')

        # Training optimizations based on device and config
        self.use_amp = self.training_config.get('use_amp', False) and self.device_manager.supports_amp
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)

        # Timing statistics
        self.timing_stats = {
            'data_loading': [],
            'forward_pass': [],
            'backward_pass': [],
            'optimizer_step': []
        }

        print(f"\nTrainer initialized:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Device: {self.device} ({self.device_manager.type})")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"  SSL Method: {self.ssl_method}")


    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_data(self, modality: str = 'ppg'):
        """Setup data loaders with device-optimized settings."""
        print(f"\nSetting up {modality.upper()} data loaders...")

        # Use device manager to determine optimal settings
        num_workers = self.device_manager.get_num_workers()
        pin_memory = self.device_manager.is_cuda

        # Get batch size from config
        batch_size = self.training_config.get('batch_size', 64)

        # Create data loaders with optimizations
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=self.config.data_dir,
            modality=modality,
            batch_size=batch_size,
            num_workers=num_workers,
            config_path='configs/config.yaml',
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            quality_filter=False,  # Use all data
            downsample=self.downsample
        )

        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        print(f"  Workers: {num_workers}, Pin memory: {pin_memory}")

        # Create augmentation (it will use global device manager internally)
        self.augmentation = ContrastiveAugmentation(
            modality=modality,
            config_path='configs/config.yaml',
            ssl_method=self.ssl_method
        )

        self.modality = modality

    def setup_model(self, modality: str = 'ppg'):
        """Setup SSL model with modality support."""
        print(f"\nSetting up {modality.upper()} model...")

        # Create foundation model (it will use global device manager internally)
        foundation_model = BiosignalFoundationModel(
            config_path='configs/config.yaml',
            modality=modality
        )

        self.model = create_ssl_model(
            encoder=foundation_model.encoder,
            projection_head=foundation_model.projection_head,
            config_path='configs/config.yaml',
            ssl_method=self.ssl_method
        )

        # Move to device (model already on device from initialization)
        self.model = self.model.to(self.device)

        # Use DataParallel if multiple GPUs
        if self.device_manager.is_cuda and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)

        # Optional: Compile model for faster execution (PyTorch 2.0+)
        compile_model = self.config.get('training.compile_model', False)
        if self.device_manager.supports_compile and compile_model:
            try:
                self.model = torch.compile(self.model, mode='default')
                print("  Model compiled with torch.compile")
            except Exception as e:
                print(f"  Could not compile model: {e}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

        # Show memory usage if CUDA
        if self.device_manager.is_cuda:
            mem_stats = self.device_manager.memory_stats()
            print(f"  GPU memory after model load: {mem_stats.get('allocated', 0):.2f} GB")

    def setup_optimizer(self):
        """Setup optimizer and scheduler with optimizations."""
        print("\nSetting up optimizer...")

        # Get optimizer config
        optimizer_type = self.training_config.get('optimizer', 'adam').lower()
        learning_rate = self.training_config.get('learning_rate', 0.0001)
        weight_decay = self.training_config.get('weight_decay', 0.00001)

        # Create optimizer based on config
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config.get('training.sgd_momentum', 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler from config
        scheduler_type = self.training_config.get('scheduler', 'step').lower()
        num_epochs = self.training_config.get('num_epochs', 30)

        # Warmup epochs from config
        self.warmup_epochs = self.config.get('training.warmup_epochs', 0)

        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - self.warmup_epochs,
                eta_min=self.config.get('training.min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            lr_step_size = self.training_config.get('lr_step_size', 30)
            lr_gamma = self.training_config.get('lr_gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_step_size,
                gamma=lr_gamma
            )
        elif scheduler_type == 'exponential':
            lr_gamma = self.training_config.get('lr_gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=lr_gamma
            )
        else:
            # No scheduler
            self.scheduler = None

        # Mixed precision scaler (only if device supports it)
        if self.use_amp and self.device_manager.supports_amp:
            self.scaler = GradScaler()
            print("  Using Automatic Mixed Precision (AMP)")
        else:
            self.scaler = None

        print(f"  Optimizer: {optimizer_type}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Warmup epochs: {self.warmup_epochs}")
        print(f"  Scheduler: {scheduler_type if self.scheduler else 'None'}")

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch with device optimizations."""
        self.model.train()

        epoch_losses = []
        epoch_metrics = {
            'loss_contrastive': [],
            'loss_koleo': [],
            'loss_variance': []
        }

        # Timing
        batch_times = []
        data_times = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")

        # Start timing
        end = time.time()

        # Get gradient clipping value from config
        max_grad_norm = self.config.get('training.max_grad_norm', 1.0)

        for batch_idx, batch in enumerate(pbar):
            # Measure data loading time
            data_time = time.time() - end
            data_times.append(data_time)

            # Get positive pairs
            if len(batch) == 2:
                seg1, seg2 = batch
            else:
                seg1, seg2 = batch[0], batch[1]

            # Move to device with optimization for CUDA
            if self.device_manager.is_cuda:
                seg1 = seg1.to(self.device, non_blocking=True)
                seg2 = seg2.to(self.device, non_blocking=True)
            else:
                seg1 = seg1.to(self.device)
                seg2 = seg2.to(self.device)

            # Apply augmentations
            seg1_aug, seg2_aug = self.augmentation(seg1, seg2)

            # Mixed precision forward pass (only if device supports it)
            if self.use_amp and self.device_manager.supports_amp:
                with autocast():
                    loss, metrics = self.model(seg1_aug, seg2_aug)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss, metrics = self.model(seg1_aug, seg2_aug)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update momentum networks
                if hasattr(self.model, 'module'):
                    self.model.module.update_momentum_networks()
                else:
                    self.model.update_momentum_networks()

            # Track losses
            epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
            epoch_metrics['loss_contrastive'].append(metrics['loss_contrastive'])
            epoch_metrics['loss_koleo'].append(metrics.get('loss_koleo', 0))
            epoch_metrics['loss_variance'].append(metrics.get('loss_variance', 0))

            # Measure batch time
            batch_time = time.time() - end
            batch_times.append(batch_time)
            end = time.time()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'cont': f"{metrics['loss_contrastive']:.4f}",
                'koleo': f"{metrics.get('loss_koleo', 0):.4f}",
                'data_t': f"{np.mean(data_times[-10:]) if data_times else 0:.3f}",
                'batch_t': f"{np.mean(batch_times[-10:]) if batch_times else 0:.3f}"
            })

            # Optional: Synchronize for accurate timing (CUDA only)
            sync_frequency = self.config.get('training.cuda_sync_frequency', 100)
            if self.device_manager.is_cuda and batch_idx % sync_frequency == 0:
                self.device_manager.synchronize()

        # Clean up GPU memory after epoch
        if self.device_manager.is_cuda:
            self.device_manager.empty_cache()

        # Compute epoch statistics
        stats = {
            'loss': np.mean(epoch_losses),
            'loss_contrastive': np.mean(epoch_metrics['loss_contrastive']),
            'loss_koleo': np.mean(epoch_metrics['loss_koleo']),
            'loss_variance': np.mean(epoch_metrics['loss_variance']),
            'lr': self.optimizer.param_groups[0]['lr'],
            'avg_data_time': np.mean(data_times) if data_times else 0,
            'avg_batch_time': np.mean(batch_times) if batch_times else 0
        }

        # Add memory stats if CUDA
        if self.device_manager.is_cuda:
            mem_stats = self.device_manager.memory_stats()
            stats['gpu_memory_gb'] = mem_stats.get('allocated', 0)

        return stats

    def validate(self, epoch: int) -> Dict:
        """Validate model with device optimizations."""
        self.model.eval()

        val_losses = []
        val_metrics = {
            'loss_contrastive': [],
            'loss_koleo': [],
            'loss_variance': []
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")

            for batch in pbar:
                # Get positive pairs
                if len(batch) == 2:
                    seg1, seg2 = batch
                else:
                    seg1, seg2 = batch[0], batch[1]

                # Move to device with optimization
                if self.device_manager.is_cuda:
                    seg1 = seg1.to(self.device, non_blocking=True)
                    seg2 = seg2.to(self.device, non_blocking=True)
                else:
                    seg1 = seg1.to(self.device)
                    seg2 = seg2.to(self.device)

                # No augmentation for validation
                if self.use_amp and self.device_manager.supports_amp:
                    with autocast():
                        loss, metrics = self.model(seg1, seg2)
                else:
                    loss, metrics = self.model(seg1, seg2)

                val_losses.append(loss.item())
                val_metrics['loss_contrastive'].append(metrics['loss_contrastive'])
                val_metrics['loss_koleo'].append(metrics.get('loss_koleo', 0))
                val_metrics['loss_variance'].append(metrics.get('loss_variance', 0))

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Compute validation statistics
        stats = {
            'loss': np.mean(val_losses),
            'loss_contrastive': np.mean(val_metrics['loss_contrastive']),
            'loss_koleo': np.mean(val_metrics['loss_koleo']),
            'loss_variance': np.mean(val_metrics['loss_variance'])
        }

        return stats

    def apply_warmup(self, epoch: int):
        """Apply learning rate warmup."""
        if epoch < self.warmup_epochs:
            base_lr = self.training_config.get('learning_rate', 0.0001)
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale

    def train(
            self,
            modality: str = 'ppg',
            num_epochs: Optional[int] = None,
            resume_from: Optional[str] = None,
            early_stopping_patience: Optional[int] = None
    ):
        """Main training loop with device optimizations."""
        # Setup
        self.setup_data(modality)
        self.setup_model(modality)
        self.setup_optimizer()

        # Resume if specified
        start_epoch = 0
        if resume_from:
            print(f"\nResuming from checkpoint: {resume_from}")
            if hasattr(self.model, 'module'):
                start_epoch, self.best_val_loss = self.model.module.load_checkpoint(
                    resume_from, self.optimizer
                )
            else:
                start_epoch, self.best_val_loss = self.model.load_checkpoint(
                    resume_from, self.optimizer
                )

        # Training parameters from config
        if num_epochs is None:
            num_epochs = self.training_config.get('num_epochs', 30)

        if early_stopping_patience is None:
            self.early_stopping_patience = self.config.get('early_stopping_patience', 30)

        print(f"\nStarting training:")
        print(f"  Modality: {modality.upper()}")
        print(f"  Device: {self.device_manager.type}")
        print(f"  Epochs: {start_epoch} -> {num_epochs}")
        print(f"  Batch size: {self.training_config.get('batch_size', 64)}")
        print(
            f"  Effective batch size: {self.training_config.get('batch_size', 64) * self.gradient_accumulation_steps}")
        print(f"  Early stopping patience: {self.early_stopping_patience}")

        # Show initial memory stats
        if self.device_manager.is_cuda:
            mem_stats = self.device_manager.memory_stats()
            print(
                f"  Initial GPU memory: {mem_stats.get('allocated', 0):.2f}/{mem_stats.get('reserved', 0):.2f} GB (allocated/reserved)")

        # Early stopping
        patience_counter = 0

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            # Apply warmup
            self.apply_warmup(epoch)

            # Train
            train_start = time.time()
            train_stats = self.train_epoch(epoch)
            train_time = time.time() - train_start
            self.train_history.append(train_stats)

            # Validate
            val_start = time.time()
            val_stats = self.validate(epoch)
            val_time = time.time() - val_start
            self.val_history.append(val_stats)

            # Update scheduler (after warmup)
            if epoch >= self.warmup_epochs and self.scheduler:
                self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_stats['loss']:.4f}")
            print(f"    - Contrastive: {train_stats['loss_contrastive']:.4f}")
            print(f"    - KoLeo: {train_stats['loss_koleo']:.4f}")
            print(f"    - Variance: {train_stats['loss_variance']:.4f}")
            print(f"  Val Loss: {val_stats['loss']:.4f}")
            print(f"    - Contrastive: {val_stats['loss_contrastive']:.4f}")
            print(f"    - KoLeo: {val_stats['loss_koleo']:.4f}")
            print(f"    - Variance: {val_stats['loss_variance']:.4f}")
            print(f"  Learning Rate: {train_stats['lr']:.6f}")
            print(f"  Timing: Train {train_time:.1f}s, Val {val_time:.1f}s")
            print(f"  Avg batch time: {train_stats['avg_batch_time']:.3f}s")

            if self.device_manager.is_cuda:
                print(f"  GPU Memory: {train_stats.get('gpu_memory_gb', 0):.2f} GB")

            # Save checkpoints based on config frequency
            save_freq = self.training_config.get('save_freq', 10)
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                if hasattr(self.model, 'module'):
                    self.model.module.save_checkpoint(
                        str(checkpoint_path),
                        self.optimizer,
                        epoch + 1,
                        val_stats['loss']
                    )
                else:
                    self.model.save_checkpoint(
                        str(checkpoint_path),
                        self.optimizer,
                        epoch + 1,
                        val_stats['loss']
                    )

            # Save best model and early stopping
            if val_stats['loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['loss']
                patience_counter = 0

                best_path = self.checkpoint_dir / "best_model.pt"
                if hasattr(self.model, 'module'):
                    self.model.module.save_checkpoint(
                        str(best_path),
                        self.optimizer,
                        epoch + 1,
                        self.best_val_loss
                    )
                else:
                    self.model.save_checkpoint(
                        str(best_path),
                        self.optimizer,
                        epoch + 1,
                        self.best_val_loss
                    )
                print(f"  New best model saved (val loss: {self.best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    print(" we o")
                    # break

            # Save training history
            self.save_history()

            # Clean GPU cache periodically based on config
            cache_clean_frequency = self.config.get('cache_clean_frequency', 10)
            if self.device_manager.is_cuda and (epoch + 1) % cache_clean_frequency == 0:
                self.device_manager.empty_cache()
                print("  Cleared GPU cache")

        print("\n" + "=" * 50)
        print("Training completed!")
        print("=" * 50)

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        if hasattr(self.model, 'module'):
            self.model.module.save_checkpoint(
                str(final_path),
                self.optimizer,
                epoch + 1,
                val_stats['loss']
            )
            # Save encoder only for downstream tasks
            encoder_path = self.checkpoint_dir / "encoder.pt"
            torch.save(self.model.module.encoder.state_dict(), encoder_path)
        else:
            self.model.save_checkpoint(
                str(final_path),
                self.optimizer,
                epoch + 1,
                val_stats['loss']
            )
            # Save encoder only for downstream tasks
            encoder_path = self.checkpoint_dir / "encoder.pt"
            torch.save(self.model.encoder.state_dict(), encoder_path)

        print(f"Encoder saved to {encoder_path}")

        # Final cleanup
        if self.device_manager.is_cuda:
            self.device_manager.empty_cache()
            final_mem = self.device_manager.memory_stats()
            print(f"Final GPU memory: {final_mem.get('allocated', 0):.2f} GB")

        # Print training summary
        self.print_training_summary()

        return self.checkpoint_dir

    def print_training_summary(self):
        """Print comprehensive training summary."""
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)

        if self.train_history:
            # Best metrics
            best_train_loss = min([h['loss'] for h in self.train_history])
            best_val_loss = min([h['loss'] for h in self.val_history])

            print(f"Best train loss: {best_train_loss:.4f}")
            print(f"Best val loss: {best_val_loss:.4f}")

            # Final metrics
            final_train = self.train_history[-1]
            final_val = self.val_history[-1]

            print(f"\nFinal metrics:")
            print(f"  Train loss: {final_train['loss']:.4f}")
            print(f"  Val loss: {final_val['loss']:.4f}")

            # Device and optimization info
            print(f"\nDevice used: {self.device_manager.type}")
            print(f"Mixed precision: {self.use_amp}")

            # Timing statistics
            if 'avg_batch_time' in final_train:
                total_time = sum([h.get('avg_batch_time', 0) * len(self.train_loader)
                                  for h in self.train_history])
                print(f"\nTotal training time: {total_time / 3600:.2f} hours")
                print(f"Avg epoch time: {total_time / len(self.train_history) / 60:.2f} minutes")

            # Memory statistics (CUDA only)
            if self.device_manager.is_cuda and 'gpu_memory_gb' in final_train:
                max_memory = max([h.get('gpu_memory_gb', 0) for h in self.train_history])
                print(f"\nPeak GPU memory: {max_memory:.2f} GB")

    def save_history(self):
        """Save training history."""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'config': self.config.config,  # Save full config
            'experiment_name': self.experiment_name,
            'best_val_loss': self.best_val_loss,
            'modality': getattr(self, 'modality', 'unknown'),
            'device': self.device_manager.type,
            'device_properties': self.device_manager.get_properties(),
            'ssl_method': self.ssl_method
        }

        # Save as JSON
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        # Save as CSV for easy plotting
        if self.train_history:
            train_df = pd.DataFrame(self.train_history)
            val_df = pd.DataFrame(self.val_history)

            train_df.to_csv(self.log_dir / "train_history.csv", index=False)
            val_df.to_csv(self.log_dir / "val_history.csv", index=False)


# ============= TEST FUNCTIONS =============

def test_training():
    """Test training functionality with device manager."""
    print("=" * 50)
    print("Testing Training Pipeline with Device Manager")
    print("=" * 50)

    # Get device manager
    device_manager = get_device_manager()
    print(f"\nTesting on device: {device_manager.type}")

    # Load config
    config = get_config()

    # Override some config values for testing
    test_overrides = {
        'training': {
            'batch_size': 4,
            'num_epochs': 2,
            'save_freq': 1
        }
    }
    config.update_for_testing(test_overrides)

    print("\n1. Testing Trainer initialization with device manager:")
    trainer = Trainer(
        config_path='configs/config.yaml',
        experiment_name='test_run',
        device_manager=device_manager
    )

    print(f" Early stopping  {Trainer.early_stopping_patience} ")
    print("   ✓ Trainer initialized with device manager")

    # Test setup for different modalities
    for modality in ['ppg', 'ecg', 'acc']:
        print(f"\n{'=' * 40}")
        print(f"Testing {modality.upper()}")
        print(f"{'=' * 40}")

        print(f"\n2. Testing {modality.upper()} data setup:")
        trainer.setup_data(modality=modality)
        print(f"   ✓ {modality.upper()} data loaders created")

        print(f"\n3. Testing {modality.upper()} model setup:")
        trainer.setup_model(modality=modality)
        print(f"   ✓ {modality.upper()} model created on {device_manager.type}")

        print(f"\n4. Testing optimizer setup:")
        trainer.setup_optimizer()
        print("   ✓ Optimizer created")

        # Show memory stats if CUDA
        if device_manager.is_cuda:
            mem_stats = device_manager.memory_stats()
            print(f"   GPU memory: {mem_stats.get('allocated', 0):.2f} GB")

        print(f"\n5. Testing single {modality.upper()} training step:")
        # Get one batch
        for batch in trainer.train_loader:
            if len(batch) == 2:
                seg1, seg2 = batch
            else:
                seg1, seg2 = batch[0], batch[1]

            seg1 = seg1.to(trainer.device)
            seg2 = seg2.to(trainer.device)

            # Check shapes based on modality
            if modality == 'acc':
                expected_channels = config.get('dataset.acc.channels', 3)
                assert seg1.shape[
                           1] == expected_channels, f"ACC should have {expected_channels} channels, got {seg1.shape[1]}"
            else:
                assert seg1.shape[1] == 1, f"{modality.upper()} should have 1 channel, got {seg1.shape[1]}"

            # Forward pass
            loss, metrics = trainer.model(seg1, seg2)

            print(f"   Loss: {loss:.4f}")
            print(f"   Metrics: {list(metrics.keys())}")
            assert loss > 0, "Loss should be positive"
            print(f"   ✓ {modality.upper()} training step successful on {device_manager.type}")
            break

        # Clean GPU memory between modalities
        if device_manager.is_cuda:
            device_manager.empty_cache()

    # Clean up test files
    import shutil
    if trainer.checkpoint_dir.exists():
        shutil.rmtree(trainer.checkpoint_dir.parent)
    if trainer.log_dir.exists():
        shutil.rmtree(trainer.log_dir.parent)

    print("\n" + "=" * 50)
    print(f"All training tests passed successfully on {device_manager.type}!")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_training()
    else:
        # Main training
        config = get_config()
        modality = sys.argv[1] if len(sys.argv) > 1 else 'ppg'
        num_epochs = config.get('training.num_epochs', 30)

        # Get device manager from command line or auto-detect
        device_manager = get_device_manager()

        trainer = Trainer(device_manager=device_manager)
        trainer.train(modality=modality, num_epochs=num_epochs)