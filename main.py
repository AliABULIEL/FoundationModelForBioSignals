#!/usr/bin/env python3
"""
Main script to run the biosignal foundation model pipeline
Supports PPG, ECG, and ACC modalities with pretrain/finetune/evaluate flows
"""

import argparse
import torch
import yaml
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import device manager first
from device import DeviceManager, get_device_manager

# Import other modules
from train import Trainer
from evaluate import DownstreamEvaluator
from compare import ResultsComparator
from data import BUTPPGDataset, create_dataloaders
from config_loader import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check the computing environment and dependencies."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")

    # Initialize device manager
    device_manager = get_device_manager()

    print(f"\n📱 Device Information:")
    print(f"  Selected device: {device_manager.device}")
    print(f"  Device type: {device_manager.type}")

    # Show device properties
    properties = device_manager.get_properties()
    if 'name' in properties:
        print(f"  Device name: {properties['name']}")

    if device_manager.is_cuda:
        print(f"  GPU count: {properties.get('device_count', 1)}")
        print(f"  Memory: {properties.get('memory_total', 0) / 1e9:.2f} GB")
        print(f"  AMP support: {device_manager.supports_amp}")

        # Show memory stats
        mem_stats = device_manager.memory_stats()
        if mem_stats:
            print(f"  Memory free: {mem_stats.get('free', 0):.2f} GB")

    elif device_manager.is_mps:
        print(f"  ✓ Apple Silicon GPU (MPS) active")
        print(f"  AMP support: {device_manager.supports_amp}")

    else:
        print(f"  ⚠ Using CPU (no GPU detected)")
        print(f"  Threads: {properties.get('threads', 'unknown')}")

    # Check data directory
    data_dir = Path("data/but_ppg/dataset")
    if data_dir.exists():
        print(f"\n✓ Data directory found: {data_dir}")
        record_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        print(f"  Found {len(record_dirs)} record directories")
    else:
        print(f"\n✗ Data directory not found: {data_dir}")
        print("  Please download the BUT PPG dataset first!")

    print("=" * 60)
    return device_manager


def run_pretrain(args, device_manager):
    """Pre-train on VitalDB dataset."""
    logger.info(f"Starting pre-training on VitalDB for {args.modality.upper()}")

    config = get_config()
    pretrain_config = config.config.get('pretrain', {})

    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"vitaldb_{args.modality}_{timestamp}"

    # Initialize trainer
    trainer = Trainer(
        config_path=args.config,
        experiment_name=experiment_name,
        device_manager=device_manager,
        ssl_method=args.ssl_method,
        phase='pretrain'  # Important: set phase
    )

    # Create VitalDB dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        modality=args.modality,
        batch_size=pretrain_config.get('batch_size', 128),
        num_workers=pretrain_config.get('num_workers', 8),
        dataset_type='vitaldb',
        config_path=args.config
    )

    # Override dataloaders
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    trainer.test_loader = test_loader

    # Setup model and optimizer
    trainer.setup_model(modality=args.modality)
    trainer.setup_optimizer()

    # Override learning rate from pretrain config
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = pretrain_config.get('learning_rate', 0.0001)

    print(f"\n🚀 Pre-training Configuration:")
    print(f"  Dataset: VitalDB")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {pretrain_config.get('batch_size', 128)}")
    print(f"  Learning rate: {pretrain_config.get('learning_rate', 0.0001)}")
    print(f"  Epochs: {pretrain_config.get('epochs', 50)}")

    # Train
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=pretrain_config.get('epochs', 50)
    )

    logger.info(f"Pre-training completed! Encoder saved to: {checkpoint_dir}/encoder.pt")
    return checkpoint_dir


def run_finetune(args, device_manager):
    """Fine-tune on BUT PPG dataset."""
    if not args.pretrained_path:
        raise ValueError("--pretrained-path required for fine-tuning")

    logger.info(f"Starting fine-tuning on BUT PPG for {args.modality.upper()}")

    config = get_config()
    finetune_config = config.config.get('finetune', {})

    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"finetune_{args.modality}_{timestamp}"

    # Initialize trainer
    trainer = Trainer(
        config_path=args.config,
        experiment_name=experiment_name,
        device_manager=device_manager,
        ssl_method=args.ssl_method,
        phase='finetune'  # Important: set phase
    )

    # Setup with BUT PPG data
    trainer.setup_data(modality=args.modality)
    trainer.setup_model(modality=args.modality)

    # Load pretrained encoder
    encoder_state = torch.load(args.pretrained_path, map_location=trainer.device)
    if hasattr(trainer.model, 'module'):
        trainer.model.module.encoder.load_state_dict(encoder_state, strict=False)
    else:
        trainer.model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"✓ Loaded pretrained encoder from {args.pretrained_path}")

    # Optionally freeze encoder
    if finetune_config.get('freeze_encoder', False):
        if hasattr(trainer.model, 'module'):
            for param in trainer.model.module.encoder.parameters():
                param.requires_grad = False
        else:
            for param in trainer.model.encoder.parameters():
                param.requires_grad = False
        print("✓ Encoder layers frozen")

    # Setup optimizer with lower learning rate
    trainer.setup_optimizer()
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = finetune_config.get('learning_rate', 0.00001)

    print(f"\n🚀 Fine-tuning Configuration:")
    print(f"  Dataset: BUT PPG")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {finetune_config.get('batch_size', 64)}")
    print(f"  Learning rate: {finetune_config.get('learning_rate', 0.00001)}")
    print(f"  Epochs: {finetune_config.get('epochs', 20)}")
    print(f"  Encoder frozen: {finetune_config.get('freeze_encoder', False)}")

    # Train
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=finetune_config.get('epochs', 20)
    )

    logger.info(f"Fine-tuning completed! Model saved to: {checkpoint_dir}")
    return checkpoint_dir


def run_evaluate(args, device_manager):
    """Evaluate model on BUT PPG test set."""
    logger.info(f"Evaluating {args.modality.upper()} model")

    # Find checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_path = checkpoint_dir / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / "encoder.pt"
    else:
        raise ValueError("Either --checkpoint or --checkpoint-dir required for evaluation")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create evaluator
    evaluator = DownstreamEvaluator(
        encoder_path=str(checkpoint_path),
        config_path=args.config,
        device_manager=device_manager
    )

    # Create test dataset
    test_dataset = BUTPPGDataset(
        data_dir=args.data_dir if hasattr(args, 'data_dir') else 'data/but_ppg/dataset',
        modality=args.modality,
        split='test',
        return_participant_id=True,
        return_labels=True,
        quality_filter=False,
        downsample=getattr(args, 'downsample', False)
    )

    # Run evaluation
    results_dir = checkpoint_path.parent
    results_path = results_dir / f"downstream_results_{args.modality}.csv"
    results_df = evaluator.evaluate_all_tasks(
        test_dataset,
        save_path=str(results_path)
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - {args.modality.upper()}")
    print("=" * 60)
    print(results_df.to_string())

    # Compare with paper if requested
    if getattr(args, 'compare', True):
        comparator = ResultsComparator(config_path=args.config)
        comparator.load_results(str(results_path), modality=args.modality)
        comparison_dir = results_dir / 'comparisons'
        comparator.generate_report(save_dir=str(comparison_dir))
        comparator.print_summary()
        comparator.plot_comparison(save_path=str(comparison_dir / f'comparison_{args.modality}.png'))

    logger.info(f"Results saved to: {results_path}")
    return results_path


def run_full_pipeline(args, device_manager):
    """Run complete pipeline: pretrain -> finetune -> evaluate."""
    logger.info(f"Running full pipeline for {args.modality.upper()}")

    # Step 1: Pre-train on VitalDB (or skip if checkpoint provided)
    if getattr(args, 'skip_pretrain', False) and args.pretrained_path:
        logger.info("Skipping pre-training, using provided checkpoint")
        pretrain_dir = Path(args.pretrained_path).parent
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: PRE-TRAINING ON VITALDB")
        logger.info("=" * 60)
        pretrain_dir = run_pretrain(args, device_manager)
        args.pretrained_path = str(pretrain_dir / "encoder.pt")

    # Step 2: Fine-tune on BUT PPG
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: FINE-TUNING ON BUT PPG")
    logger.info("=" * 60)
    finetune_dir = run_finetune(args, device_manager)

    # Step 3: Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 60)
    eval_args = argparse.Namespace(
        checkpoint_dir=str(finetune_dir),
        modality=args.modality,
        config=args.config,
        compare=True,
        downsample=getattr(args, 'downsample', False)
    )
    results_path = run_evaluate(eval_args, device_manager)

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results saved in: {finetune_dir}")

    # Final cleanup
    device_manager.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Biosignal Foundation Model Pipeline'
    )

    # Global arguments
    parser.add_argument('phase', choices=['pretrain', 'finetune', 'evaluate', 'full'],
                        help='Pipeline phase to run')
    parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                        default='ppg', help='Signal modality')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Configuration file')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'],
                        default='auto', help='Device to use')
    parser.add_argument('--device-id', type=int, default=0,
                        help='CUDA device ID')

    # SSL method
    parser.add_argument('--ssl-method', type=str, default='simsiam',
                        choices=['infonce', 'simsiam'],
                        help='SSL method (simsiam recommended for small data)')

    # Phase-specific arguments
    parser.add_argument('--pretrained-path', type=str,
                        help='Path to pretrained encoder (for finetune/full)')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint (for evaluate)')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Checkpoint directory (for evaluate)')

    # Training arguments
    parser.add_argument('--epochs', type=int,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int,
                        help='Override batch size')
    parser.add_argument('--lr', type=float,
                        help='Override learning rate')
    parser.add_argument('--workers', type=int,
                        help='Number of data workers')
    parser.add_argument('--downsample', action='store_true',
                        help='Use downsampled segments for faster training')

    # Pipeline control
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip pre-training in full pipeline')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Compare with paper benchmarks after evaluation')

    args = parser.parse_args()

    # Initialize device manager
    device_manager = get_device_manager()
    if args.device != 'auto':
        device_manager.set_device(args.device, args.device_id)

    # Check environment
    check_environment()

    # Execute phase
    if args.phase == 'pretrain':
        run_pretrain(args, device_manager)

    elif args.phase == 'finetune':
        run_finetune(args, device_manager)

    elif args.phase == 'evaluate':
        run_evaluate(args, device_manager)

    elif args.phase == 'full':
        run_full_pipeline(args, device_manager)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()