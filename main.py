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


def setup_config(args, device_manager):
    """Setup and update configuration based on arguments and device."""
    # Load base configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update device configuration
    config['device'] = {
        'backend': device_manager.type,
        'device_object': str(device_manager.device)
    }

    # Optimize batch size based on device if not specified
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    else:
        # Use device-optimized batch size
        optimal_batch = device_manager.get_optimal_batch_size(args.modality)
        config['training']['batch_size'] = optimal_batch
        logger.info(f"Auto-selected batch_size: {optimal_batch} for {device_manager.type}")

    # Optimize number of workers
    if args.workers is not None:
        config['training']['num_workers'] = args.workers
    else:
        config['training']['num_workers'] = device_manager.get_num_workers()
        logger.info(f"Auto-selected num_workers: {config['training']['num_workers']}")

    # Set AMP based on device support
    config['training']['use_amp'] = device_manager.supports_amp
    if not device_manager.supports_amp and config['training'].get('use_amp', False):
        logger.warning(f"AMP not supported on {device_manager.type}, disabling")

    # Override with other command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        logger.info(f"Overriding num_epochs to {args.epochs}")

    if args.lr:
        config['training']['learning_rate'] = args.lr
        logger.info(f"Overriding learning_rate to {args.lr}")

    # Save updated config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    updated_config_path = Path(f"configs/config_{args.modality}_{timestamp}.yaml")
    updated_config_path.parent.mkdir(exist_ok=True)

    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f)

    return str(updated_config_path), config


def run_pretrain(args, device_manager):
    """Pre-train on VitalDB dataset."""
    logger.info(f"Starting pre-training on VitalDB for {args.modality.upper()}")

    config = get_config()
    pretrain_config = config.config.get('pretrain', {})

    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"vitaldb_{args.modality}_{timestamp}"

    # Initialize trainer with pretrain phase
    trainer = Trainer(
        config_path=args.config,
        experiment_name=experiment_name,
        device_manager=device_manager,
        ssl_method=args.ssl_method,
        phase='pretrain'  # This tells trainer to use VitalDB
    )

    print(f"\n🚀 Pre-training Configuration:")
    print(f"  Dataset: VitalDB")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {args.batch_size or pretrain_config.get('batch_size', 128)}")
    print(f"  Learning rate: {args.lr or pretrain_config.get('learning_rate', 0.0001)}")
    print(f"  Epochs: {args.epochs or pretrain_config.get('epochs', 50)}")
    print(f"  SSL Method: {args.ssl_method}")

    # Train - this will call setup_data() internally with correct dataset
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=args.epochs or pretrain_config.get('epochs', 50),
        early_stopping_patience=args.patience or pretrain_config.get('early_stopping_patience', 15)
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

    # Initialize trainer with finetune phase
    trainer = Trainer(
        config_path=args.config,
        experiment_name=experiment_name,
        device_manager=device_manager,
        ssl_method=args.ssl_method,
        phase='finetune'  # This tells trainer to use BUT PPG
    )

    # Store pretrained path for trainer to load
    trainer.pretrained_path = args.pretrained_path

    print(f"\n🚀 Fine-tuning Configuration:")
    print(f"  Dataset: BUT PPG")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {args.batch_size or finetune_config.get('batch_size', 64)}")
    print(f"  Learning rate: {args.lr or finetune_config.get('learning_rate', 0.00001)}")
    print(f"  Epochs: {args.epochs or finetune_config.get('epochs', 20)}")
    print(f"  Pretrained model: {args.pretrained_path}")

    # Train
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=args.epochs or finetune_config.get('epochs', 20),
        early_stopping_patience=args.patience or finetune_config.get('early_stopping_patience', 10)
    )

    logger.info(f"Fine-tuning completed! Model saved to: {checkpoint_dir}")
    return checkpoint_dir


def run_evaluate(args, device_manager):
    """Run evaluation on a trained model."""
    logger.info(f"Evaluating {args.modality.upper()} model")

    # Find checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    encoder_path = checkpoint_dir / "encoder.pt"
    if not encoder_path.exists():
        encoder_path = checkpoint_dir / "best_model.pt"

    if not encoder_path.exists():
        logger.error(f"No model found in {checkpoint_dir}")
        sys.exit(1)

    # Get data directory
    data_dir = args.data_dir or 'data/but_ppg/dataset'

    # Create evaluator
    evaluator = DownstreamEvaluator(
        encoder_path=str(encoder_path),
        config_path=args.config,
        device_manager=device_manager
    )

    # Create test dataset
    test_dataset = BUTPPGDataset(
        data_dir=data_dir,
        modality=args.modality,
        split='test',
        return_participant_id=True,
        return_labels=True,
        quality_filter=False,
        downsample=args.downsample
    )

    # Run evaluation
    results_path = checkpoint_dir / f"downstream_results_{args.modality}.csv"
    results_df = evaluator.evaluate_all_tasks(
        test_dataset,
        save_path=str(results_path)
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - {args.modality.upper()}")
    print("=" * 60)
    print(results_df.to_string())

    device_manager.empty_cache()
    logger.info(f"Results saved to: {results_path}")
    return results_path


def run_compare(args, device_manager):
    """Compare results with paper benchmarks."""
    logger.info("Comparing with paper benchmarks")

    results_path = Path(args.results_path)

    comparator = ResultsComparator(config_path=args.config)
    comparator.load_results(str(results_path), modality=args.modality)

    comparison_dir = results_path.parent / 'comparisons'
    report = comparator.generate_report(save_dir=str(comparison_dir))
    comparator.print_summary()

    plot_path = comparison_dir / f'comparison_{args.modality}.png'
    comparator.plot_comparison(save_path=str(plot_path))

    logger.info(f"Comparison saved to: {comparison_dir}")
    device_manager.empty_cache()
    return report


def run_train(args, device_manager):
    """Run standard SSL training on BUT PPG dataset."""
    logger.info(f"Starting SSL training for {args.modality.upper()} on {device_manager.type}")

    config_path, config = setup_config(args, device_manager)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.modality}_{args.name}_{timestamp}"

    trainer = Trainer(
        config_path=config_path,
        experiment_name=experiment_name,
        device_manager=device_manager,
        downsample=args.downsample,
        ssl_method=args.ssl_method,
        phase=None  # No phase for standard training
    )

    print(f"\n🚀 Training Configuration:")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Workers: {config['training']['num_workers']}")
    print(f"  SSL Method: {args.ssl_method}")

    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=config['training']['num_epochs'],
        resume_from=args.resume,
        early_stopping_patience=args.patience
    )

    device_manager.empty_cache()
    logger.info(f"Training completed! Checkpoints saved to: {checkpoint_dir}")
    return checkpoint_dir


def run_full(args, device_manager):
    """Run complete pipeline: pretrain -> finetune -> evaluate -> compare."""
    logger.info("Running complete pipeline")

    # Step 1: Pre-train (or skip if pretrained path provided)
    if args.pretrained_path:
        logger.info("Using provided pretrained model, skipping pre-training")
        pretrain_dir = Path(args.pretrained_path).parent
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: PRE-TRAINING ON VITALDB")
        logger.info("=" * 60)
        pretrain_dir = run_pretrain(args, device_manager)
        args.pretrained_path = str(pretrain_dir / "encoder.pt")

    # Step 2: Fine-tune
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: FINE-TUNING ON BUT PPG")
    logger.info("=" * 60)
    finetune_dir = run_finetune(args, device_manager)

    # Step 3: Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 60)
    args.checkpoint_dir = str(finetune_dir)
    results_path = run_evaluate(args, device_manager)

    # Step 4: Compare with paper benchmarks
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: COMPARISON WITH BENCHMARKS")
    logger.info("=" * 60)
    args.results_path = str(results_path)
    run_compare(args, device_manager)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    device_manager.empty_cache()


def run_test(args):
    """Run tests for specified modules."""
    if args.module in ['all', 'data']:
        from data import test_data_loading
        test_data_loading()

    if args.module in ['all', 'device']:
        device_manager = get_device_manager()
        print(f"Device test: {device_manager.device}")

    if args.module in ['all', 'model']:
        # Add model tests here
        pass

    print("Tests completed!")


def main():
    # Create main parser with shared arguments
    parser = argparse.ArgumentParser(
        description='Biosignal Foundation Model Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === GLOBAL ARGUMENTS (available for all commands) ===
    parser.add_argument('command',
                        choices=['pretrain', 'finetune', 'evaluate', 'compare',
                                 'train', 'full', 'test'],
                        help='Command to run')

    parser.add_argument('--modality', type=str,
                        choices=['ppg', 'ecg', 'acc'],
                        default='ppg',
                        help='Signal modality')

    parser.add_argument('--config', type=str,
                        default='configs/config.yaml',
                        help='Configuration file')

    parser.add_argument('--device', type=str,
                        choices=['cuda', 'mps', 'cpu', 'auto'],
                        default='auto',
                        help='Device to use')

    parser.add_argument('--device-id', type=int,
                        default=0,
                        help='CUDA device ID for multi-GPU systems')

    # === TRAINING ARGUMENTS (shared by pretrain, finetune, train, full) ===
    parser.add_argument('--ssl-method', type=str,
                        default='simsiam',
                        choices=['infonce', 'simsiam'],
                        help='SSL method to use')

    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')

    parser.add_argument('--batch-size', type=int,
                        help='Batch size')

    parser.add_argument('--lr', type=float,
                        help='Learning rate')

    parser.add_argument('--workers', type=int,
                        help='Number of data workers')

    parser.add_argument('--patience', type=int,
                        default=10,
                        help='Early stopping patience')

    parser.add_argument('--downsample', action='store_true',
                        help='Use downsampled segments')

    parser.add_argument('--name', type=str,
                        default='experiment',
                        help='Experiment name')

    parser.add_argument('--resume', type=str,
                        help='Resume from checkpoint')

    # === COMMAND-SPECIFIC ARGUMENTS ===
    # For finetune and full commands
    parser.add_argument('--pretrained-path', type=str,
                        help='Path to pretrained encoder (for finetune/full)')

    # For evaluate command
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Checkpoint directory (for evaluate)')

    # For compare command
    parser.add_argument('--results-path', type=str,
                        help='Path to evaluation results (for compare)')

    # For test command
    parser.add_argument('--module', type=str,
                        choices=['all', 'data', 'model', 'augment', 'ssl', 'train', 'device'],
                        default='all',
                        help='Module to test (for test command)')

    # Additional optional arguments
    parser.add_argument('--data-dir', type=str,
                        help='Data directory path')

    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training phase in full pipeline')

    # Parse arguments
    args = parser.parse_args()

    # Validate command-specific required arguments
    if args.command == 'finetune' and not args.pretrained_path:
        parser.error("finetune requires --pretrained-path")

    if args.command == 'evaluate' and not args.checkpoint_dir:
        parser.error("evaluate requires --checkpoint-dir")

    if args.command == 'compare' and not args.results_path:
        parser.error("compare requires --results-path")

    # Initialize device manager
    device_manager = get_device_manager()
    if args.device != 'auto':
        device_manager.set_device(args.device, args.device_id)

    # Check environment
    check_environment()

    # Route to appropriate function based on command
    command_map = {
        'pretrain': run_pretrain,
        'finetune': run_finetune,
        'evaluate': run_evaluate,
        'compare': run_compare,
        'train': run_train,
        'full': run_full,
        'test': lambda args, dm: run_test(args)
    }

    # Execute the command
    if args.command in command_map:
        command_map[args.command](args, device_manager)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()