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
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size
    else:
        # Use device-optimized batch size
        optimal_batch = device_manager.get_optimal_batch_size(args.modality)
        config['training']['batch_size'] = optimal_batch
        logger.info(f"Auto-selected batch_size: {optimal_batch} for {device_manager.type}")

    # Optimize number of workers
    if hasattr(args, 'workers') and args.workers is not None:
        config['training']['num_workers'] = args.workers
    else:
        config['training']['num_workers'] = device_manager.get_num_workers()
        logger.info(f"Auto-selected num_workers: {config['training']['num_workers']}")

    # Set AMP based on device support
    config['training']['use_amp'] = device_manager.supports_amp
    if not device_manager.supports_amp and config['training'].get('use_amp', False):
        logger.warning(f"AMP not supported on {device_manager.type}, disabling")

    # Override with other command line arguments
    if hasattr(args, 'epochs') and args.epochs:
        config['training']['num_epochs'] = args.epochs
        logger.info(f"Overriding num_epochs to {args.epochs}")

    if hasattr(args, 'lr') and args.lr:
        config['training']['learning_rate'] = args.lr
        logger.info(f"Overriding learning_rate to {args.lr}")

    # Save updated config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    updated_config_path = Path(f"configs/config_{args.modality}_{timestamp}.yaml")
    updated_config_path.parent.mkdir(exist_ok=True)

    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f)

    return str(updated_config_path), config


# FIXED run_pretrain function for main.py
# Replace the existing run_pretrain function with this corrected version

def run_pretrain(args, device_manager):
    """Pre-train on VitalDB dataset - FIXED VERSION."""
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

    # DON'T manually override data loaders!
    # The trainer.setup_data() method will automatically use VitalDB
    # when phase='pretrain' is set

    print(f"\n🚀 Pre-training Configuration:")
    print(f"  Dataset: VitalDB")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {pretrain_config.get('batch_size', 128)}")
    print(f"  Learning rate: {pretrain_config.get('learning_rate', 0.0001)}")
    print(f"  Epochs: {pretrain_config.get('epochs', 50)}")
    print(f"  SSL Method: {args.ssl_method}")

    # Train - this will call setup_data() internally with correct dataset
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=pretrain_config.get('epochs', 50),
        early_stopping_patience=pretrain_config.get('early_stopping_patience', 15)
    )

    logger.info(f"Pre-training completed! Encoder saved to: {checkpoint_dir}/encoder.pt")
    return checkpoint_dir


def run_finetune(args, device_manager):
    """Fine-tune on BUT PPG dataset - FIXED VERSION."""
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

    # Load pretrained encoder if provided
    if args.pretrained_path:
        logger.info(f"Loading pretrained encoder from: {args.pretrained_path}")

        # Load the pretrained weights into the model
        # This should be done after setup_model is called in train()
        # So we'll pass it as a parameter
        trainer.pretrained_path = args.pretrained_path

    print(f"\n🚀 Fine-tuning Configuration:")
    print(f"  Dataset: BUT PPG")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {finetune_config.get('batch_size', 64)}")
    print(f"  Learning rate: {finetune_config.get('learning_rate', 0.00001)}")
    print(f"  Epochs: {finetune_config.get('epochs', 20)}")
    print(f"  Pretrained model: {args.pretrained_path}")

    # Train
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=finetune_config.get('epochs', 20),
        early_stopping_patience=finetune_config.get('early_stopping_patience', 10)
    )

    logger.info(f"Fine-tuning completed! Model saved to: {checkpoint_dir}")
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
        phase='finetune'  # Pass phase
    )

    # Setup with BUT PPG data
    trainer.setup_data(modality=args.modality)
    trainer.setup_model(modality=args.modality)

    # Load pretrained encoder
    encoder_state = torch.load(args.pretrained_path)
    trainer.model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"✓ Loaded pretrained encoder from {args.pretrained_path}")

    # Optionally freeze encoder
    if finetune_config.get('freeze_encoder', False):
        for param in trainer.model.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder layers frozen")

    # Setup optimizer with lower learning rate
    trainer.setup_optimizer()
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = finetune_config.get('learning_rate', 0.00001)

    # Train
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=finetune_config.get('epochs', 20)
    )

    logger.info(f"Fine-tuning completed! Model saved to: {checkpoint_dir}")
    return checkpoint_dir


def run_evaluation(args, device_manager=None):
    """Run evaluation - can be called from phase or original evaluate command."""
    if device_manager is None:
        device_manager = get_device_manager()

    logger.info(f"Evaluating {args.modality.upper()} model")

    # Find checkpoint
    if hasattr(args, 'checkpoint_dir'):
        checkpoint_dir = Path(args.checkpoint_dir)
        encoder_path = checkpoint_dir / "encoder.pt"
        if not encoder_path.exists():
            encoder_path = checkpoint_dir / "best_model.pt"
    else:
        logger.error("No checkpoint directory specified")
        sys.exit(1)

    if not encoder_path.exists():
        logger.error(f"No model found in {checkpoint_dir}")
        sys.exit(1)

    # Get data directory
    data_dir = getattr(args, 'data_dir', 'data/but_ppg/dataset')

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
        downsample=getattr(args, 'downsample', False)
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


def run_full_pipeline(args, device_manager):
    """Run complete pipeline based on phase or original full command."""

    # Handle phase-based full pipeline
    if args.phase == 'full':
        logger.info("Running complete pipeline: pretrain -> finetune -> evaluate")

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
        eval_args = argparse.Namespace(
            checkpoint_dir=str(finetune_dir),
            modality=args.modality,
            config=args.config,
            downsample=getattr(args, 'downsample', False)
        )
        results_path = run_evaluation(eval_args, device_manager)

    else:
        # Original full pipeline (train -> evaluate -> compare)
        logger.info(f"Running original full pipeline for {args.modality.upper()}")

        # Step 1: Train
        if args.skip_training and args.checkpoint_dir:
            logger.info("Skipping training, using existing checkpoint")
            checkpoint_dir = Path(args.checkpoint_dir)
        else:
            checkpoint_dir = train_model(args, device_manager)

        # Step 2: Evaluate
        eval_args = argparse.Namespace(
            checkpoint_dir=str(checkpoint_dir),
            modality=args.modality,
            config=args.config,
            data_dir=getattr(args, 'data_dir', 'data/but_ppg/dataset')
        )
        results_path = run_evaluation(eval_args, device_manager)

        # Step 3: Compare
        compare_args = argparse.Namespace(
            results_path=str(results_path),
            modality=args.modality,
            config=args.config
        )
        compare_with_paper(compare_args, device_manager, results_path)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    device_manager.empty_cache()


def train_model(args, device_manager):
    """Original train function for backward compatibility."""
    logger.info(f"Starting SSL training for {args.modality.upper()} on {device_manager.type}")

    config_path, config = setup_config(args, device_manager)
    ssl_method = getattr(args, 'ssl_method', 'simsiam')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.modality}_{args.name}_{timestamp}"

    trainer = Trainer(
        config_path=config_path,
        experiment_name=experiment_name,
        device_manager=device_manager,
        downsample=getattr(args, 'downsample', False),
        ssl_method=ssl_method,
        phase=None  # No phase for original training
    )

    print(f"\n🚀 Training Configuration:")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Workers: {config['training']['num_workers']}")

    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=config['training']['num_epochs'],
        resume_from=getattr(args, 'resume', None),
        early_stopping_patience=getattr(args, 'patience', 10)
    )

    device_manager.empty_cache()
    logger.info(f"Training completed! Checkpoints saved to: {checkpoint_dir}")
    return checkpoint_dir


def compare_with_paper(args, device_manager, results_path):
    """Compare results with paper benchmarks."""
    logger.info("Comparing with paper benchmarks")

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


def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Biosignal Foundation Model Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add device arguments
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'],
                        default='auto', help='Device to use')
    parser.add_argument('--device-id', type=int, default=0,
                        help='CUDA device ID for multi-GPU systems')

    # Create subparsers for both phase-based and command-based approaches
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # === PHASE-BASED COMMANDS (NEW) ===
    phase_parser = subparsers.add_parser('phase', help='Run phase-based pipeline')
    phase_parser.add_argument('phase', choices=['pretrain', 'finetune', 'evaluate', 'full'],
                              help='Pipeline phase: pretrain (VitalDB), finetune (BUT PPG), evaluate, or full')
    phase_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                              default='ppg', help='Signal modality')
    phase_parser.add_argument('--config', type=str, default='configs/config.yaml',
                              help='Configuration file')
    phase_parser.add_argument('--ssl-method', type=str, default='simsiam',
                              choices=['infonce', 'simsiam'],
                              help='SSL method')
    phase_parser.add_argument('--pretrained-path', type=str,
                              help='Path to pretrained encoder (for finetune/full)')
    phase_parser.add_argument('--checkpoint-dir', type=str,
                              help='Checkpoint directory (for evaluate)')
    phase_parser.add_argument('--downsample', action='store_true',
                              help='Use downsampled segments')

    # === ORIGINAL COMMANDS (BACKWARD COMPATIBLE) ===
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SSL model')
    train_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                              default='ppg', help='Signal modality')
    train_parser.add_argument('--name', type=str, default='ssl',
                              help='Experiment name')
    train_parser.add_argument('--config', type=str, default='configs/config.yaml',
                              help='Configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--workers', type=int, help='Number of workers')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--patience', type=int, default=10,
                              help='Early stopping patience')
    train_parser.add_argument('--ssl-method', type=str, default='simsiam',
                              choices=['infonce', 'simsiam'],
                              help='SSL method')
    train_parser.add_argument('--downsample', action='store_true',
                              help='Use downsampled segments')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--checkpoint-dir', type=str, required=True,
                             help='Path to checkpoint directory')
    eval_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                             default='ppg', help='Signal modality')
    eval_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Configuration file')
    eval_parser.add_argument('--downsample', action='store_true',
                             help='Use downsampled segments')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare with paper benchmarks')
    compare_parser.add_argument('--results-path', type=str, required=True,
                                help='Path to evaluation results')
    compare_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                                default='ppg', help='Signal modality')
    compare_parser.add_argument('--config', type=str, default='configs/config.yaml',
                                help='Configuration file')

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                             default='ppg', help='Signal modality')
    full_parser.add_argument('--name', type=str, default='full',
                             help='Experiment name')
    full_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Configuration file')
    full_parser.add_argument('--epochs', type=int, help='Number of epochs')
    full_parser.add_argument('--batch-size', type=int, help='Batch size')
    full_parser.add_argument('--skip-training', action='store_true',
                             help='Skip training, use existing checkpoint')
    full_parser.add_argument('--checkpoint-dir', type=str,
                             help='Existing checkpoint directory')
    full_parser.add_argument('--ssl-method', type=str, default='simsiam',
                             choices=['infonce', 'simsiam'],
                             help='SSL method')
    full_parser.add_argument('--downsample', action='store_true',
                             help='Use downsampled segments')
    full_parser.add_argument('--patience', type=int, default=10,
                             help='Early stopping patience')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--module', type=str,
                             choices=['all', 'data', 'model', 'augment', 'ssl', 'train', 'device'],
                             default='all', help='Module to test')

    # Parse arguments
    args = parser.parse_args()

    # Initialize device manager
    device_manager = get_device_manager()
    if args.device != 'auto':
        device_manager.set_device(args.device, args.device_id)

    # Check environment
    check_environment()

    # Route to appropriate function
    if args.command == 'phase':
        # New phase-based approach
        if args.phase == 'pretrain':
            run_pretrain(args, device_manager)
        elif args.phase == 'finetune':
            run_finetune(args, device_manager)
        elif args.phase == 'evaluate':
            run_evaluation(args, device_manager)
        elif args.phase == 'full':
            run_full_pipeline(args, device_manager)

    elif args.command == 'train':
        # Original train command
        train_model(args, device_manager)

    elif args.command == 'evaluate':
        # Original evaluate command
        run_evaluation(args, device_manager)

    elif args.command == 'compare':
        # Original compare command
        compare_with_paper(args, device_manager, Path(args.results_path))

    elif args.command == 'full':
        # Original full pipeline
        run_full_pipeline(args, device_manager)

    elif args.command == 'test':
        # Run tests
        if args.module in ['all', 'data']:
            from data import test_data_loading
            test_data_loading()
        print("Tests completed!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()