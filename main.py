#!/usr/bin/env python3
"""
Main script to run the biosignal foundation model pipeline
Supports PPG, ECG, and ACC modalities with centralized device management
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
from data import BUTPPGDataset

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


def train_model(args, device_manager):
    """Train the SSL model with device manager."""
    logger.info(f"Starting SSL training for {args.modality.upper()} on {device_manager.type}")
    downsample = getattr(args, 'downsample', False)
    print(f"Down sample valuye is {downsample}")

    # Setup configuration
    config_path, config = setup_config(args, device_manager)
    ssl_method = getattr(args, 'ssl_method', 'infonce').replace('-', '_')
    if ssl_method == 'simsiam':
        # Add SimSiam config to the config file
        config['simsiam'] = {
            'projection_dim': getattr(args, 'simsiam_proj_dim', 2048),
            'prediction_dim': getattr(args, 'simsiam_pred_dim', 512)
        }

    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.modality}_{args.name}_{timestamp}"

    # Initialize trainer with device manager
    trainer = Trainer(
        config_path=config_path,
        experiment_name=experiment_name,
        device_manager=device_manager,
        downsample=args.downsample,# Pass device manager
        ssl_method=ssl_method
    )

    # Show optimized settings
    print(f"\n🚀 Training Configuration:")
    print(f"  Device: {device_manager.type}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Workers: {config['training']['num_workers']}")
    print(f"  AMP: {config['training']['use_amp']}")

    # Get memory stats before training
    if device_manager.is_cuda:
        mem_before = device_manager.memory_stats()
        print(f"  Memory available: {mem_before.get('free', 0):.2f} GB")

    # Get resume path if it exists
    resume_from = getattr(args, 'resume', None)
    patience = getattr(args, 'patience', 10)

    # Run training
    checkpoint_dir = trainer.train(
        modality=args.modality,
        num_epochs=config['training']['num_epochs'],
        resume_from=resume_from,
        early_stopping_patience=patience
    )

    # Clean up GPU memory after training
    device_manager.empty_cache()

    logger.info(f"Training completed! Checkpoints saved to: {checkpoint_dir}")
    return checkpoint_dir


def evaluate_model(args, device_manager):
    """Evaluate a trained model on downstream tasks."""
    logger.info(f"Evaluating {args.modality.upper()} model on {device_manager.type}")

    # Find encoder checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    encoder_path = checkpoint_dir / "encoder.pt"

    if not encoder_path.exists():
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            logger.info(f"Encoder not found, using best model: {best_model_path}")
            encoder_path = best_model_path
        else:
            logger.error(f"No encoder found in {checkpoint_dir}")
            sys.exit(1)

    # Get data directory
    data_dir = getattr(args, 'data_dir', 'data/but_ppg/dataset')

    # Create evaluator with device manager
    evaluator = DownstreamEvaluator(
        encoder_path=str(encoder_path),
        config_path=args.config,
        device_manager=device_manager  # Pass device manager
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

    # Clean up GPU memory
    device_manager.empty_cache()

    logger.info(f"Results saved to: {results_path}")
    return results_path


def compare_with_paper(args, device_manager, results_path):
    """Compare results with paper benchmarks."""
    logger.info("Comparing with paper benchmarks")

    comparator = ResultsComparator(config_path=args.config)

    # Load results
    comparator.load_results(str(results_path), modality=args.modality)

    # Generate comparison
    comparison_dir = results_path.parent / 'comparisons'
    report = comparator.generate_report(save_dir=str(comparison_dir))

    # Print summary
    comparator.print_summary()

    # Generate plot
    plot_path = comparison_dir / f'comparison_{args.modality}.png'
    comparator.plot_comparison(save_path=str(plot_path))

    logger.info(f"Comparison saved to: {comparison_dir}")

    # Clean up GPU memory if needed
    device_manager.empty_cache()

    return report


def run_full_pipeline(args, device_manager):
    """Run the complete pipeline: train -> evaluate -> compare."""
    logger.info(f"Running full pipeline for {args.modality.upper()} on {device_manager.type}")

    # Step 1: Train (or use existing checkpoint)
    if args.skip_training and args.checkpoint_dir:
        logger.info("Skipping training, using existing checkpoint")
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)
    else:
        checkpoint_dir = train_model(args, device_manager)

    # Step 2: Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("Starting evaluation phase...")
    logger.info("=" * 60)

    # Create args for evaluate
    eval_args = argparse.Namespace(
        checkpoint_dir=str(checkpoint_dir),
        modality=args.modality,
        config=args.config,
        data_dir=getattr(args, 'data_dir', 'data/but_ppg/dataset')
    )
    results_path = evaluate_model(eval_args, device_manager)

    # Step 3: Compare with paper
    logger.info("\n" + "=" * 60)
    logger.info("Starting comparison with paper benchmarks...")
    logger.info("=" * 60)

    # Create args for compare
    compare_args = argparse.Namespace(
        results_path=str(results_path),
        modality=args.modality,
        config=args.config
    )
    compare_with_paper(compare_args, device_manager, results_path)

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results saved in: {checkpoint_dir}")

    # Final cleanup
    device_manager.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Biosignal Foundation Model - Main Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add device argument
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'],
                        default='auto', help='Device to use (auto for automatic selection)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='CUDA device ID for multi-GPU systems')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # === TRAIN COMMAND ===
    train_parser = subparsers.add_parser('train', help='Train SSL model')
    train_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                              default='ppg', help='Signal modality to train')
    train_parser.add_argument('--name', type=str, default='ssl',
                              help='Experiment name')
    train_parser.add_argument('--config', type=str, default='configs/config.yaml',
                              help='Path to configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (auto if not specified)')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--workers', type=int, help='Number of data workers (auto if not specified)')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--patience', type=int, default=10,
                              help='Early stopping patience')

    train_parser.add_argument('--ssl-method', type=str, default='infonce',
                              choices=['infonce', 'simsiam'],
                              help='SSL method: infonce (Apple paper) or simsiam (small data)')
    train_parser.add_argument('--simsiam-proj-dim', type=int, default=2048,
                              help='SimSiam projection dimension')
    train_parser.add_argument('--simsiam-pred-dim', type=int, default=512,
                              help='SimSiam prediction dimension')



    # === EVALUATE COMMAND ===
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--checkpoint-dir', type=str, required=True,
                             help='Path to checkpoint directory')
    eval_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                             default='ppg', help='Signal modality')
    eval_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Path to configuration file')

    # === COMPARE COMMAND ===
    compare_parser = subparsers.add_parser('compare', help='Compare with paper benchmarks')
    compare_parser.add_argument('--results-path', type=str, required=True,
                                help='Path to evaluation results')
    compare_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                                default='ppg', help='Signal modality')
    compare_parser.add_argument('--config', type=str, default='configs/config.yaml',
                                help='Path to configuration file')

    # === FULL PIPELINE COMMAND ===
    full_parser = subparsers.add_parser('full', help='Run full pipeline (train + evaluate + compare)')
    full_parser.add_argument('--modality', type=str, choices=['ppg', 'ecg', 'acc'],
                             default='ppg', help='Signal modality')
    full_parser.add_argument('--name', type=str, default='full',
                             help='Experiment name')
    full_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Path to configuration file')
    full_parser.add_argument('--data-dir', type=str, default='data/but_ppg/dataset',
                             help='Path to dataset directory')
    full_parser.add_argument('--epochs', type=int, help='Number of epochs')
    full_parser.add_argument('--batch-size', type=int, help='Batch size (auto if not specified)')
    full_parser.add_argument('--lr', type=float, help='Learning rate')
    full_parser.add_argument('--workers', type=int, help='Number of data workers (auto if not specified)')
    full_parser.add_argument('--skip-training', action='store_true',
                             help='Skip training, use existing checkpoint')
    full_parser.add_argument('--checkpoint-dir', type=str,
                             help='Existing checkpoint directory (if skipping training)')
    full_parser.add_argument('--patience', type=int, default=10,
                             help='Early stopping patience')
    full_parser.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint')

    # Around line 425 in full_parser section, add the same:
    full_parser.add_argument('--ssl-method', type=str, default='infonce',
                             choices=['infonce', 'simsiam'],
                             help='SSL method: infonce (Apple paper) or simsiam (small data)')
    full_parser.add_argument('--simsiam-proj-dim', type=int, default=2048,
                             help='SimSiam projection dimension')
    full_parser.add_argument('--simsiam-pred-dim', type=int, default=512,
                             help='SimSiam prediction dimension')

    # === TEST COMMAND ===
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--module', type=str,
                             choices=['all', 'data', 'model', 'augment', 'ssl', 'train', 'device'],
                             default='all', help='Module to test')

    train_parser.add_argument('--downsample', action='store_true',
                              help='Use 15s segments instead of 60s for 4x faster training')

    # Add to evaluate parser:
    eval_parser.add_argument('--downsample', action='store_true',
                             help='Use 15s segments for faster evaluation')

    # Add to full parser:
    full_parser.add_argument('--downsample', action='store_true',
                             help='Use 15s segments for faster training and evaluation')
    parser.add_argument('--downsample', action='store_true',
                        help='Enable downsampling to 10-second segments')
    args = parser.parse_args()

    # Initialize device manager with user preference
    device_manager = get_device_manager()
    if hasattr(args, 'device') and args.device != 'auto':
        device_manager.set_device(args.device, args.device_id)

    # Check environment
    check_environment()

    # Execute command
    if args.command == 'train':
        train_model(args, device_manager)

    elif args.command == 'evaluate':
        evaluate_model(args, device_manager)

    elif args.command == 'compare':
        compare_with_paper(args, device_manager, Path(args.results_path))

    elif args.command == 'full':
        run_full_pipeline(args, device_manager)

    elif args.command == 'test':
        print("\nRunning tests...")
        if args.module in ['all', 'device']:
            print("\n=== Testing Device Manager ===")
            test_device_manager()
        if args.module in ['all', 'data']:
            from data import test_data_loading
            test_data_loading()
        if args.module in ['all', 'model']:
            from model import test_model
            test_model()
        if args.module in ['all', 'augment']:
            from augment import test_augmentations
            test_augmentations()
        if args.module in ['all', 'ssl']:
            from ssl_model import test_ssl
            test_ssl()
        if args.module in ['all', 'train']:
            from train import test_training
            test_training()
        print("\nAll tests completed!")

    else:
        parser.print_help()
        print_quick_start_guide()


def test_device_manager():
    """Test device manager functionality."""
    print("Testing Device Manager...")

    dm = get_device_manager()
    print(f"  Device: {dm.device}")
    print(f"  Type: {dm.type}")
    print(f"  Supports AMP: {dm.supports_amp}")
    print(f"  Supports compile: {dm.supports_compile}")

    # Test batch size recommendations
    for modality in ['ppg', 'ecg', 'acc']:
        batch_size = dm.get_optimal_batch_size(modality)
        print(f"  Recommended batch size for {modality}: {batch_size}")

    # Test memory stats (CUDA only)
    if dm.is_cuda:
        stats = dm.memory_stats()
        print(f"  Memory stats: {stats}")

    print("  ✓ Device manager test passed!")


def print_quick_start_guide():
    print("\n" + "=" * 60)
    print("QUICK START GUIDE")
    print("=" * 60)
    print("""
1. Check your environment:
   python main.py

2. Quick training test (few epochs):
   python main.py train --modality ppg --epochs 5 --batch-size 16

3. Full training:
   python main.py train --modality ppg --epochs 100

4. Train with specific device:
   python main.py --device cuda train --modality ppg
   python main.py --device mps train --modality ecg
   python main.py --device cpu train --modality acc

5. Run complete pipeline (train + evaluate + compare):
   python main.py full --modality ppg --epochs 100
   python main.py full --modality ecg --epochs 50
   python main.py full --modality acc --epochs 100

6. Skip training in full pipeline (use existing model):
   python main.py full --modality ppg --skip-training --checkpoint-dir [path]

7. Resume training:
   python main.py train --modality ppg --resume [checkpoint_path]
   python main.py full --modality ppg --resume [checkpoint_path]

8. Evaluate existing model:
   python main.py evaluate --checkpoint-dir [path] --modality ppg

9. Compare with paper benchmarks:
   python main.py compare --results-path [path] --modality ppg

10. Multi-GPU training (CUDA):
    python main.py --device cuda --device-id 1 train --modality ppg

11. Run tests:
    python main.py test --module all
    python main.py test --module device

Need help with a specific command?
   python main.py [command] --help
""")


if __name__ == "__main__":
    main()


    """bash# Basic PPG training
python main.py train --modality ppg --epochs 20 --batch-size 32

# PPG with custom settings for faster training
python main.py train --modality ppg --epochs 50 --batch-size 32 --lr 0.001

# PPG with more workers for faster data loading
python main.py train --modality ppg --epochs 100 --workers 8
Train ECG Model:
bash# Basic ECG training
python main.py train --modality ecg --epochs 100

# ECG with custom settings
python main.py train --modality ecg --epochs 20 --batch-size 64 --lr 0.0005
Train ACC Model (Accelerometer):
bash# Basic ACC training
python main.py train --modality acc --epochs 100

# ACC with custom settings
python main.py train --modality acc --epochs 100 --batch-size 32 --name acc_experiment
3. Evaluation Commands:
bash# Evaluate a trained model
python main.py evaluate --modality ppg --checkpoint-dir data/outputs/checkpoints/ppg_ssl_20240101

# Evaluate ACC model
python main.py evaluate --modality acc --checkpoint-dir data/outputs/checkpoints/acc_ssl_20240101
4. Full Pipeline (Train + Evaluate + Compare):
bash# Run everything for PPG
python main.py full --modality ppg --epochs 100

# Run everything for ACC
python main.py full --modality acc --epochs 100 --batch-size 32

# Skip training if you already have a model
python main.py full --modality ppg --skip-training --checkpoint-dir data/outputs/checkpoints/existing_model
5. Resume Training:
bash# Resume from checkpoint
python main.py train --modality ppg --resume data/outputs/checkpoints/ppg_ssl/checkpoint_epoch_50.pt
6. Run Tests:
bash# Test everything
python main.py test

# Test specific module
python main.py test --module model
python main.py test --module augment
"""