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
import pandas as pd


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import device manager first
from device import get_device_manager

# Import other modules
from deprecated.train import Trainer
from deprecated.evaluate import DownstreamEvaluator
from compare import ResultsComparator
from data import BUTPPGDataset
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


# Updates for main.py to support new evaluation and fine-tuning flows

def run_evaluate_extended(args, device_manager):
    """
    Extended evaluation supporting both BUT PPG and VitalDB.
    Can evaluate on one or both datasets.
    """
    logger.info(f"Evaluating {args.modality.upper()} model on {args.dataset_type}")

    # Find checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    encoder_path = checkpoint_dir / "encoder.pt"
    if not encoder_path.exists():
        encoder_path = checkpoint_dir / "best_model.pt"

    if not encoder_path.exists():
        logger.error(f"No model found in {checkpoint_dir}")
        sys.exit(1)

    # Create evaluator
    evaluator = DownstreamEvaluator(
        encoder_path=str(encoder_path),
        config_path=args.config,
        device_manager=device_manager
    )

    all_results = []

    # Evaluate on specified datasets
    if args.dataset_type == 'both':
        # Evaluate on both VitalDB and BUT PPG
        datasets_to_eval = ['vitaldb', 'but_ppg']
    else:
        datasets_to_eval = [args.dataset_type]

    for dataset_type in datasets_to_eval:
        print(f"\n{'=' * 60}")
        print(f"EVALUATING ON {dataset_type.upper()}")
        print(f"{'=' * 60}")

        evaluator.clear_cache()

        # Run evaluation
        results_path = checkpoint_dir / f"downstream_results_{args.modality}_{dataset_type}.csv"

        results_df = evaluator.evaluate_all_tasks_extended(
            dataset_type=dataset_type,
            modality=args.modality,
            split='test',
            data_dir=args.data_dir if dataset_type == 'but_ppg' else None,
            save_path=str(results_path),
            downsample=args.downsample
        )

        # Print results
        print(f"\n{dataset_type.upper()} RESULTS - {args.modality.upper()}")
        print("=" * 60)
        print(results_df.to_string())

        all_results.append(results_df)

        # Clean GPU memory between evaluations
        device_manager.empty_cache()

    # If evaluating both, save combined results
    if len(all_results) > 1:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = checkpoint_dir / f"downstream_results_{args.modality}_combined.csv"
        combined_df.to_csv(combined_path, index=False)

        print("\n" + "=" * 60)
        print("COMBINED RESULTS COMPARISON")
        print("=" * 60)

        # Print side-by-side comparison
        for task in combined_df['task'].unique():
            print(f"\n{task}:")
            task_df = combined_df[combined_df['task'] == task]
            for _, row in task_df.iterrows():
                dataset = row['dataset']
                if 'auc' in row:
                    print(f"  {dataset:10s}: AUC={row.get('auc', 0):.3f}, Acc={row.get('accuracy', 0):.3f}")
                elif 'mae' in row:
                    print(f"  {dataset:10s}: MAE={row.get('mae', 0):.3f}, R2={row.get('r2', 0):.3f}")

        logger.info(f"Combined results saved to: {combined_path}")
        return combined_path
    else:
        return results_path


def run_compare_extended(args, device_manager):
    """
    Extended comparison supporting VitalDB results.
    Can compare VitalDB results with paper benchmarks.
    """
    logger.info(f"Comparing {args.dataset_type} results with paper benchmarks")

    results_path = Path(args.results_path)

    comparator = ResultsComparator(config_path=args.config)

    # Determine dataset type from filename if not specified
    if args.dataset_type == 'auto':
        if 'vitaldb' in str(results_path).lower():
            dataset_type = 'vitaldb'
        elif 'combined' in str(results_path).lower():
            dataset_type = 'combined'
        else:
            dataset_type = 'but_ppg'
    else:
        dataset_type = args.dataset_type

    # Load results
    comparator.load_results(str(results_path), modality=args.modality)

    # Generate comparison report
    comparison_dir = results_path.parent / f'comparisons_{dataset_type}'
    report = comparator.generate_report(save_dir=str(comparison_dir))

    # Print summary with dataset type
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {dataset_type.upper()} vs PAPER BENCHMARKS")
    print(f"{'=' * 60}")
    comparator.print_summary()

    # Create comparison plot
    plot_path = comparison_dir / f'comparison_{args.modality}_{dataset_type}.png'
    comparator.plot_comparison(save_path=str(plot_path))

    logger.info(f"Comparison saved to: {comparison_dir}")
    device_manager.empty_cache()
    return report


def run_finetune_extended(args, device_manager):
    """
    Extended fine-tuning supporting both datasets.
    Can fine-tune on VitalDB, BUT PPG, or both sequentially.
    """
    if not args.pretrained_path:
        raise ValueError("--pretrained-path required for fine-tuning")

    dataset_types = []
    if args.finetune_dataset == 'both':
        # Fine-tune on VitalDB first, then BUT PPG
        dataset_types = ['vitaldb', 'but_ppg']
    else:
        dataset_types = [args.finetune_dataset]

    current_model_path = args.pretrained_path

    for dataset_type in dataset_types:
        logger.info(f"Starting fine-tuning on {dataset_type.upper()} for {args.modality.upper()}")

        config = get_config()

        # Get dataset-specific config
        if dataset_type == 'vitaldb':
            finetune_config = config.config.get('finetune_vitaldb',
                                                config.config.get('finetune', {}))
        else:
            finetune_config = config.config.get('finetune', {})

        # Create experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"finetune_{dataset_type}_{args.modality}_{timestamp}"

        # Initialize trainer
        trainer = Trainer(
            config_path=args.config,
            experiment_name=experiment_name,
            device_manager=device_manager,
            ssl_method=args.ssl_method,
            phase='finetune'
        )

        # Set dataset type for trainer
        trainer.dataset_type = dataset_type
        trainer.pretrained_path = current_model_path

        print(f"\n🚀 Fine-tuning Configuration:")
        print(f"  Dataset: {dataset_type.upper()}")
        print(f"  Device: {device_manager.type}")
        print(f"  Batch size: {args.batch_size or finetune_config.get('batch_size', 64)}")
        print(f"  Learning rate: {args.lr or finetune_config.get('learning_rate', 0.00001)}")
        print(f"  Epochs: {args.epochs or finetune_config.get('epochs', 20)}")
        print(f"  Pretrained model: {current_model_path}")

        # Train
        checkpoint_dir = trainer.train(
            modality=args.modality,
            num_epochs=args.epochs or finetune_config.get('epochs', 20),
            early_stopping_patience=args.patience or finetune_config.get('early_stopping_patience', 10)
        )

        # Update model path for next iteration (if fine-tuning on both)
        if args.finetune_dataset == 'both':
            current_model_path = str(checkpoint_dir / "encoder.pt")
            logger.info(f"Using {dataset_type} fine-tuned model for next stage: {current_model_path}")

    logger.info(f"Fine-tuning completed! Final model saved to: {checkpoint_dir}")
    return checkpoint_dir


def run_full_extended(args, device_manager):
    """
    Extended full pipeline with correct evaluation order:
    pretrain(VitalDB) -> eval(VitalDB) -> finetune(BUT_PPG) -> eval(BUT_PPG) -> compare
    """
    logger.info("Running extended full pipeline")

    all_results = {}  # Store all results for comparison

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

    # Step 2: Evaluate pre-trained model on VitalDB
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: EVALUATING PRE-TRAINED MODEL ON VITALDB")
    logger.info("=" * 60)

    args.checkpoint_dir = str(pretrain_dir)
    args.dataset_type = 'vitaldb'
    vitaldb_pretrain_results = run_evaluate_extended(args, device_manager)
    all_results['vitaldb_pretrained'] = vitaldb_pretrain_results

    # Step 3: Fine-tune on BUT PPG
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: FINE-TUNING ON BUT_PPG")
    logger.info("=" * 60)

    # Force fine-tuning dataset to BUT PPG for this pipeline
    original_finetune_dataset = args.finetune_dataset
    args.finetune_dataset = 'but_ppg'
    finetune_dir = run_finetune_extended(args, device_manager)
    args.finetune_dataset = original_finetune_dataset  # Restore original

    # Step 4: Evaluate fine-tuned model on BUT PPG
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: EVALUATING FINE-TUNED MODEL ON BUT_PPG")
    logger.info("=" * 60)

    args.checkpoint_dir = str(finetune_dir)
    args.dataset_type = 'but_ppg'
    butppg_finetune_results = run_evaluate_extended(args, device_manager)
    all_results['but_ppg_finetuned'] = butppg_finetune_results

    # Optional Step 5: Evaluate fine-tuned model on VitalDB to see transfer effect
    if args.eval_dataset == 'both':
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: EVALUATING FINE-TUNED MODEL ON VITALDB (TRANSFER CHECK)")
        logger.info("=" * 60)

        args.dataset_type = 'vitaldb'
        vitaldb_finetune_results = run_evaluate_extended(args, device_manager)
        all_results['vitaldb_finetuned'] = vitaldb_finetune_results

    # Step 6: Compare results
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6: COMPARISON AND SUMMARY")
    logger.info("=" * 60)

    # Print summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    if 'vitaldb_pretrained' in all_results:
        print("\n1. Pre-trained Model on VitalDB:")
        print(f"   Results saved: {all_results['vitaldb_pretrained']}")

    if 'but_ppg_finetuned' in all_results:
        print("\n2. Fine-tuned Model on BUT PPG:")
        print(f"   Results saved: {all_results['but_ppg_finetuned']}")

    if 'vitaldb_finetuned' in all_results:
        print("\n3. Fine-tuned Model on VitalDB (Transfer):")
        print(f"   Results saved: {all_results['vitaldb_finetuned']}")
        print("   → Shows how BUT PPG fine-tuning affects VitalDB performance")

    # Compare with paper benchmarks for each result
    for result_name, result_path in all_results.items():
        if result_path and Path(result_path).exists():
            logger.info(f"\nComparing {result_name} with benchmarks...")
            args.results_path = str(result_path)
            args.dataset_type = 'vitaldb' if 'vitaldb' in result_name else 'but_ppg'
            run_compare_extended(args, device_manager)

    logger.info("\n" + "=" * 60)
    logger.info("EXTENDED PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print("✓ Pre-training on VitalDB: Complete")
    print("✓ Evaluation on VitalDB (baseline): Complete")
    print("✓ Fine-tuning on BUT PPG: Complete")
    print("✓ Evaluation on BUT PPG: Complete")
    if 'vitaldb_finetuned' in all_results:
        print("✓ Transfer evaluation on VitalDB: Complete")
    print("✓ Benchmark comparisons: Complete")

    device_manager.empty_cache()
    return all_results


def run_eval_supervised(args, device_manager):
    """Evaluate supervised heads of a semi-supervised model."""
    logger.info("Evaluating supervised heads")

    # Load checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "final_model.pt"

    if not checkpoint_path.exists():
        logger.error(f"No model found in {checkpoint_dir}")
        return None

    # Load model
    from deprecated.ssl_model import create_ssl_model
    from deprecated.model import EfficientNet1D, ProjectionHead
    from config_loader import get_config

    config = get_config()

    # Get model dimensions from config
    embedding_dim = config.get('model.embedding_dim', 256)
    projection_dim = config.get('model.projection_dim', 128)

    # Create model architecture
    encoder = EfficientNet1D(
        in_channels=1,
        embedding_dim=embedding_dim,
        modality=args.modality
    )
    projection_head = ProjectionHead(
        input_dim=embedding_dim,
        output_dim=projection_dim
    )

    # Create SSL model with supervised heads
    model = create_ssl_model(
        encoder=encoder,
        projection_head=projection_head,
        ssl_method=args.ssl_method,
        use_supervised=True  # Must be True to have heads
    )

    # Load checkpoint - handle different checkpoint formats
    checkpoint = torch.load(checkpoint_path, map_location=device_manager.device, weights_only=False)

    # The checkpoint structure depends on how it was saved
    if 'encoder_state_dict' in checkpoint:
        # Load encoder weights
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)

        # Load projection head if available
        if 'projection_state_dict' in checkpoint:
            model.projection_head.load_state_dict(checkpoint['projection_state_dict'], strict=False)

        # Load supervised heads if available
        if 'age_classifier_state_dict' in checkpoint:
            model.age_classifier.load_state_dict(checkpoint['age_classifier_state_dict'], strict=False)
        if 'bmi_regressor_state_dict' in checkpoint:
            model.bmi_regressor.load_state_dict(checkpoint['bmi_regressor_state_dict'], strict=False)
        if 'sex_classifier_state_dict' in checkpoint:
            model.sex_classifier.load_state_dict(checkpoint['sex_classifier_state_dict'], strict=False)
    else:
        # Try loading the full model state
        try:
            model.load_state_dict(checkpoint, strict=False)
        except:
            logger.error("Could not load model weights - checkpoint format not recognized")
            return None

    model = model.to(device_manager.device)
    model.eval()

    # Create dataset using the same method as evaluate_extended
    from data import create_dataloaders

    # Determine dataset type
    dataset_type = args.dataset_type if hasattr(args, 'dataset_type') else 'but_ppg'

    # Create data loaders with labels
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir if dataset_type == 'but_ppg' else None,
        modality=args.modality,
        batch_size=32,
        num_workers=0,
        dataset_type=dataset_type,
        return_labels=True,  # Important: need labels for evaluation
        downsample=args.downsample if hasattr(args, 'downsample') else False
    )

    # Create evaluator
    from deprecated.evaluate import DownstreamEvaluator
    evaluator = DownstreamEvaluator(
        encoder_path='dummy',  # Not used, we pass model directly
        device_manager=device_manager
    )

    # Evaluate supervised heads using the test dataset
    metrics = evaluator.evaluate_supervised_heads(model, test_loader.dataset)

    print("\n" + "=" * 60)
    print("SUPERVISED HEAD EVALUATION RESULTS")
    print("=" * 60)

    if metrics:
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:.4f}")

        # Save results
        results_path = checkpoint_dir / f"supervised_head_results_{dataset_type}.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Results saved to {results_path}")
    else:
        print("No metrics returned - check if model has supervised heads")
        logger.warning("No supervised metrics computed")

    return metrics
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
                                 'train', 'full', 'test', 'eval-supervised'],  # Added
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

    parser.add_argument('--dataset-type', type=str,
                        choices=['but_ppg', 'vitaldb', 'both', 'auto'],
                        default='but_ppg',
                        help='Dataset to use for evaluation')

    parser.add_argument('--finetune-dataset', type=str,
                        choices=['but_ppg', 'vitaldb', 'both'],
                        default='but_ppg',
                        help='Dataset to use for fine-tuning')

    parser.add_argument('--eval-dataset', type=str,
                        choices=['but_ppg', 'vitaldb', 'both'],
                        default='but_ppg',
                        help='Dataset to use for evaluation')

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
        'finetune': run_finetune_extended,  # Use extended version
        'evaluate': run_evaluate_extended,  # Use extended version
        'compare': run_compare_extended,  # Use extended version
        'train': run_train,
        'full': run_full_extended,  # Use extended version
        'test': lambda args, dm: run_test(args),
        'eval-supervised': run_eval_supervised,
    }

    # Execute the command
    if args.command in command_map:
        command_map[args.command](args, device_manager)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()