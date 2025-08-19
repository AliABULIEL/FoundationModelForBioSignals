#!/usr/bin/env python3
"""
Unified training script for SSL pre-training and downstream evaluation.
Implements the complete training pipeline from the Apple paper with MIMIC-IV data.
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import random
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import all necessary modules
from biosignal.data import MIMICWaveformDataset, create_dataloaders
from biosignal.augmentations import ContrastiveAugmentation
from biosignal.ssl_loss import create_ssl_model
from biosignal.evaluate import DownstreamEvaluator

# Optional: Weights & Biases for experiment tracking
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def train_ssl(args):
    """
    SSL pre-training following the paper's protocol.

    Key aspects from paper:
    - Participant-level positive pairs (Section 3.1)
    - Regularized InfoNCE loss with KoLeo (Section 3.2)
    - Momentum encoder with τ = 0.99 (Section 3.3)
    - Batch size = 256 (Section 4.3)
    - Learning rate = 0.001 with step decay
    - Training on 32 A100 GPUs (we'll use what's available)
    """
    device = get_device()

    # Initialize wandb for experiment tracking (optional)
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="biosignal-foundation-mimic",
            name=f"{args.modality}_ssl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )

    # Create datasets with paper specifications
    print(f"\nLoading {args.modality.upper()} dataset from MIMIC-IV")
    print(f"  Index: {args.index_csv}")
    print(f"  Data root: {args.data_root}")
    print(f"  Labels: {args.labels_csv if args.labels_csv else 'None'}")

    # Create train/val/test dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        index_csv=args.index_csv,
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modality=args.modality,
        min_segments_per_participant=args.min_segments
    )

    print(f"\nDataset statistics:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create augmentation module (modality-specific)
    augment = ContrastiveAugmentation(modality=args.modality)

    # Create SSL model with paper parameters
    model = create_ssl_model(
        modality=args.modality,
        temperature=args.temperature,
        lambda_koleo=args.lambda_koleo,
        momentum_rate=args.momentum_rate
    ).to(device)

    # Count parameters (should match paper approximately)
    total_params = sum(p.numel() for p in model.encoder.parameters())
    trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    if args.modality == 'ppg':
        print(f"  Paper reports: ~3.3M (with 4 channels, we have 1)")
    else:
        print(f"  Paper reports: ~2.5M")

    # Create optimizer (paper uses Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler (step decay as mentioned in paper)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    # Mixed precision training for efficiency
    use_amp = args.use_amp and device.type in ['cuda', 'mps']
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_epoch, best_loss = model.load_checkpoint(args.resume_from, optimizer)
        print(f"  Starting from epoch {start_epoch}")
        print(f"  Best loss so far: {best_loss:.4f}")

    # Training loop
    print(f"\nStarting SSL pre-training")
    print(f"  Epochs: {start_epoch} -> {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Lambda KoLeo: {args.lambda_koleo}")
    print(f"  Momentum rate: {args.momentum_rate}")

    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_losses = []
        train_metrics = {
            'loss_contrastive': [],
            'loss_koleo': [],
        }

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 2:
                seg1, seg2 = batch
            elif len(batch) == 3:
                seg1, seg2, _ = batch  # Ignore participant IDs during training
            elif len(batch) == 4:
                seg1, seg2, _, _ = batch  # Ignore IDs and labels
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            # Move to device
            seg1 = seg1.to(device)
            seg2 = seg2.to(device)

            # Apply augmentations
            seg1_aug, seg2_aug = augment(seg1, seg2)

            # Forward pass with mixed precision
            if use_amp and scaler is not None:
                with autocast():
                    loss, loss_dict = model(seg1_aug, seg2_aug)
            else:
                loss, loss_dict = model(seg1_aug, seg2_aug)

            # Backward pass
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Update momentum networks (critical for paper's approach)
            model.update_momentum_networks()

            # Track losses
            train_losses.append(loss.item())
            train_metrics['loss_contrastive'].append(loss_dict['loss_contrastive'])
            train_metrics['loss_koleo'].append(loss_dict['loss_koleo'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cont': f"{loss_dict['loss_contrastive']:.4f}",
                'koleo': f"{loss_dict['loss_koleo']:.4f}"
            })

            # Log to wandb
            if args.use_wandb and WANDB_AVAILABLE and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_contrastive': loss_dict['loss_contrastive'],
                    'train/loss_koleo': loss_dict['loss_koleo'],
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': epoch * len(train_loader) + batch_idx
                })

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"):
                # Handle batch format
                if len(batch) == 2:
                    seg1, seg2 = batch
                elif len(batch) >= 3:
                    seg1, seg2 = batch[0], batch[1]

                seg1 = seg1.to(device)
                seg2 = seg2.to(device)

                # No augmentation for validation
                loss, loss_dict = model(seg1, seg2)
                val_losses.append(loss.item())

        # Epoch summary
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"\nEpoch {epoch + 1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"    - Contrastive: {np.mean(train_metrics['loss_contrastive']):.4f}")
        print(f"    - KoLeo: {np.mean(train_metrics['loss_koleo']):.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoints
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"{args.modality}_checkpoint_epoch_{epoch + 1}.pt"
            model.save_checkpoint(checkpoint_path, optimizer, epoch + 1, avg_val_loss)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_path = checkpoint_dir / f"{args.modality}_best_model.pt"
            model.save_checkpoint(best_path, optimizer, epoch + 1, best_loss)
            print(f"  New best model saved (val loss: {best_loss:.4f})")

            # Also save encoder only for downstream tasks
            encoder_path = checkpoint_dir / f"{args.modality}_best_encoder.pt"
            model.save_encoder(encoder_path)

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/avg_loss': avg_train_loss,
                'val/avg_loss': avg_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

    print("\nTraining completed!")

    # Save final model
    final_checkpoint = checkpoint_dir / f"{args.modality}_final_model.pt"
    model.save_checkpoint(final_checkpoint, optimizer, args.epochs, avg_val_loss)

    final_encoder = checkpoint_dir / f"{args.modality}_final_encoder.pt"
    model.save_encoder(final_encoder)
    print(f"Final encoder saved: {final_encoder}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return str(final_encoder)


def evaluate_downstream(args):
    """
    Evaluate pre-trained model on downstream tasks.
    Following paper's evaluation protocol (Section 4.2 and Section 5).
    """
    device = get_device()

    print(f"\nDownstream Evaluation")
    print(f"  Encoder: {args.encoder_checkpoint}")
    print(f"  Modality: {args.modality}")
    print(f"  Task: {args.task}")

    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    print(f"  Labels loaded: {len(labels_df)} participants")

    # Create data loader for evaluation
    from biosignal.data import MIMICWaveformDataset

    dataset = MIMICWaveformDataset(
        index_csv=args.index_csv,
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        modality=args.modality,
        split='test',
        return_participant_id=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create evaluator
    evaluator = DownstreamEvaluator(
        encoder_checkpoint=args.encoder_checkpoint,
        modality=args.modality,
        device=device
    )

    # Evaluate based on task
    if args.task == 'all':
        # Evaluate all demographic tasks (reproduce Table 2)
        print("\nEvaluating all demographic tasks...")
        results_df = evaluator.evaluate_all_demographics(dataloader, labels_df)

        print("\n" + "=" * 60)
        print("Results (Paper Table 2 format):")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print("=" * 60)

        # Save results
        if args.save_results:
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            results_path = results_dir / f"{args.modality}_all_demographics.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")

    elif args.task in evaluator.demographic_tasks:
        # Single demographic task
        results = evaluator.evaluate_demographics(
            dataloader, labels_df, args.task
        )

        print(f"\nResults for {args.task}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        # Compare with paper results
        paper_results = {
            'ppg': {
                'age_classification': 0.976,
                'bmi_classification': 0.918,
                'sex_classification': 0.993
            },
            'ecg': {
                'age_classification': 0.916,
                'bmi_classification': 0.797,
                'sex_classification': 0.951
            }
        }

        if 'classification' in args.task and args.task in paper_results[args.modality]:
            paper_auc = paper_results[args.modality][args.task]
            our_auc = results['auc']
            print(f"\nComparison with paper:")
            print(f"  Paper AUC: {paper_auc:.3f}")
            print(f"  Our AUC: {our_auc:.3f}")
            print(f"  Difference: {our_auc - paper_auc:+.3f}")

    else:
        print(f"Unknown task: {args.task}")
        print(f"Available tasks: {list(evaluator.demographic_tasks.keys())} or 'all'")


def main():
    parser = argparse.ArgumentParser(
        description="Foundation Model for Biosignals - MIMIC-IV Implementation"
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # SSL Pre-training arguments
    ssl_parser = subparsers.add_parser(
        'pretrain',
        help='SSL pre-training with participant-level contrastive learning'
    )

    # Data arguments
    ssl_parser.add_argument('--index_csv', type=str, required=True,
                            help='Path to waveform_index.csv from download script')
    ssl_parser.add_argument('--data_root', type=str, required=True,
                            help='Path to MIMIC-IV waves directory')
    ssl_parser.add_argument('--labels_csv', type=str, default=None,
                            help='Path to labels.csv (optional for pretraining)')
    ssl_parser.add_argument('--modality', type=str, default='ppg',
                            choices=['ppg', 'ecg'],
                            help='Signal modality (default: ppg)')
    ssl_parser.add_argument('--min_segments', type=int, default=4,
                            help='Minimum segments per participant (paper: 4)')

    # Model arguments (paper defaults)
    ssl_parser.add_argument('--temperature', type=float, default=0.04,
                            help='InfoNCE temperature (paper: 0.04)')
    ssl_parser.add_argument('--lambda_koleo', type=float, default=0.1,
                            help='KoLeo regularization weight (paper: 0.1)')
    ssl_parser.add_argument('--momentum_rate', type=float, default=0.99,
                            help='Momentum encoder update rate (paper: 0.99)')

    # Training arguments
    ssl_parser.add_argument('--epochs', type=int, default=300,
                            help='Number of training epochs')
    ssl_parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size (paper: 256)')
    ssl_parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='Initial learning rate (paper: 0.001)')
    ssl_parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='Weight decay')
    ssl_parser.add_argument('--lr_step_size', type=int, default=100,
                            help='LR scheduler step size')
    ssl_parser.add_argument('--lr_gamma', type=float, default=0.1,
                            help='LR scheduler gamma')

    # System arguments
    ssl_parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of data loading workers')
    ssl_parser.add_argument('--use_amp', action='store_true',
                            help='Use mixed precision training')
    ssl_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                            help='Directory to save checkpoints')
    ssl_parser.add_argument('--save_freq', type=int, default=10,
                            help='Save checkpoint every N epochs')
    ssl_parser.add_argument('--resume_from', type=str, default=None,
                            help='Resume training from checkpoint')
    ssl_parser.add_argument('--use_wandb', action='store_true',
                            help='Use Weights & Biases for logging')
    ssl_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')

    # Downstream evaluation arguments
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate pre-trained model on downstream tasks'
    )

    eval_parser.add_argument('--encoder_checkpoint', type=str, required=True,
                             help='Path to pre-trained encoder checkpoint')
    eval_parser.add_argument('--index_csv', type=str, required=True,
                             help='Path to waveform_index.csv')
    eval_parser.add_argument('--data_root', type=str, required=True,
                             help='Path to MIMIC-IV waves directory')
    eval_parser.add_argument('--labels_csv', type=str, required=True,
                             help='Path to labels.csv with demographics')
    eval_parser.add_argument('--task', type=str, default='all',
                             help='Task name or "all" for all demographics')
    eval_parser.add_argument('--modality', type=str, required=True,
                             choices=['ppg', 'ecg'],
                             help='Signal modality')
    eval_parser.add_argument('--batch_size', type=int, default=512,
                             help='Batch size for evaluation')
    eval_parser.add_argument('--num_workers', type=int, default=4,
                             help='Number of data loading workers')
    eval_parser.add_argument('--save_results', action='store_true',
                             help='Save evaluation results')
    eval_parser.add_argument('--results_dir', type=str, default='results',
                             help='Directory to save results')
    eval_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Execute command
    if args.command == 'pretrain':
        train_ssl(args)
    elif args.command == 'evaluate':
        evaluate_downstream(args)


if __name__ == '__main__':
    main()



    """
    # PPG pre-training
python train.py pretrain \
    --index_csv data/outputs/waveform_index.csv \
    --data_root data/mimic4wdb/physionet.org/files/mimic4wdb/0.1.0/waves \
    --labels_csv data/outputs/labels.csv \
    --modality ppg \
    --batch_size 256 \
    --epochs 300 \
    --use_amp \
    --checkpoint_dir checkpoints/ppg

# ECG pre-training  
python train.py pretrain \
    --index_csv data/outputs/waveform_index.csv \
    --data_root data/mimic4wdb/physionet.org/files/mimic4wdb/0.1.0/waves \
    --modality ecg \
    --batch_size 256 \
    --epochs 300 
    
    
    
    python train.py evaluate \
    --encoder_checkpoint checkpoints/ppg/ppg_best_encoder.pt \
    --index_csv data/outputs/waveform_index.csv \
    --data_root data/mimic4wdb/physionet.org/files/mimic4wdb/0.1.0/waves \
    --labels_csv data/outputs/labels.csv \
    --modality ppg \
    --task all \
    --save_results

# Evaluate specific task
python train.py evaluate \
    --encoder_checkpoint checkpoints/ecg/ecg_best_encoder.pt \
    --index_csv data/outputs/waveform_index.csv \
    --data_root data/mimic4wdb/physionet.org/files/mimic4wdb/0.1.0/waves \
    --labels_csv data/outputs/labels.csv \
    --modality ecg \
    --task age_classification  
    """