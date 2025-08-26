#!/usr/bin/env python3
"""
Standalone script to downsample BUT PPG dataset for faster training/evaluation.
This script reads the original dataset and creates a downsampled version.

Usage:
    python downsample_dataset.py --input_dir data/but_ppg/dataset --output_dir data/but_ppg_downsampled
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def downsample_signal(data, original_fs, target_fs):
    """
    Downsample a signal from original_fs to target_fs.

    Args:
        data: Input signal
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency

    Returns:
        Downsampled signal
    """
    if original_fs == target_fs:
        return data

    # Calculate downsampling factor
    downsample_factor = original_fs / target_fs

    if downsample_factor != int(downsample_factor):
        # Use resample for non-integer factors
        num_samples = int(len(data) * target_fs / original_fs)
        return signal.resample(data, num_samples)
    else:
        # Use decimate for integer factors
        return signal.decimate(data, int(downsample_factor), zero_phase=True)


def process_ppg_file(file_path, config):
    """
    Process a single PPG file: load, downsample, and segment.

    Args:
        file_path: Path to .hea or .csv file
        config: Downsampling configuration

    Returns:
        List of downsampled segments
    """
    segments = []

    try:
        # Read the PPG data
        if file_path.endswith('.csv'):
            # CSV format (might have PPG data)
            df = pd.read_csv(file_path)
            if 'ppg' in df.columns:
                data = df['ppg'].values
            elif df.shape[1] >= 1:
                # Assume first column is PPG if not labeled
                data = df.iloc[:, 0].values
            else:
                return segments
        elif file_path.endswith('.hea'):
            # WFDB format - read the corresponding .dat file
            base_path = file_path.replace('.hea', '')
            dat_file = base_path + '.dat'
            if os.path.exists(dat_file):
                # Simple binary read (you might need to adjust based on actual format)
                data = np.fromfile(dat_file, dtype=np.int16)
            else:
                return segments
        else:
            return segments

        # Skip if too short
        min_samples = config['original_segment_length']
        if len(data) < min_samples:
            return segments

        # Normalize the signal
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)

        # Create segments from the original signal
        num_segments = min(
            config['max_segments_per_file'],
            len(data) // min_samples
        )

        for i in range(num_segments):
            start = i * min_samples
            end = start + min_samples
            segment = data[start:end]

            # Downsample the segment
            if config['downsample_temporal']:
                segment = downsample_signal(
                    segment,
                    config['original_fs'],
                    config['target_fs']
                )

            segments.append(segment)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return segments


def process_ecg_file(file_path, config):
    """
    Process a single ECG file: load, downsample, and segment.

    Args:
        file_path: Path to ECG file
        config: Downsampling configuration

    Returns:
        List of downsampled segments
    """
    # Similar to PPG but with 128Hz original sampling rate
    config_ecg = config.copy()
    config_ecg['original_fs'] = 128
    config_ecg['original_segment_length'] = int(config['segment_duration'] * 128)

    return process_ppg_file(file_path, config_ecg)


def get_participant_info(subject_csv_path):
    """
    Load participant information from subject-info.csv.

    Args:
        subject_csv_path: Path to subject-info.csv

    Returns:
        DataFrame with participant information
    """
    try:
        df = pd.read_csv(subject_csv_path)
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Extract participant ID from ID column
        if 'id' in df.columns:
            df['participant_id'] = df['id'].astype(str).str[:3]  # First 3 digits

        return df
    except Exception as e:
        print(f"Error loading subject info: {e}")
        return pd.DataFrame()


def create_downsampled_dataset(input_dir, output_dir, config):
    """
    Create a downsampled version of the BUT PPG dataset.

    Args:
        input_dir: Path to original dataset
        output_dir: Path for downsampled dataset
        config: Downsampling configuration
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy subject info
    subject_csv = input_path / 'subject-info.csv'
    if subject_csv.exists():
        subject_df = get_participant_info(subject_csv)

        # Optionally limit participants
        if config['max_participants']:
            # Get unique participant IDs
            unique_pids = subject_df['participant_id'].unique()[:config['max_participants']]
            subject_df = subject_df[subject_df['participant_id'].isin(unique_pids)]

        # Save filtered subject info
        subject_df.to_csv(output_path / 'subject-info.csv', index=False)
        print(f"Saved subject info with {len(subject_df)} records")

    # Process waves directory
    waves_dir = input_path / 'waves'
    if not waves_dir.exists():
        print(f"Waves directory not found: {waves_dir}")
        return

    output_waves = output_path / 'waves'
    output_waves.mkdir(exist_ok=True)

    # Get all participant directories (e.g., p158, p159, etc.)
    participant_dirs = sorted([d for d in waves_dir.iterdir() if d.is_dir() and d.name.startswith('p')])

    # Limit participants if configured
    if config['max_participants']:
        participant_dirs = participant_dirs[:config['max_participants']]

    print(f"Processing {len(participant_dirs)} participants...")

    stats = {
        'participants': 0,
        'records': 0,
        'ppg_segments': 0,
        'ecg_segments': 0,
        'total_size_mb': 0,
        'downsampled_size_mb': 0
    }

    # Process each participant
    for p_dir in tqdm(participant_dirs, desc="Participants"):
        participant_id = p_dir.name
        output_p_dir = output_waves / participant_id
        output_p_dir.mkdir(exist_ok=True)
        stats['participants'] += 1

        # Get all record directories for this participant
        record_dirs = sorted([d for d in p_dir.iterdir() if d.is_dir()])

        # Limit records per participant
        if config['max_records_per_participant']:
            record_dirs = record_dirs[:config['max_records_per_participant']]

        # Process each record
        for r_dir in record_dirs:
            record_id = r_dir.name
            output_r_dir = output_p_dir / record_id
            output_r_dir.mkdir(exist_ok=True)
            stats['records'] += 1

            # Process PPG files
            ppg_files = list(r_dir.glob('*ppg*.csv')) + list(r_dir.glob('*ppg*.hea'))
            for ppg_file in ppg_files[:config['max_files_per_type']]:
                segments = process_ppg_file(str(ppg_file), config)

                # Save downsampled segments
                for i, segment in enumerate(segments):
                    output_file = output_r_dir / f"{ppg_file.stem}_seg{i}.npy"
                    np.save(output_file, segment.astype(np.float32))
                    stats['ppg_segments'] += 1
                    stats['downsampled_size_mb'] += segment.nbytes / 1024 / 1024

                # Track original size
                stats['total_size_mb'] += ppg_file.stat().st_size / 1024 / 1024

            # Process ECG files
            ecg_files = list(r_dir.glob('*ecg*.csv')) + list(r_dir.glob('*ecg*.hea'))
            for ecg_file in ecg_files[:config['max_files_per_type']]:
                segments = process_ecg_file(str(ecg_file), config)

                # Save downsampled segments
                for i, segment in enumerate(segments):
                    output_file = output_r_dir / f"{ecg_file.stem}_seg{i}.npy"
                    np.save(output_file, segment.astype(np.float32))
                    stats['ecg_segments'] += 1
                    stats['downsampled_size_mb'] += segment.nbytes / 1024 / 1024

                # Track original size
                stats['total_size_mb'] += ecg_file.stat().st_size / 1024 / 1024

            # Copy any metadata files
            for meta_file in r_dir.glob('*.json'):
                shutil.copy2(meta_file, output_r_dir)

    # Save configuration and statistics
    config_output = output_path / 'downsample_config.json'
    with open(config_output, 'w') as f:
        json.dump({
            'config': config,
            'stats': stats
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNSAMPLING COMPLETE")
    print("=" * 60)
    print(f"Participants processed: {stats['participants']}")
    print(f"Records processed: {stats['records']}")
    print(f"PPG segments created: {stats['ppg_segments']}")
    print(f"ECG segments created: {stats['ecg_segments']}")
    print(f"Original size: {stats['total_size_mb']:.2f} MB")
    print(f"Downsampled size: {stats['downsampled_size_mb']:.2f} MB")
    print(f"Compression ratio: {stats['total_size_mb'] / max(stats['downsampled_size_mb'], 0.01):.2f}x")
    print(f"\nOutput directory: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Downsample BUT PPG dataset for faster training')
    parser.add_argument('--input_dir', type=str, default='data/but_ppg/dataset',
                        help='Path to original dataset')
    parser.add_argument('--output_dir', type=str, default='data/but_ppg_downsampled',
                        help='Path for downsampled dataset')

    # Downsampling parameters
    parser.add_argument('--segment_duration', type=float, default=15.0,
                        help='Segment duration in seconds (default: 15s, original: 60s)')
    parser.add_argument('--target_fs', type=int, default=32,
                        help='Target sampling frequency in Hz (default: 32Hz, original: 64Hz for PPG)')
    parser.add_argument('--max_participants', type=int, default=30,
                        help='Maximum number of participants to process (default: 30, None for all)')
    parser.add_argument('--max_records_per_participant', type=int, default=10,
                        help='Maximum records per participant (default: 10)')
    parser.add_argument('--max_segments_per_file', type=int, default=4,
                        help='Maximum segments to extract from each file (default: 4)')
    parser.add_argument('--max_files_per_type', type=int, default=2,
                        help='Maximum files per type (PPG/ECG) per record (default: 2)')
    parser.add_argument('--no_temporal_downsample', action='store_true',
                        help='Skip temporal downsampling (keep original sampling rate)')

    args = parser.parse_args()

    # Configuration
    config = {
        'segment_duration': args.segment_duration,
        'original_fs': 64,  # PPG default, will be adjusted for ECG
        'target_fs': args.target_fs,
        'original_segment_length': int(args.segment_duration * 64),
        'target_segment_length': int(args.segment_duration * args.target_fs),
        'max_participants': args.max_participants,
        'max_records_per_participant': args.max_records_per_participant,
        'max_segments_per_file': args.max_segments_per_file,
        'max_files_per_type': args.max_files_per_type,
        'downsample_temporal': not args.no_temporal_downsample
    }

    print("=" * 60)
    print("BUT PPG DATASET DOWNSAMPLER")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nDownsampling Configuration:")
    print(f"  Segment duration: {config['segment_duration']}s")
    print(f"  Sampling rate: {config['original_fs']}Hz -> {config['target_fs']}Hz")
    print(f"  Segment length: {config['original_segment_length']} -> {config['target_segment_length']} samples")
    print(f"  Max participants: {config['max_participants'] or 'All'}")
    print(f"  Max records per participant: {config['max_records_per_participant']}")
    print(f"  Max segments per file: {config['max_segments_per_file']}")
    print("=" * 60)

    # Create downsampled dataset
    create_downsampled_dataset(args.input_dir, args.output_dir, config)


if __name__ == "__main__":
    main()