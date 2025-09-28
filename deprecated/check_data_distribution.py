# check_data_distribution.py
"""
Analyze BUT PPG dataset distribution for demographics and class balance.
Fixed version with proper overlap checking and visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import your dataset
from data import BUTPPGDataset
from config_loader import get_config


def analyze_data_distribution():
    """Analyze demographic distribution across splits."""

    config = get_config()

    # Store results
    results = {
        'train': {'participants': [], 'ages': [], 'sex': [], 'bmi': []},
        'val': {'participants': [], 'ages': [], 'sex': [], 'bmi': []},
        'test': {'participants': [], 'ages': [], 'sex': [], 'bmi': []}
    }

    print("=" * 70)
    print("BUT PPG DATASET DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Analyze each split
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} SET:")
        print("-" * 40)

        # Create dataset
        dataset = BUTPPGDataset(
            data_dir=config.data_dir,
            modality='ppg',
            split=split,
            return_labels=True,
            return_participant_id=True
        )

        # Get the ACTUAL split participants (not all participants)
        split_participants = dataset.split_participants
        results[split]['participants'] = split_participants

        print(f"Participants: {len(split_participants)}")
        print(f"Total recordings: {len(dataset.split_records)}")
        print(f"Pairs created: {len(dataset.segment_pairs)}")
        if len(split_participants) > 0:
            print(f"Avg recordings per participant: {len(dataset.split_records) / len(split_participants):.1f}")

        # Collect demographics
        ages = []
        sexes = []
        bmis = []

        for pid in split_participants:
            info = dataset._get_participant_info(pid)
            if info['age'] > 0:
                ages.append(info['age'])
            if info['sex'] >= 0:
                sexes.append(info['sex'])
            if info['bmi'] > 0:
                bmis.append(info['bmi'])

        results[split]['ages'] = ages
        results[split]['sex'] = sexes
        results[split]['bmi'] = bmis

        # Age distribution
        if ages:
            print(f"\nAge statistics:")
            print(f"  Range: {min(ages):.0f} - {max(ages):.0f}")
            print(f"  Mean: {np.mean(ages):.1f} ± {np.std(ages):.1f}")
            print(f"  Median: {np.median(ages):.0f}")

            # Age classification (threshold=50)
            age_over_50 = sum(1 for a in ages if a >= 50)
            print(f"  Age >= 50: {age_over_50}/{len(ages)} ({100 * age_over_50 / len(ages):.1f}%)")

        # Sex distribution
        if sexes:
            print(f"\nSex distribution:")
            sex_counts = Counter(sexes)
            print(f"  Male (1): {sex_counts.get(1, 0)}")
            print(f"  Female (0): {sex_counts.get(0, 0)}")
            if len(sex_counts) > 1 and sex_counts[0] > 0:
                ratio = sex_counts[1] / sex_counts[0]
                print(f"  M/F ratio: {ratio:.2f}")

        # BMI distribution
        if bmis:
            print(f"\nBMI statistics:")
            print(f"  Range: {min(bmis):.1f} - {max(bmis):.1f}")
            print(f"  Mean: {np.mean(bmis):.1f} ± {np.std(bmis):.1f}")
            print(f"  Median: {np.median(bmis):.1f}")

            # BMI classifications
            bmi_25 = sum(1 for b in bmis if b >= 25)
            bmi_30 = sum(1 for b in bmis if b >= 30)
            print(f"  BMI >= 25: {bmi_25}/{len(bmis)} ({100 * bmi_25 / len(bmis):.1f}%)")
            print(f"  BMI >= 30: {bmi_30}/{len(bmis)} ({100 * bmi_30 / len(bmis):.1f}%)")

    # Cross-split analysis
    print("\n" + "=" * 70)
    print("CROSS-SPLIT ANALYSIS")
    print("=" * 70)

    # Check for ACTUAL overlap using split_participants
    train_set = set(results['train']['participants'])
    val_set = set(results['val']['participants'])
    test_set = set(results['test']['participants'])

    print(f"\nParticipant overlap check:")
    print(f"  Train participants: {len(train_set)}")
    print(f"  Val participants: {len(val_set)}")
    print(f"  Test participants: {len(test_set)}")
    print(f"  Total unique: {len(train_set | val_set | test_set)}")
    print(f"\nOverlap:")
    print(f"  Train ∩ Val: {len(train_set & val_set)} participants")
    print(f"  Train ∩ Test: {len(train_set & test_set)} participants")
    print(f"  Val ∩ Test: {len(val_set & test_set)} participants")

    if len(train_set & val_set) > 0 or len(train_set & test_set) > 0:
        print("  ⚠️ WARNING: Data leakage detected!")
        print("  Overlapping participants:")
        if train_set & test_set:
            print(f"    Train-Test: {train_set & test_set}")
    else:
        print("  ✓ No participant overlap - splits are clean")

    # Statistical tests for distribution differences
    print("\nDistribution balance check:")

    # Age balance
    if results['train']['ages'] and results['test']['ages']:
        from scipy import stats
        _, p_age = stats.ttest_ind(results['train']['ages'], results['test']['ages'])
        print(f"  Age distribution (t-test): p={p_age:.3f}")
        if p_age < 0.05:
            print("    ⚠️ Significant age difference between train and test")

    # Sex balance
    if results['train']['sex'] and results['test']['sex']:
        train_male_ratio = sum(results['train']['sex']) / len(results['train']['sex'])
        test_male_ratio = sum(results['test']['sex']) / len(results['test']['sex'])
        print(f"  Train male ratio: {train_male_ratio:.2f}")
        print(f"  Test male ratio: {test_male_ratio:.2f}")
        if abs(train_male_ratio - test_male_ratio) > 0.2:
            print("    ⚠️ Sex distribution imbalance")

    # BMI balance
    if results['train']['bmi'] and results['test']['bmi']:
        _, p_bmi = stats.ttest_ind(results['train']['bmi'], results['test']['bmi'])
        print(f"  BMI distribution (t-test): p={p_bmi:.3f}")
        if p_bmi < 0.05:
            print("    ⚠️ Significant BMI difference between train and test")

    # Visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('BUT PPG Dataset Distribution Analysis', fontsize=16)

    # Age distributions
    ax = axes[0, 0]
    for split in ['train', 'val', 'test']:
        if results[split]['ages']:
            ax.hist(results[split]['ages'], alpha=0.5, label=split, bins=10)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Age Distribution')
    ax.legend()
    ax.axvline(x=50, color='r', linestyle='--', alpha=0.5)

    # Sex distribution
    ax = axes[0, 1]
    x_labels = []
    female_counts = []
    male_counts = []

    for split in ['train', 'val', 'test']:
        if results[split]['sex']:
            male_count = sum(results[split]['sex'])
            female_count = len(results[split]['sex']) - male_count
            x_labels.append(split)
            female_counts.append(female_count)
            male_counts.append(male_count)

    if x_labels:
        x = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x - width / 2, female_counts, width, label='Female', alpha=0.7)
        ax.bar(x + width / 2, male_counts, width, label='Male', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Count')
        ax.set_title('Sex Distribution')
        ax.legend()

    # BMI distributions
    ax = axes[0, 2]
    for split in ['train', 'val', 'test']:
        if results[split]['bmi']:
            ax.hist(results[split]['bmi'], alpha=0.5, label=split, bins=10)
    ax.set_xlabel('BMI')
    ax.set_ylabel('Count')
    ax.set_title('BMI Distribution')
    ax.legend()
    ax.axvline(x=25, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=30, color='r', linestyle='--', alpha=0.5)

    # Age classification threshold
    ax = axes[1, 0]
    x_pos = np.arange(3)
    under_50_counts = []
    over_50_counts = []

    for split in ['train', 'val', 'test']:
        ages = results[split]['ages']
        if ages:
            over_50 = sum(1 for a in ages if a >= 50)
            under_50 = len(ages) - over_50
            under_50_counts.append(under_50)
            over_50_counts.append(over_50)
        else:
            under_50_counts.append(0)
            over_50_counts.append(0)

    ax.bar(x_pos, under_50_counts, label='<50', alpha=0.7, color='blue')
    ax.bar(x_pos, over_50_counts, bottom=under_50_counts, label='≥50', alpha=0.7, color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['train', 'val', 'test'])
    ax.set_ylabel('Count')
    ax.set_title('Age Classification Split (threshold=50)')
    ax.legend()

    # BMI classification threshold 25
    ax = axes[1, 1]
    under_25_counts = []
    over_25_counts = []

    for split in ['train', 'val', 'test']:
        bmis = results[split]['bmi']
        if bmis:
            over_25 = sum(1 for b in bmis if b >= 25)
            under_25 = len(bmis) - over_25
            under_25_counts.append(under_25)
            over_25_counts.append(over_25)
        else:
            under_25_counts.append(0)
            over_25_counts.append(0)

    ax.bar(x_pos, under_25_counts, label='<25', alpha=0.7, color='green')
    ax.bar(x_pos, over_25_counts, bottom=under_25_counts, label='≥25', alpha=0.7, color='orange')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['train', 'val', 'test'])
    ax.set_ylabel('Count')
    ax.set_title('BMI Classification Split (threshold=25)')
    ax.legend()

    # BMI classification threshold 30
    ax = axes[1, 2]
    under_30_counts = []
    over_30_counts = []

    for split in ['train', 'val', 'test']:
        bmis = results[split]['bmi']
        if bmis:
            over_30 = sum(1 for b in bmis if b >= 30)
            under_30 = len(bmis) - over_30
            under_30_counts.append(under_30)
            over_30_counts.append(over_30)
        else:
            under_30_counts.append(0)
            over_30_counts.append(0)

    ax.bar(x_pos, under_30_counts, label='<30', alpha=0.7, color='green')
    ax.bar(x_pos, over_30_counts, bottom=under_30_counts, label='≥30', alpha=0.7, color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['train', 'val', 'test'])
    ax.set_ylabel('Count')
    ax.set_title('BMI Classification Split (threshold=30)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=150)
    print("Saved visualization to: data_distribution_analysis.png")
    plt.show()

    # Summary recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Check for issues and provide recommendations
    issues = []

    # Age balance
    if results['train']['ages'] and results['test']['ages']:
        train_over_50_pct = 100 * sum(1 for a in results['train']['ages'] if a >= 50) / len(results['train']['ages'])
        test_over_50_pct = 100 * sum(1 for a in results['test']['ages'] if a >= 50) / len(results['test']['ages'])
        if abs(train_over_50_pct - test_over_50_pct) > 20:
            issues.append(
                f"Age imbalance: Train has {train_over_50_pct:.1f}% over 50, Test has {test_over_50_pct:.1f}%")

    # Sex balance
    if results['train']['sex'] and results['test']['sex']:
        train_male_pct = 100 * sum(results['train']['sex']) / len(results['train']['sex'])
        test_male_pct = 100 * sum(results['test']['sex']) / len(results['test']['sex'])
        if abs(train_male_pct - test_male_pct) > 20:
            issues.append(f"Sex imbalance: Train is {train_male_pct:.0f}% male, Test is {test_male_pct:.0f}% male")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendations:")
        print("  1. Consider stratified splitting by age and sex")
        print("  2. Use data augmentation to balance classes")
        print("  3. Apply class weights during training")
        print("  4. Report metrics separately for each demographic group")
    else:
        print("✓ Dataset appears well-balanced across splits")

    return results


if __name__ == "__main__":
    results = analyze_data_distribution()