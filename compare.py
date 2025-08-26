"""
compare.py - FULLY BACKWARD COMPATIBLE VERSION
No name changes, maintains exact original interface
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PaperBenchmarks:
    """Apple paper benchmark results for comparison."""

    def __init__(self):
        """Initialize with paper benchmarks."""
        self.ppg = {
            # Classification tasks (AUC)
            'age_classification': {'auc': 0.917, 'threshold': 50},
            'sex_classification': {'auc': 0.973},
            'bmi_classification': {'auc': 0.849, 'threshold': 30},
            'bp_classification': {'auc': 0.723, 'threshold': 140},

            # Regression tasks (MAE)
            'age_regression': {'mae': 6.42},
            'bmi_regression': {'mae': 3.67},
            'heart_rate_regression': {'mae': 7.23},
            'spo2_regression': {'mae': 1.82}
        }

        self.ecg = {
            # Classification tasks (AUC)
            'age_classification': {'auc': 0.916, 'threshold': 50},
            'sex_classification': {'auc': 0.951},
            'bmi_classification': {'auc': 0.797, 'threshold': 30},

            # Regression tasks (MAE)
            'age_regression': {'mae': 6.33},
            'bmi_regression': {'mae': 3.72},
            'heart_rate_regression': {'mae': 5.91}
        }


class ModelComparator:
    """Compare model results with paper benchmarks."""

    def __init__(self):
        """Initialize comparator."""
        self.paper_benchmarks = PaperBenchmarks()
        self.results = {}
        self.comparisons = {}

    def load_results(self, results_path: str, modality: str = 'ppg'):
        """Load model results from CSV."""
        df = pd.read_csv(results_path)
        self.results[modality] = df
        return df

    def compare(self, modality: str = 'ppg', verbose: bool = True):
        """Compare results with paper benchmarks."""
        if modality not in self.results:
            print(f"No results loaded for {modality}")
            return None

        results_df = self.results[modality]
        benchmarks = self.paper_benchmarks.ppg if modality == 'ppg' else self.paper_benchmarks.ecg

        comparisons = []

        for _, row in results_df.iterrows():
            task = row['task']
            task_type = row['type']

            if task not in benchmarks:
                continue

            benchmark = benchmarks[task]
            comparison = {
                'task': task,
                'type': task_type,
                'modality': modality
            }

            # Handle classification tasks
            if task_type == 'classification' and 'auc' in benchmark:
                your_auc = row['auc'] if 'auc' in row else None
                if your_auc is not None and not np.isnan(your_auc):
                    comparison['your_auc'] = your_auc
                    comparison['paper_auc'] = benchmark['auc']
                    comparison['auc_diff'] = your_auc - benchmark['auc']

            # Handle regression tasks
            elif task_type == 'regression' and 'mae' in benchmark:
                your_mae = row['mae'] if 'mae' in row else None
                if your_mae is not None and not np.isnan(your_mae):
                    comparison['your_mae'] = your_mae
                    comparison['paper_mae'] = benchmark['mae']
                    comparison['mae_diff'] = benchmark['mae'] - your_mae  # Lower is better

            comparisons.append(comparison)

        self.comparisons[modality] = pd.DataFrame(comparisons)

        if verbose:
            self._print_comparison(modality)

        return self.comparisons[modality]

    def _print_comparison(self, modality: str):
        """Print formatted comparison results."""
        df = self.comparisons[modality]

        print(f"\n{'='*60}")
        print(f"COMPARISON WITH APPLE PAPER RESULTS")
        print(f"{'='*60}")

        print(f"\n{modality.upper()} Results:")
        print("-" * 40)

        for _, row in df.iterrows():
            task = row['task'].replace('_', ' ').title()
            print(f"\n{task}:")

            if 'your_auc' in row:
                print(f"  AUC: {row['your_auc']:.3f} (Paper: {row['paper_auc']:.3f}, "
                      f"Diff: {row['auc_diff']:+.3f})")

            if 'your_mae' in row:
                print(f"  MAE: {row['your_mae']:.2f} (Paper: {row['paper_mae']:.2f}, "
                      f"Diff: {row['mae_diff']:+.2f})")

        print("\n" + "="*60)

    def plot_comparison(self, modality: str = 'both', save_path: Optional[str] = None):
        """Create comparison plots - FIXED to handle mismatched arrays."""
        # Determine which modalities to plot
        modalities = []
        if modality == 'both':
            modalities = [m for m in ['ppg', 'ecg'] if m in self.comparisons]
        elif modality in self.comparisons:
            modalities = [modality]

        if not modalities:
            print(f"No comparison data available for {modality}")
            return

        # Create figure
        n_modalities = len(modalities)
        fig, axes = plt.subplots(1, n_modalities, figsize=(8*n_modalities, 6))

        if n_modalities == 1:
            axes = [axes]

        for idx, mod in enumerate(modalities):
            ax = axes[idx]
            df = self.comparisons[mod]

            if df.empty:
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{mod.upper()} Comparison')
                continue

            # Collect all valid data points
            valid_tasks = []
            auc_yours = []
            auc_paper = []
            mae_yours = []
            mae_paper = []

            for _, row in df.iterrows():
                task_name = row['task'].replace('_', '\n')

                if 'your_auc' in row and pd.notna(row.get('your_auc')):
                    valid_tasks.append(task_name)
                    auc_yours.append(row['your_auc'])
                    auc_paper.append(row['paper_auc'])
                    # Add placeholder for MAE
                    mae_yours.append(0)
                    mae_paper.append(0)
                elif 'your_mae' in row and pd.notna(row.get('your_mae')):
                    valid_tasks.append(task_name)
                    mae_yours.append(row['your_mae'])
                    mae_paper.append(row['paper_mae'])
                    # Add placeholder for AUC
                    auc_yours.append(0)
                    auc_paper.append(0)

            if not valid_tasks:
                ax.text(0.5, 0.5, 'No valid metrics to plot',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{mod.upper()} Comparison')
                continue

            # Create bar positions
            x = np.arange(len(valid_tasks))
            width = 0.2

            # Plot bars - handle cases where some metrics might be zero
            if any(v != 0 for v in auc_yours):
                ax.bar(x - width*1.5, auc_yours, width, label='AUC (Yours)', color='blue', alpha=0.7)
                ax.bar(x - width*0.5, auc_paper, width, label='AUC (Paper)', color='blue', alpha=0.3)

            if any(v != 0 for v in mae_yours):
                # Scale MAE for visibility if needed
                mae_scale = 0.01  # Scale factor to make MAE visible with AUC
                ax.bar(x + width*0.5, np.array(mae_yours)*mae_scale, width,
                      label=f'MAE (Yours) x{mae_scale}', color='orange', alpha=0.7)
                ax.bar(x + width*1.5, np.array(mae_paper)*mae_scale, width,
                      label=f'MAE (Paper) x{mae_scale}', color='orange', alpha=0.3)

            # Customize plot
            ax.set_xlabel('Task')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'{mod.upper()} Model Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(valid_tasks, rotation=45, ha='right')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_comparison_table(self, modality: str, output_path: str):
        """Save comparison table to CSV."""
        if modality not in self.comparisons:
            print(f"No comparison data for {modality}")
            return

        self.comparisons[modality].to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")

    def save_report(self, output_dir: str):
        """Save comprehensive report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = {
            'comparisons': {},
            'summary': {}
        }

        for modality in self.comparisons:
            df = self.comparisons[modality]
            report['comparisons'][modality] = df.to_dict('records')

            # Calculate summary
            summary = {}
            if 'auc_diff' in df.columns:
                auc_diffs = df['auc_diff'].dropna()
                if len(auc_diffs) > 0:
                    summary['avg_auc_diff'] = auc_diffs.mean()
                    summary['better_auc_count'] = (auc_diffs > 0).sum()
                    summary['total_auc_tasks'] = len(auc_diffs)

            if 'mae_diff' in df.columns:
                mae_diffs = df['mae_diff'].dropna()
                if len(mae_diffs) > 0:
                    summary['avg_mae_diff'] = mae_diffs.mean()
                    summary['better_mae_count'] = (mae_diffs > 0).sum()
                    summary['total_mae_tasks'] = len(mae_diffs)

            report['summary'][modality] = summary

        # Save JSON report
        json_path = output_path / 'comparison_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {json_path}")

        # Save CSV tables
        for modality in self.comparisons:
            csv_path = output_path / f'comparison_table_{modality}.csv'
            self.comparisons[modality].to_csv(csv_path, index=False)
            print(f"Comparison table saved to {csv_path}")

    def calculate_percentile(self, modality: str = 'ppg') -> float:
        """Calculate performance percentile."""
        if modality not in self.comparisons:
            return 0.0

        df = self.comparisons[modality]

        better_count = 0
        total_count = 0

        if 'auc_diff' in df.columns:
            auc_diffs = df['auc_diff'].dropna()
            better_count += (auc_diffs > 0).sum()
            total_count += len(auc_diffs)

        if 'mae_diff' in df.columns:
            mae_diffs = df['mae_diff'].dropna()
            better_count += (mae_diffs > 0).sum()
            total_count += len(mae_diffs)

        if total_count == 0:
            return 0.0

        return (better_count / total_count) * 100


def test_comparator():
    """Test function."""
    print("Testing ModelComparator...")

    # Create test data
    test_results = pd.DataFrame({
        'task': ['age_classification', 'sex_classification', 'bmi_regression'],
        'type': ['classification', 'classification', 'regression'],
        'auc': [0.85, 0.45, np.nan],
        'mae': [np.nan, np.nan, 6.5],
        'accuracy': [0.8, 0.5, np.nan]
    })

    # Save test data
    test_path = 'test_results.csv'
    test_results.to_csv(test_path, index=False)

    # Test comparator
    comparator = ModelComparator()
    comparator.load_results(test_path, 'ecg')
    comparator.compare('ecg')

    # Try plotting
    try:
        comparator.plot_comparison('ecg')
        print("✅ Plotting test passed")
    except Exception as e:
        print(f"❌ Plotting failed: {e}")

    # Clean up
    Path(test_path).unlink()

    print("Testing complete!")


if __name__ == "__main__":
    test_comparator()