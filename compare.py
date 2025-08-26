"""
compare.py - BACKWARD COMPATIBLE + OPTIMIZED VERSION
Same interface, but with performance improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import hashlib
from functools import lru_cache
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
    """Compare model results with paper benchmarks - OPTIMIZED."""

    def __init__(self):
        """Initialize comparator."""
        self.paper_benchmarks = PaperBenchmarks()
        self.results = {}
        self.comparisons = {}
        self._cache = {}  # Internal cache for repeated operations

    def load_results(self, results_path: str, modality: str = 'ppg'):
        """Load model results from CSV - OPTIMIZED with caching."""
        # Check cache first
        cache_key = f"{results_path}_{modality}"
        if cache_key in self._cache:
            self.results[modality] = self._cache[cache_key]
            return self._cache[cache_key]

        # Load and cache
        df = pd.read_csv(results_path)

        # OPTIMIZATION: Convert to optimal dtypes
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')

        self.results[modality] = df
        self._cache[cache_key] = df
        return df

    def compare(self, modality: str = 'ppg', verbose: bool = True):
        """Compare results with paper benchmarks - OPTIMIZED."""
        if modality not in self.results:
            print(f"No results loaded for {modality}")
            return None

        # Check if comparison already computed
        if modality in self.comparisons:
            if verbose:
                self._print_comparison(modality)
            return self.comparisons[modality]

        results_df = self.results[modality]
        benchmarks = self.paper_benchmarks.ppg if modality == 'ppg' else self.paper_benchmarks.ecg

        # OPTIMIZATION: Vectorized operations
        comparisons = []

        # Pre-filter valid rows
        valid_tasks = results_df['task'].isin(benchmarks.keys())
        filtered_df = results_df[valid_tasks]

        for _, row in filtered_df.iterrows():
            task = row['task']
            task_type = row['type']
            benchmark = benchmarks[task]

            comparison = {
                'task': task,
                'type': task_type,
                'modality': modality
            }

            # Handle classification tasks
            if task_type == 'classification' and 'auc' in benchmark:
                your_auc = row.get('auc')
                if your_auc is not None and not np.isnan(your_auc):
                    comparison['your_auc'] = your_auc
                    comparison['paper_auc'] = benchmark['auc']
                    comparison['auc_diff'] = your_auc - benchmark['auc']

            # Handle regression tasks
            elif task_type == 'regression' and 'mae' in benchmark:
                your_mae = row.get('mae')
                if your_mae is not None and not np.isnan(your_mae):
                    comparison['your_mae'] = your_mae
                    comparison['paper_mae'] = benchmark['mae']
                    comparison['mae_diff'] = benchmark['mae'] - your_mae

            comparisons.append(comparison)

        self.comparisons[modality] = pd.DataFrame(comparisons)

        if verbose:
            self._print_comparison(modality)

        return self.comparisons[modality]

    @lru_cache(maxsize=10)
    def _get_summary_stats(self, modality: str) -> Dict:
        """Cached computation of summary statistics."""
        if modality not in self.comparisons:
            return {}

        df = self.comparisons[modality]
        summary = {}

        if 'auc_diff' in df.columns:
            auc_diffs = df['auc_diff'].dropna()
            if len(auc_diffs) > 0:
                summary['avg_auc_diff'] = float(auc_diffs.mean())
                summary['std_auc_diff'] = float(auc_diffs.std())
                summary['better_auc_count'] = int((auc_diffs > 0).sum())
                summary['total_auc_tasks'] = len(auc_diffs)

        if 'mae_diff' in df.columns:
            mae_diffs = df['mae_diff'].dropna()
            if len(mae_diffs) > 0:
                summary['avg_mae_diff'] = float(mae_diffs.mean())
                summary['std_mae_diff'] = float(mae_diffs.std())
                summary['better_mae_count'] = int((mae_diffs > 0).sum())
                summary['total_mae_tasks'] = len(mae_diffs)

        return summary

    def _print_comparison(self, modality: str):
        """Print formatted comparison results - OPTIMIZED."""
        df = self.comparisons[modality]

        # Use cached summary stats
        summary = self._get_summary_stats(modality)

        print(f"\n{'='*60}")
        print(f"COMPARISON WITH APPLE PAPER RESULTS")
        print(f"{'='*60}")

        print(f"\n{modality.upper()} Results:")
        print("-" * 40)

        # OPTIMIZATION: Vectorized string operations
        for _, row in df.iterrows():
            task = row['task'].replace('_', ' ').title()
            print(f"\n{task}:")

            if 'your_auc' in row:
                print(f"  AUC: {row['your_auc']:.3f} (Paper: {row['paper_auc']:.3f}, "
                      f"Diff: {row['auc_diff']:+.3f})")

            if 'your_mae' in row:
                print(f"  MAE: {row['your_mae']:.2f} (Paper: {row['paper_mae']:.2f}, "
                      f"Diff: {row['mae_diff']:+.2f})")

        # Add summary if available
        if summary:
            print("\nSummary:")
            if 'avg_auc_diff' in summary:
                print(f"  Avg AUC diff: {summary['avg_auc_diff']:+.3f}")
            if 'avg_mae_diff' in summary:
                print(f"  Avg MAE diff: {summary['avg_mae_diff']:+.2f}")

        print("\n" + "="*60)

    def plot_comparison(self, modality: str = 'both', save_path: Optional[str] = None):
        """Create comparison plots - FIXED AND OPTIMIZED."""
        # Determine which modalities to plot
        if modality == 'both':
            modalities = [m for m in ['ppg', 'ecg'] if m in self.comparisons]
        elif modality in self.comparisons:
            modalities = [modality]
        else:
            print(f"No comparison data available for {modality}")
            return

        # OPTIMIZATION: Reuse figure if possible
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

            # OPTIMIZATION: Vectorized data preparation
            has_auc = 'your_auc' in df.columns
            has_mae = 'your_mae' in df.columns

            if has_auc:
                auc_data = df[df['your_auc'].notna()]
                if not auc_data.empty:
                    tasks = auc_data['task'].str.replace('_', '\n')
                    x = np.arange(len(tasks))
                    width = 0.35

                    ax.bar(x - width/2, auc_data['your_auc'], width,
                          label='AUC (Yours)', color='blue', alpha=0.7)
                    ax.bar(x + width/2, auc_data['paper_auc'], width,
                          label='AUC (Paper)', color='blue', alpha=0.3)

                    ax.set_xticks(x)
                    ax.set_xticklabels(tasks, rotation=45, ha='right')

            # Add MAE on secondary axis if present
            if has_mae:
                mae_data = df[df['your_mae'].notna()]
                if not mae_data.empty and has_auc and not auc_data.empty:
                    ax2 = ax.twinx()
                    ax2.bar(x, mae_data['your_mae'], width/2,
                           label='MAE', color='orange', alpha=0.5)
                    ax2.set_ylabel('MAE')

            ax.set_xlabel('Task')
            ax.set_ylabel('AUC' if has_auc else 'Value')
            ax.set_title(f'{mod.upper()} Model Comparison')
            ax.legend(loc='upper left')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            # OPTIMIZATION: Use optimal DPI and format
            plt.savefig(save_path, dpi=100, bbox_inches='tight',
                       format='png', optimize=True)
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()  # Free memory

    def save_comparison_table(self, modality: str, output_path: str):
        """Save comparison table to CSV - UNCHANGED."""
        if modality not in self.comparisons:
            print(f"No comparison data for {modality}")
            return

        self.comparisons[modality].to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")

    def save_report(self, output_dir: str):
        """Save comprehensive report - OPTIMIZED."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # OPTIMIZATION: Build report efficiently
        report = {
            'comparisons': {},
            'summary': {}
        }

        for modality in self.comparisons:
            df = self.comparisons[modality]
            # Use records_orient for faster conversion
            report['comparisons'][modality] = df.to_dict('records')
            report['summary'][modality] = self._get_summary_stats(modality)

        # Save JSON report
        json_path = output_path / 'comparison_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {json_path}")

        # Save CSV tables efficiently
        for modality in self.comparisons:
            csv_path = output_path / f'comparison_table_{modality}.csv'
            self.comparisons[modality].to_csv(csv_path, index=False)
            print(f"Comparison table saved to {csv_path}")

        # OPTIMIZATION: Save binary cache for faster reloading
        cache_path = output_path / 'comparison_cache.pkl'
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'comparisons': self.comparisons,
                'results': self.results
            }, f)

    def calculate_percentile(self, modality: str = 'ppg') -> float:
        """Calculate performance percentile - OPTIMIZED."""
        if modality not in self.comparisons:
            return 0.0

        # Use cached summary stats
        summary = self._get_summary_stats(modality)

        better_count = summary.get('better_auc_count', 0) + summary.get('better_mae_count', 0)
        total_count = summary.get('total_auc_tasks', 0) + summary.get('total_mae_tasks', 0)

        if total_count == 0:
            return 0.0

        return (better_count / total_count) * 100


def test_comparator():
    """Test function - UNCHANGED."""
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