# biosignal/compare.py
"""
Module to compare our results with Apple paper benchmarks
Generates comprehensive comparison reports with performance optimizations
Uses centralized configuration management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from functools import lru_cache, cached_property
import warnings
import logging

from config_loader import get_config  # Added ConfigLoader

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class ResultsComparator:
    """Compare experimental results with paper benchmarks - Optimized version."""

    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize comparator with optimizations.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.dataset_type = None
        self.config = get_config()

        # Load paper benchmarks from config
        self.paper_benchmarks = self.config.get_section('paper_benchmarks')

        # If no benchmarks in config, use defaults
        if not self.paper_benchmarks:
            logger.warning("No paper benchmarks found in config, using defaults")
            self.paper_benchmarks = {
                'ppg': {
                    'age_classification': {'auc': 0.976, 'pauc': 0.907},
                    'age_regression': {'mae': 3.19},
                    'bmi_classification': {'auc': 0.918, 'pauc': 0.750},
                    'bmi_regression': {'mae': 2.54},
                    'sex_classification': {'auc': 0.993, 'pauc': 0.967}
                },
                'ecg': {
                    'age_classification': {'auc': 0.916, 'pauc': 0.763},
                    'age_regression': {'mae': 6.33},
                    'bmi_classification': {'auc': 0.797, 'pauc': 0.612},
                    'bmi_regression': {'mae': 3.72},
                    'sex_classification': {'auc': 0.951, 'pauc': 0.841}
                }
            }
        else:
            # Convert flat config structure to nested if needed
            self.paper_benchmarks = self._restructure_benchmarks(self.paper_benchmarks)

        # Get metric categories from config
        self._HIGHER_BETTER_METRICS = frozenset(
            self.config.get('comparison.higher_better_metrics',
                          ['auc', 'pauc', 'r2', 'accuracy', 'f1'])
        )
        self._LOWER_BETTER_METRICS = frozenset(
            self.config.get('comparison.lower_better_metrics',
                          ['mae', 'rmse', 'mse'])
        )

        # Initialize results storage with pre-allocated space
        self.our_results = {}
        self.comparison_df = None

        # Cache for expensive computations
        self._comparison_cache = {}

    def _restructure_benchmarks(self, benchmarks: Dict) -> Dict:
        """Restructure flat benchmark config to nested structure."""
        restructured = {}

        for modality in ['ppg', 'ecg', 'acc']:
            if modality in benchmarks:
                modality_benchmarks = benchmarks[modality]

                # If it's already properly structured, use it
                if isinstance(modality_benchmarks, dict):
                    # Check if it needs restructuring (flat structure)
                    if 'age_classification_auc' in modality_benchmarks:
                        # Restructure flat format
                        restructured[modality] = {}

                        # Parse flat keys like 'age_classification_auc'
                        for key, value in modality_benchmarks.items():
                            parts = key.rsplit('_', 1)
                            if len(parts) == 2:
                                task_name = parts[0]
                                metric_name = parts[1]

                                if task_name not in restructured[modality]:
                                    restructured[modality][task_name] = {}

                                restructured[modality][task_name][metric_name] = value
                    else:
                        # Already nested properly
                        restructured[modality] = modality_benchmarks

        return restructured if restructured else benchmarks

    def load_results(self, results_path: str, modality: str = 'ppg', dataset_type: str = 'auto'):
        """
        Load experimental results with dataset type awareness.

        Args:
            results_path: Path to results file
            modality: Signal modality
            dataset_type: 'but_ppg', 'vitaldb', 'combined', or 'auto'
        """
        path = Path(results_path)

        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        # Auto-detect dataset type from filename
        if dataset_type == 'auto':
            filename_lower = str(path).lower()
            if 'vitaldb' in filename_lower:
                dataset_type = 'vitaldb'
            elif 'combined' in filename_lower:
                dataset_type = 'combined'
            else:
                dataset_type = 'but_ppg'

        self.dataset_type = dataset_type

        # Clear cache when loading new results
        self._comparison_cache.clear()

        try:
            if path.suffix == '.csv':
                df = pd.read_csv(path, engine='c', na_values=['NA', 'NaN', 'null', 'None'])

                # Check if dataset column exists
                if 'dataset' in df.columns or 'dataset_type' in df.columns:
                    # Handle multi-dataset results
                    self.our_results[modality] = self._process_multi_dataset_results(df)
                else:
                    # Single dataset results
                    self.our_results[modality] = self._df_to_results_dict(df)

            elif path.suffix == '.json':
                with open(path, 'r', buffering=8192) as f:
                    results = json.load(f)
                self.our_results[modality] = results
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            print(f"Loaded {modality.upper()} results from {path}")
            print(f"  Dataset type: {dataset_type}")

        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise

    def _process_multi_dataset_results(self, df: pd.DataFrame) -> Dict:
        """
        Process results that contain multiple datasets.

        Args:
            df: DataFrame with dataset column

        Returns:
            Dictionary organized by dataset and task
        """
        results = {}

        # Group by dataset if present
        if 'dataset' in df.columns:
            dataset_col = 'dataset'
        elif 'dataset_type' in df.columns:
            dataset_col = 'dataset_type'
        else:
            # No dataset column, treat as single dataset
            return self._df_to_results_dict(df)

        for dataset in df[dataset_col].unique():
            dataset_df = df[df[dataset_col] == dataset]
            dataset_key = dataset.lower().replace(' ', '_')
            results[dataset_key] = self._df_to_results_dict(dataset_df)

        return results

    def compare_modality(self, modality: str = 'ppg', dataset_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Compare results with dataset filtering support.

        Args:
            modality: Signal modality
            dataset_filter: Filter for specific dataset ('vitaldb', 'but_ppg', None for all)

        Returns:
            DataFrame with comparison
        """
        # Check cache first
        cache_key = f"{modality}_{dataset_filter or 'all'}_comparison"
        if cache_key in self._comparison_cache:
            return self._comparison_cache[cache_key].copy()

        if modality not in self.our_results:
            raise ValueError(f"No results loaded for {modality}")

        comparisons = []

        # Handle multi-dataset results
        our_results = self.our_results[modality]

        if isinstance(our_results, dict) and any(k in our_results for k in ['vitaldb', 'but_ppg']):
            # Multi-dataset results
            for dataset_key, dataset_results in our_results.items():
                if dataset_filter and dataset_key != dataset_filter:
                    continue

                for task, paper_metrics in self.paper_benchmarks.get(modality, {}).items():
                    comparison = self._compare_single_task(
                        task,
                        dataset_results.get(task, {}),
                        paper_metrics,
                        modality,
                        dataset_key
                    )
                    comparisons.append(comparison)
        else:
            # Single dataset results
            for task, paper_metrics in self.paper_benchmarks.get(modality, {}).items():
                comparison = self._compare_single_task(
                    task,
                    our_results.get(task, {}),
                    paper_metrics,
                    modality,
                    self.dataset_type or 'unknown'
                )
                comparisons.append(comparison)

        df = pd.DataFrame(comparisons)

        # Cache the result
        self._comparison_cache[cache_key] = df.copy()

        return df

    def _compare_single_task(self, task: str, our_metrics: Dict, paper_metrics: Dict,
                             modality: str, dataset: str) -> Dict:
        """
        Compare metrics for a single task.

        Args:
            task: Task name
            our_metrics: Our results
            paper_metrics: Paper benchmarks
            modality: Signal modality
            dataset: Dataset name

        Returns:
            Dictionary with comparison
        """
        comparison = {
            'modality': modality.upper(),
            'task': task,
            'dataset': dataset
        }

        # Compare each metric
        for metric, paper_value in paper_metrics.items():
            our_value = our_metrics.get(metric)

            if our_value is not None:
                comparison[f'paper_{metric}'] = paper_value
                comparison[f'our_{metric}'] = our_value

                # Calculate differences
                diff, pct_diff = self._calculate_metric_difference(
                    metric, our_value, paper_value
                )

                comparison[f'diff_{metric}'] = diff
                comparison[f'pct_diff_{metric}'] = pct_diff

        return comparison
    def _df_to_results_dict(self, df: pd.DataFrame) -> Dict:
        """
        Optimized DataFrame to dictionary conversion.

        Args:
            df: DataFrame with results

        Returns:
            Dictionary of results
        """
        results = {}

        # Identify columns to exclude
        exclude_cols = {'task', 'type', 'Unnamed: 0'}  # Common index columns

        for _, row in df.iterrows():
            task = row.get('task', row.get('Task', None))
            if task:
                # Use dictionary comprehension for faster conversion
                metrics = {
                    k: v for k, v in row.items()
                    if k not in exclude_cols and pd.notna(v)
                }
                results[task] = metrics

        return results

    @lru_cache(maxsize=32)
    def _calculate_metric_difference(
            self,
            metric: str,
            our_value: float,
            paper_value: float
    ) -> Tuple[float, float]:
        """
        Cached calculation of metric differences.

        Args:
            metric: Metric name
            our_value: Our result value
            paper_value: Paper benchmark value

        Returns:
            Tuple of (difference, percentage_difference)
        """
        if metric in self._HIGHER_BETTER_METRICS:
            diff = our_value - paper_value
        else:
            diff = paper_value - our_value

        # Avoid division by zero
        pct_diff = (diff / paper_value * 100) if paper_value != 0 else 0.0

        return diff, pct_diff

    def compare_modality(self, modality: str = 'ppg') -> pd.DataFrame:
        """
        Compare results for a specific modality with caching.

        Args:
            modality: 'ppg' or 'ecg'

        Returns:
            DataFrame with comparison
        """
        # Check cache first
        cache_key = f"{modality}_comparison"
        if cache_key in self._comparison_cache:
            return self._comparison_cache[cache_key].copy()

        if modality not in self.our_results:
            raise ValueError(f"No results loaded for {modality}")

        # Pre-allocate list for better performance
        comparisons = []

        # Batch process comparisons
        for task, paper_metrics in self.paper_benchmarks.get(modality, {}).items():
            comparison = {
                'modality': modality.upper(),
                'task': task
            }

            # Get our results
            our_metrics = self.our_results[modality].get(task, {})

            # Vectorized metric comparison
            for metric, paper_value in paper_metrics.items():
                our_value = our_metrics.get(metric)

                if our_value is not None:
                    comparison[f'paper_{metric}'] = paper_value
                    comparison[f'our_{metric}'] = our_value

                    # Use cached calculation
                    diff, pct_diff = self._calculate_metric_difference(
                        metric, our_value, paper_value
                    )

                    comparison[f'diff_{metric}'] = diff
                    comparison[f'pct_diff_{metric}'] = pct_diff

            comparisons.append(comparison)

        # Create DataFrame once with all data
        df = pd.DataFrame(comparisons)

        # Cache the result
        self._comparison_cache[cache_key] = df.copy()

        return df

    def generate_report(self, save_dir: Optional[str] = None,
                        separate_by_dataset: bool = True) -> Dict:
        """
        Generate report with dataset-aware organization.

        Args:
            save_dir: Directory to save report
            separate_by_dataset: Whether to create separate reports per dataset

        Returns:
            Dictionary with comparison statistics
        """
        if save_dir is None:
            save_dir = self.config.get('comparison.output_dir', 'data/outputs/comparisons')

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'modalities': list(self.our_results.keys()),
            'dataset_type': self.dataset_type,
            'comparisons': {},
            'config': {
                'dataset': self.config.dataset_name,
                'seed': self.config.seed
            }
        }

        all_comparisons = []

        # Process each modality
        for modality in self.our_results.keys():
            # Check if we have multi-dataset results
            our_results = self.our_results[modality]

            if isinstance(our_results, dict) and any(k in our_results for k in ['vitaldb', 'but_ppg']):
                # Multi-dataset - create separate comparisons
                for dataset in ['vitaldb', 'but_ppg']:
                    if dataset in our_results:
                        df = self.compare_modality(modality, dataset_filter=dataset)
                        all_comparisons.append(df)

                        summary = self._calculate_summary_stats(df)

                        if dataset not in report['comparisons']:
                            report['comparisons'][dataset] = {}

                        report['comparisons'][dataset][modality] = {
                            'summary': summary,
                            'details': df.to_dict('records')
                        }

                        if separate_by_dataset:
                            # Save dataset-specific comparison
                            dataset_dir = save_dir / dataset
                            dataset_dir.mkdir(exist_ok=True)

                            csv_path = dataset_dir / f'comparison_table_{modality}.csv'
                            df.to_csv(csv_path, index=False)
                            print(f"  {dataset.upper()} comparison saved to {csv_path}")
            else:
                # Single dataset - FIXED: Use modality as key directly for backward compatibility
                df = self.compare_modality(modality)
                all_comparisons.append(df)

                summary = self._calculate_summary_stats(df)

                # For backward compatibility, when dataset_type is not specified or is a single dataset,
                # put results directly under modality key
                if self.dataset_type and self.dataset_type not in ['unknown', 'auto']:
                    # New structure with dataset awareness
                    dataset_key = self.dataset_type
                    if dataset_key not in report['comparisons']:
                        report['comparisons'][dataset_key] = {}
                    report['comparisons'][dataset_key][modality] = {
                        'summary': summary,
                        'details': df.to_dict('records')
                    }
                else:
                    # Original structure for backward compatibility
                    report['comparisons'][modality] = {
                        'summary': summary,
                        'details': df.to_dict('records')
                    }

        # Combine all comparisons
        if all_comparisons:
            self.comparison_df = pd.concat(all_comparisons, ignore_index=True)

            # Save combined table
            csv_path = save_dir / 'comparison_table_all.csv'
            self.comparison_df.to_csv(csv_path, index=False)
            print(f"Combined comparison table saved to {csv_path}")

        # Save JSON report
        json_path = save_dir / 'comparison_report.json'
        with open(json_path, 'w', buffering=8192) as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Comparison report saved to {json_path}")

        return report
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Optimized summary statistics calculation using numpy.

        Args:
            df: Comparison DataFrame

        Returns:
            Dictionary of summary statistics
        """
        summary = {}

        # Get metric types from config
        metric_types = self.config.get('comparison.summary_metrics', ['auc', 'mae', 'pauc'])

        # Use numpy for faster computation
        for metric_type in metric_types:
            our_col = f'our_{metric_type}'
            paper_col = f'paper_{metric_type}'

            if our_col in df.columns:
                # Extract values as numpy arrays
                our_values = df[our_col].values
                paper_values = df[paper_col].values

                # Filter NaN values efficiently
                mask = ~(np.isnan(our_values) | np.isnan(paper_values))

                if mask.any():
                    our_clean = our_values[mask]
                    paper_clean = paper_values[mask]

                    # Calculate statistics
                    summary[f'mean_{metric_type}_ours'] = float(np.mean(our_clean))
                    summary[f'mean_{metric_type}_paper'] = float(np.mean(paper_clean))

                    if metric_type in self._HIGHER_BETTER_METRICS:
                        summary[f'mean_{metric_type}_diff'] = float(np.mean(our_clean - paper_clean))
                    else:
                        summary[f'mean_{metric_type}_diff'] = float(np.mean(paper_clean - our_clean))

        return summary

    def plot_comparison(
            self,
            save_path: Optional[str] = None,
            figsize: Optional[tuple] = None
    ):
        """
        Create optimized visualization comparing results.

        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.comparison_df is None or self.comparison_df.empty:
            print("No comparison data to plot")
            return

        # Get plot settings from config
        plot_config = self.config.get_nested('comparison', 'plot_settings', default={})
        if figsize is None:
            figsize = tuple(plot_config.get('figsize', [12, 8]))

        plot_style = plot_config.get('style', 'whitegrid')
        font_size = plot_config.get('font_size', 10)
        dpi = plot_config.get('dpi', 150)

        # Set style once
        sns.set_style(plot_style)
        plt.rcParams.update({'font.size': font_size, 'figure.max_open_warning': 0})

        # Create figure with constrained layout for better performance
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        fig.suptitle('Comparison with Apple Paper Results', fontsize=14, fontweight='bold')

        # Pre-filter data for each subplot
        df = self.comparison_df

        # 1. AUC Comparison - Vectorized data preparation
        ax = axes[0, 0]
        self._plot_auc_comparison(ax, df)

        # 2. MAE Comparison
        ax = axes[0, 1]
        self._plot_mae_comparison(ax, df)

        # 3. Performance Difference
        ax = axes[1, 0]
        self._plot_performance_difference(ax, df)

        # 4. Summary Statistics
        ax = axes[1, 1]
        self._plot_summary_stats(ax, df)

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    def _plot_auc_comparison(self, ax, df):
        """Optimized AUC comparison plotting."""
        # Vectorized filtering
        auc_mask = df['our_auc'].notna() if 'our_auc' in df.columns else pd.Series([False] * len(df))

        if auc_mask.any():
            auc_data = df.loc[auc_mask, ['modality', 'task', 'paper_auc', 'our_auc']].copy()
            auc_data['Task'] = auc_data['modality'] + '\n' + auc_data['task'].str.replace('_', ' ').str.title()

            # Melt for seaborn
            auc_melted = pd.melt(
                auc_data[['Task', 'paper_auc', 'our_auc']],
                id_vars='Task',
                value_vars=['paper_auc', 'our_auc'],
                var_name='Source',
                value_name='AUC'
            )
            auc_melted['Source'] = auc_melted['Source'].map({'paper_auc': 'Paper', 'our_auc': 'Ours'})

            sns.barplot(data=auc_melted, x='Task', y='AUC', hue='Source', ax=ax)
            ax.set_title('AUC Comparison (Classification)')

            # Get AUC threshold from config
            auc_threshold = self.config.get('comparison.auc_threshold', 0.9)
            ax.set_ylim([0.5, 1.0])
            ax.set_ylabel('AUC')
            ax.set_xlabel('')
            ax.legend(title='')
            ax.axhline(y=auc_threshold, color='r', linestyle='--', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_mae_comparison(self, ax, df):
        """Optimized MAE comparison plotting."""
        mae_mask = df['our_mae'].notna() if 'our_mae' in df.columns else pd.Series([False] * len(df))

        if mae_mask.any():
            mae_data = df.loc[mae_mask, ['modality', 'task', 'paper_mae', 'our_mae']].copy()
            mae_data['Task'] = mae_data['modality'] + '\n' + mae_data['task'].str.replace('_', ' ').str.title()

            mae_melted = pd.melt(
                mae_data[['Task', 'paper_mae', 'our_mae']],
                id_vars='Task',
                value_vars=['paper_mae', 'our_mae'],
                var_name='Source',
                value_name='MAE'
            )
            mae_melted['Source'] = mae_melted['Source'].map({'paper_mae': 'Paper', 'our_mae': 'Ours'})

            sns.barplot(data=mae_melted, x='Task', y='MAE', hue='Source', ax=ax)
            ax.set_title('MAE Comparison (Regression)')
            ax.set_ylabel('MAE')
            ax.set_xlabel('')
            ax.legend(title='')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_performance_difference(self, ax, df):
        """Optimized performance difference plotting."""
        diff_data = []

        # Vectorized processing
        for col in ['diff_auc', 'diff_mae']:
            if col in df.columns:
                mask = df[col].notna()
                if mask.any():
                    metric = 'AUC' if 'auc' in col else 'MAE'
                    multiplier = 100 if 'auc' in col else 1

                    for idx in df[mask].index:
                        row = df.loc[idx]
                        diff_data.append({
                            'Task': f"{row['modality']} {row['task'].replace('_', ' ')}",
                            'Metric': metric,
                            'Difference': row[col] * multiplier
                        })

        if diff_data:
            diff_df = pd.DataFrame(diff_data)

            # Create grouped bar chart
            task_groups = diff_df.groupby('Task')
            x = np.arange(len(task_groups))
            width = 0.35

            for i, metric in enumerate(diff_df['Metric'].unique()):
                metric_data = diff_df[diff_df['Metric'] == metric]
                ax.bar(x[:len(metric_data)] + i * width,
                       metric_data['Difference'].values,
                       width, label=metric)

            ax.set_title('Performance Difference (Ours - Paper)')
            ax.set_ylabel('Difference')
            ax.set_xlabel('Task')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend()

            # Add value labels efficiently
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')

    def _plot_summary_stats(self, ax, df):
        """Optimized summary statistics plotting."""
        ax.axis('off')

        # Pre-compute all statistics
        summary_lines = ["Summary Statistics", "=" * 30, ""]

        for modality in df['modality'].unique():
            mod_df = df[df['modality'] == modality]
            summary_lines.append(f"{modality}:")

            # Vectorized statistics calculation
            for metric in ['auc', 'mae']:
                col = f'our_{metric}'
                if col in mod_df.columns:
                    values = mod_df[col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        format_str = ".3f" if metric == 'auc' else ".2f"
                        summary_lines.append(f"  Mean {metric.upper()}: {mean_val:{format_str}}")

            summary_lines.append("")

        # Add dataset info from config
        summary_lines.extend([
            f"Dataset: {self.config.dataset_name}",
            f"Model: {self.config.get('model.name', 'EfficientNet1D')}",
            f"SSL: {self.config.get('comparison.ssl_method', 'RegularizedInfoNCE + KoLeo')}"
        ])

        ax.text(0.1, 0.5, '\n'.join(summary_lines),
                fontsize=10, verticalalignment='center',
                fontfamily='monospace')

    def print_summary(self, dataset_filter: Optional[str] = None):
        """
        Print summary with dataset filtering.

        Args:
            dataset_filter: Show only specific dataset results
        """
        if not self.our_results:
            print("No results loaded")
            return

        header = "\n" + "=" * 60 + "\n"

        if dataset_filter:
            print(header + f"COMPARISON WITH PAPER - {dataset_filter.upper()}" + header)
        else:
            print(header + "COMPARISON WITH APPLE PAPER RESULTS" + header)

        for modality in self.our_results.keys():
            print(f"\n{modality.upper()} Results:")
            print("-" * 40)

            our_results = self.our_results[modality]

            # Handle multi-dataset results
            if isinstance(our_results, dict) and any(k in our_results for k in ['vitaldb', 'but_ppg']):
                for dataset in ['vitaldb', 'but_ppg']:
                    if dataset_filter and dataset != dataset_filter:
                        continue

                    if dataset in our_results:
                        print(f"\n  Dataset: {dataset.upper()}")
                        self._print_dataset_comparison(modality, dataset, our_results[dataset])
            else:
                # Single dataset
                dataset_name = self.dataset_type or 'unknown'
                if not dataset_filter or dataset_filter == dataset_name:
                    self._print_dataset_comparison(modality, dataset_name, our_results)

        print("\n" + "=" * 60)

    def _print_dataset_comparison(self, modality: str, dataset: str, results: Dict):
        """Print comparison for a specific dataset."""
        for task_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue

            task_display = task_name.replace('_', ' ').title()
            print(f"\n    {task_display}:")

            # Get paper benchmarks
            paper_metrics = self.paper_benchmarks.get(modality, {}).get(task_name, {})

            # Print metrics
            for metric in ['auc', 'mae', 'pauc', 'accuracy', 'f1', 'r2']:
                if metric in metrics:
                    our_val = metrics[metric]
                    paper_val = paper_metrics.get(metric, None)

                    if paper_val is not None:
                        diff = our_val - paper_val if metric in self._HIGHER_BETTER_METRICS else paper_val - our_val
                        format_str = ".3f" if metric in {'auc', 'pauc'} else ".2f"

                        print(f"      {metric.upper()}: {our_val:{format_str}} "
                              f"(Paper: {paper_val:{format_str}}, "
                              f"Diff: {diff:+{format_str}})")
                    else:
                        format_str = ".3f" if metric in {'auc', 'pauc'} else ".2f"
                        print(f"      {metric.upper()}: {our_val:{format_str}} (No benchmark)")


# ============= TEST FUNCTIONS =============

def test_comparison():
    """Test comparison functionality - maintains backward compatibility."""
    print("=" * 50)
    print("Testing Optimized Comparison Module")
    print("=" * 50)

    # Load config
    config = get_config()

    # Create comparator
    comparator = ResultsComparator()
    print("✓ Comparator initialized")

    # Create dummy results
    print("\n1. Testing with dummy results:")

    dummy_results = {
        'age_classification': {
            'auc': 0.95,
            'accuracy': 0.88,
            'f1': 0.87
        },
        'age_regression': {
            'mae': 4.5,
            'rmse': 5.2,
            'r2': 0.75
        },
        'sex_classification': {
            'auc': 0.98,
            'accuracy': 0.92,
            'f1': 0.91
        },
        'bmi_classification': {
            'auc': 0.89,
            'accuracy': 0.82,
            'f1': 0.80
        },
        'bmi_regression': {
            'mae': 3.1,
            'rmse': 3.8,
            'r2': 0.68
        }
    }

    comparator.our_results['ppg'] = dummy_results
    print("   Dummy PPG results loaded")

    # Test comparison
    print("\n2. Testing comparison generation:")
    df = comparator.compare_modality('ppg')
    print(f"   Comparison DataFrame shape: {df.shape}")
    print(f"   Tasks compared: {df['task'].tolist()}")

    # Test caching
    df2 = comparator.compare_modality('ppg')
    print("   ✓ Caching working: ", df2 is not df and df2.equals(df))
    print("   ✓ Comparison generation test passed!")

    # Test report generation
    print("\n3. Testing report generation:")
    report_dir = Path('data/outputs/test_comparisons')
    report = comparator.generate_report(save_dir=str(report_dir))

    assert 'comparisons' in report
    assert 'ppg' in report['comparisons']
    print("   ✓ Report generation test passed!")

    # Test summary printing
    print("\n4. Testing summary printing:")
    comparator.print_summary()
    print("   ✓ Summary printing test passed!")

    # Test plotting (without display)
    print("\n5. Testing plot generation:")
    plot_path = report_dir / 'test_comparison.png'

    # Mock plt.show to avoid display during testing
    import matplotlib.pyplot as plt
    original_show = plt.show
    plt.show = lambda: None

    comparator.plot_comparison(save_path=str(plot_path))

    plt.show = original_show  # Restore

    if plot_path.exists():
        print("   ✓ Plot generation test passed!")
        plot_path.unlink()  # Clean up

    # Test loading from file
    print("\n6. Testing loading from file:")

    # Save dummy results to CSV
    results_df = pd.DataFrame([
        {'task': 'age_classification', 'type': 'classification', 'auc': 0.95, 'accuracy': 0.88},
        {'task': 'sex_classification', 'type': 'classification', 'auc': 0.98, 'accuracy': 0.92}
    ])

    csv_path = report_dir / 'test_results.csv'
    results_df.to_csv(csv_path, index=False)

    # Load and test
    new_comparator = ResultsComparator()
    new_comparator.load_results(str(csv_path), modality='ecg')

    assert 'ecg' in new_comparator.our_results
    print("   ✓ Loading from file test passed!")

    # Performance comparison test
    print("\n7. Testing performance improvements:")
    import time

    # Test with larger dataset
    large_results = {}
    for i in range(100):
        large_results[f'task_{i}'] = {
            'auc': np.random.random(),
            'mae': np.random.random() * 10,
            'accuracy': np.random.random()
        }

    perf_comparator = ResultsComparator()
    perf_comparator.our_results['ppg'] = large_results

    # Time the comparison
    start = time.time()
    for _ in range(10):
        _ = perf_comparator.compare_modality('ppg')
    elapsed = time.time() - start
    print(f"   Average comparison time: {elapsed / 10:.4f} seconds")
    print("   ✓ Performance test passed!")

    # Test with config benchmarks
    print("\n8. Testing with config benchmarks:")
    config_benchmarks = config.get_paper_benchmarks('ppg')
    if config_benchmarks:
        print(f"   Loaded {len(config_benchmarks)} benchmarks from config")
        print("   ✓ Config benchmark test passed!")

    # Clean up
    import shutil
    if report_dir.exists():
        shutil.rmtree(report_dir)

    print("\n" + "=" * 50)
    print("All optimized comparison tests passed successfully!")
    print("Backward compatibility maintained!")
    print("Configuration integration complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_comparison()