"""
compare.py - FULLY FIXED VERSION
Compare model results with paper benchmarks
Handles missing data, plotting issues, and provides comprehensive analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class PaperBenchmarks:
    """Apple paper benchmark results for comparison."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with paper benchmarks.

        Args:
            config_path: Optional path to YAML config with benchmarks
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                benchmarks = yaml.safe_load(f)
                self.ppg = benchmarks.get('ppg', self._get_default_ppg())
                self.ecg = benchmarks.get('ecg', self._get_default_ecg())
        else:
            self.ppg = self._get_default_ppg()
            self.ecg = self._get_default_ecg()

    def _get_default_ppg(self):
        """Default PPG benchmarks from paper."""
        return {
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

    def _get_default_ecg(self):
        """Default ECG benchmarks from paper."""
        return {
            # Classification tasks (AUC)
            'age_classification': {'auc': 0.916, 'threshold': 50},
            'sex_classification': {'auc': 0.951},
            'bmi_classification': {'auc': 0.797, 'threshold': 30},

            # Regression tasks (MAE)
            'age_regression': {'mae': 6.33},
            'bmi_regression': {'mae': 3.72},
            'heart_rate_regression': {'mae': 5.91}
        }

    def get(self, modality: str) -> Dict:
        """Get benchmarks for specific modality."""
        if modality.lower() == 'ppg':
            return self.ppg
        elif modality.lower() == 'ecg':
            return self.ecg
        else:
            raise ValueError(f"Unknown modality: {modality}")


class ModelComparator:
    """Compare model results with paper benchmarks - FIXED VERSION."""

    def __init__(self, paper_benchmarks: Optional[PaperBenchmarks] = None):
        """Initialize comparator.

        Args:
            paper_benchmarks: Paper benchmark instance
        """
        self.paper_benchmarks = paper_benchmarks or PaperBenchmarks()
        self.results = {}
        self.comparisons = {}

    def load_results(self, results_path: str, modality: str = 'ppg'):
        """Load model results from CSV - FIXED VERSION.

        Args:
            results_path: Path to results CSV
            modality: 'ppg' or 'ecg'
        """
        if not Path(results_path).exists():
            print(f"⚠️ Warning: Results file not found: {results_path}")
            return None

        df = pd.read_csv(results_path)

        # Filter out tasks with no valid data
        initial_count = len(df)
        df = df.dropna(subset=['auc', 'mae', 'accuracy'], how='all')

        if len(df) < initial_count:
            print(f"  Filtered {initial_count - len(df)} tasks with no valid metrics")

        # Warn about suspicious results
        if 'auc' in df.columns:
            zero_auc = df[df['auc'] == 0.0]['task'].tolist()
            if zero_auc:
                print(f"⚠️ Warning: AUC = 0.0 for tasks: {zero_auc}")
                print("  This suggests model collapse or insufficient data")

        # Store results
        self.results[modality] = df

        # Get available tasks
        available_tasks = set(df['task'].unique())
        expected_tasks = set(self.paper_benchmarks.get(modality).keys())

        # Report missing tasks
        missing = expected_tasks - available_tasks
        if missing:
            print(f"ℹ️ Info: No results for tasks: {missing}")
            print(f"  This is expected if dataset lacks these labels")

        return df

    def compare(self, modality: str = 'ppg', verbose: bool = True):
        """Compare results with paper benchmarks - FIXED VERSION.

        Args:
            modality: 'ppg' or 'ecg'
            verbose: Print detailed comparison
        """
        if modality not in self.results:
            print(f"No results loaded for {modality}")
            return None

        results_df = self.results[modality]
        benchmarks = self.paper_benchmarks.get(modality)

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
                your_auc = row.get('auc', np.nan)
                paper_auc = benchmark['auc']

                if not np.isnan(your_auc):
                    comparison['your_auc'] = your_auc
                    comparison['paper_auc'] = paper_auc
                    comparison['auc_diff'] = your_auc - paper_auc
                    comparison['auc_relative'] = self._calculate_relative_performance(
                        your_auc, paper_auc, metric='auc'
                    )

            # Handle regression tasks
            elif task_type == 'regression' and 'mae' in benchmark:
                your_mae = row.get('mae', np.nan)
                paper_mae = benchmark['mae']

                if not np.isnan(your_mae):
                    comparison['your_mae'] = your_mae
                    comparison['paper_mae'] = paper_mae
                    comparison['mae_diff'] = paper_mae - your_mae  # Lower is better
                    comparison['mae_relative'] = self._calculate_relative_performance(
                        your_mae, paper_mae, metric='mae'
                    )

            comparisons.append(comparison)

        self.comparisons[modality] = pd.DataFrame(comparisons)

        if verbose:
            self._print_comparison(modality)

        return self.comparisons[modality]

    def _calculate_relative_performance(self, your_value: float, paper_value: float,
                                        metric: str = 'auc') -> str:
        """Calculate relative performance with significance threshold.

        Args:
            your_value: Your model's metric
            paper_value: Paper's metric
            metric: Type of metric ('auc' or 'mae')

        Returns:
            String description of relative performance
        """
        if paper_value == 0:
            return "N/A"

        if metric == 'auc':
            # Higher is better for AUC
            relative_perf = (your_value / paper_value) * 100

            if 95 <= relative_perf <= 105:
                return "Comparable"
            elif relative_perf > 105:
                return f"+{relative_perf - 100:.1f}% better"
            else:
                return f"{relative_perf - 100:.1f}% worse"

        elif metric == 'mae':
            # Lower is better for MAE
            relative_perf = (your_value / paper_value) * 100

            if 95 <= relative_perf <= 105:
                return "Comparable"
            elif relative_perf < 95:
                return f"{100 - relative_perf:.1f}% better"
            else:
                return f"{relative_perf - 100:.1f}% worse"

    def _print_comparison(self, modality: str):
        """Print formatted comparison results."""
        df = self.comparisons[modality]

        print(f"\n{'=' * 60}")
        print(f"COMPARISON WITH APPLE PAPER RESULTS")
        print(f"{'=' * 60}")

        print(f"\n{modality.upper()} Results:")
        print("-" * 40)

        for _, row in df.iterrows():
            task = row['task'].replace('_', ' ').title()
            print(f"\n{task}:")

            if 'your_auc' in row and not np.isnan(row.get('your_auc')):
                print(f"  AUC: {row['your_auc']:.3f} (Paper: {row['paper_auc']:.3f}, "
                      f"Diff: {row['auc_diff']:+.3f})")
                print(f"  Relative: {row['auc_relative']}")

            if 'your_mae' in row and not np.isnan(row.get('your_mae')):
                print(f"  MAE: {row['your_mae']:.2f} (Paper: {row['paper_mae']:.2f}, "
                      f"Diff: {row['mae_diff']:+.2f})")
                print(f"  Relative: {row['mae_relative']}")

        print("\n" + "=" * 60)

    def plot_comparison(self, modality: str = 'both', save_path: Optional[str] = None):
        """Create comparison plots - FULLY FIXED VERSION.

        Args:
            modality: 'ppg', 'ecg', or 'both'
            save_path: Optional path to save plot
        """
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
        fig, axes = plt.subplots(1, n_modalities, figsize=(8 * n_modalities, 6))

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

            # Prepare data for plotting
            plot_data = []

            for _, row in df.iterrows():
                task_name = row['task'].replace('_', '\n')

                # Add AUC comparison if available
                if 'your_auc' in row and not np.isnan(row.get('your_auc')):
                    plot_data.append({
                        'Task': task_name,
                        'Metric': 'AUC',
                        'Yours': row['your_auc'],
                        'Paper': row['paper_auc'],
                        'Difference': row['auc_diff']
                    })

                # Add MAE comparison if available
                if 'your_mae' in row and not np.isnan(row.get('your_mae')):
                    plot_data.append({
                        'Task': task_name,
                        'Metric': 'MAE',
                        'Yours': row['your_mae'],
                        'Paper': row['paper_mae'],
                        'Difference': -row['mae_diff']  # Negative for consistency
                    })

            if not plot_data:
                ax.text(0.5, 0.5, 'No valid metrics to plot',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{mod.upper()} Comparison')
                continue

            # Convert to DataFrame for easier plotting
            plot_df = pd.DataFrame(plot_data)

            # Create grouped bar chart
            tasks = plot_df['Task'].unique()
            metrics = plot_df['Metric'].unique()

            x = np.arange(len(tasks))
            width = 0.35 / len(metrics)

            for i, metric in enumerate(metrics):
                metric_data = plot_df[plot_df['Metric'] == metric]

                # Ensure data aligns with task order
                yours_values = []
                paper_values = []

                for task in tasks:
                    task_metric_data = metric_data[metric_data['Task'] == task]
                    if not task_metric_data.empty:
                        yours_values.append(task_metric_data.iloc[0]['Yours'])
                        paper_values.append(task_metric_data.iloc[0]['Paper'])
                    else:
                        yours_values.append(0)
                        paper_values.append(0)

                # Plot bars
                offset = i * width * 2
                ax.bar(x + offset, yours_values, width,
                       label=f'{metric} (Yours)', alpha=0.8)
                ax.bar(x + offset + width, paper_values, width,
                       label=f'{metric} (Paper)', alpha=0.8)

            # Customize plot
            ax.set_xlabel('Task')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'{mod.upper()} Model Comparison')
            ax.set_xticks(x + width * len(metrics) / 2)
            ax.set_xticklabels(tasks, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def generate_report(self) -> Dict:
        """Generate comprehensive comparison report.

        Returns:
            Dictionary with summary, detailed comparisons, and recommendations
        """
        report = {
            'summary': {},
            'detailed_comparisons': {},
            'recommendations': []
        }

        # Generate summary for each modality
        for modality in self.comparisons:
            df = self.comparisons[modality]

            if df.empty:
                continue

            # Calculate summary statistics
            summary = {
                'total_tasks': len(df),
                'classification_tasks': len(df[df['type'] == 'classification']),
                'regression_tasks': len(df[df['type'] == 'regression'])
            }

            # AUC summary
            auc_cols = [col for col in df.columns if 'your_auc' in col]
            if auc_cols:
                auc_data = df[df['your_auc'].notna()]
                if not auc_data.empty:
                    summary['avg_auc'] = auc_data['your_auc'].mean()
                    summary['avg_paper_auc'] = auc_data['paper_auc'].mean()
                    summary['auc_performance'] = (
                            (auc_data['auc_diff'] > 0).sum() / len(auc_data) * 100
                    )

            # MAE summary
            mae_cols = [col for col in df.columns if 'your_mae' in col]
            if mae_cols:
                mae_data = df[df['your_mae'].notna()]
                if not mae_data.empty:
                    summary['avg_mae'] = mae_data['your_mae'].mean()
                    summary['avg_paper_mae'] = mae_data['paper_mae'].mean()
                    summary['mae_performance'] = (
                            (mae_data['mae_diff'] > 0).sum() / len(mae_data) * 100
                    )

            report['summary'][modality] = summary
            report['detailed_comparisons'][modality] = df.to_dict('records')

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        for modality, df in self.comparisons.items():
            if df.empty:
                continue

            # Check for model collapse
            if 'your_auc' in df.columns:
                poor_auc = df[df['your_auc'] < 0.6]
                if not poor_auc.empty:
                    tasks = poor_auc['task'].tolist()
                    recommendations.append(
                        f"⚠️ {modality.upper()}: Poor AUC (<0.6) for {tasks}. "
                        "Consider: 1) Check loss functions, 2) Increase temperature, "
                        "3) Decrease KoLeo weight, 4) Verify positive pairs"
                    )

                # Check for random performance
                random_auc = df[(df['your_auc'] >= 0.45) & (df['your_auc'] <= 0.55)]
                if not random_auc.empty:
                    tasks = random_auc['task'].tolist()
                    recommendations.append(
                        f"⚠️ {modality.upper()}: Random-level AUC for {tasks}. "
                        "Model may not be learning meaningful representations."
                    )

            # Check for high MAE
            if 'your_mae' in df.columns:
                high_mae = df[df['your_mae'] > df['paper_mae'] * 2]
                if not high_mae.empty:
                    tasks = high_mae['task'].tolist()
                    recommendations.append(
                        f"⚠️ {modality.upper()}: MAE >2x paper baseline for {tasks}. "
                        "Consider: 1) Longer training, 2) Data quality checks, "
                        "3) Architecture adjustments"
                    )

            # Check overall performance
            if 'auc_diff' in df.columns:
                avg_auc_diff = df['auc_diff'].mean()
                if avg_auc_diff < -0.1:
                    recommendations.append(
                        f"📊 {modality.upper()}: Average AUC {avg_auc_diff:.3f} below paper. "
                        "Consider: 1) Increase dataset size, 2) Adjust augmentations, "
                        "3) Fine-tune hyperparameters"
                    )
                elif avg_auc_diff > 0.05:
                    recommendations.append(
                        f"✅ {modality.upper()}: Good performance! Average AUC "
                        f"{avg_auc_diff:.3f} above paper baseline."
                    )

        # General recommendations
        if not recommendations:
            recommendations.append("ℹ️ No specific issues detected. Continue monitoring training.")

        return recommendations

    def save_report(self, output_dir: str):
        """Save comprehensive report to files.

        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate report
        report = self.generate_report()

        # Save JSON report
        json_path = output_path / 'comparison_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {json_path}")

        # Save comparison tables
        for modality, df in self.comparisons.items():
            csv_path = output_path / f'comparison_table_{modality}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Comparison table saved to {csv_path}")

        # Save recommendations
        rec_path = output_path / 'recommendations.txt'
        with open(rec_path, 'w') as f:
            f.write("MODEL COMPARISON RECOMMENDATIONS\n")
            f.write("=" * 60 + "\n\n")
            for rec in report['recommendations']:
                f.write(f"{rec}\n\n")
        print(f"Recommendations saved to {rec_path}")

        # Create summary markdown
        md_path = output_path / 'summary.md'
        self._create_markdown_summary(md_path, report)
        print(f"Markdown summary saved to {md_path}")

    def _create_markdown_summary(self, path: Path, report: Dict):
        """Create markdown summary of comparison."""
        with open(path, 'w') as f:
            f.write("# Model Comparison Report\n\n")

            for modality, summary in report['summary'].items():
                f.write(f"## {modality.upper()} Results\n\n")

                if 'avg_auc' in summary:
                    f.write(f"- **Average AUC**: {summary['avg_auc']:.3f} ")
                    f.write(f"(Paper: {summary['avg_paper_auc']:.3f})\n")

                if 'avg_mae' in summary:
                    f.write(f"- **Average MAE**: {summary['avg_mae']:.2f} ")
                    f.write(f"(Paper: {summary['avg_paper_mae']:.2f})\n")

                f.write("\n### Task Details\n\n")
                f.write("| Task | Metric | Yours | Paper | Difference |\n")
                f.write("|------|--------|-------|-------|------------|\n")

                for comp in report['detailed_comparisons'][modality]:
                    task = comp['task'].replace('_', ' ').title()

                    if 'your_auc' in comp:
                        f.write(f"| {task} | AUC | {comp['your_auc']:.3f} | ")
                        f.write(f"{comp['paper_auc']:.3f} | {comp['auc_diff']:+.3f} |\n")

                    if 'your_mae' in comp:
                        f.write(f"| {task} | MAE | {comp['your_mae']:.2f} | ")
                        f.write(f"{comp['paper_mae']:.2f} | {comp['mae_diff']:+.2f} |\n")

                f.write("\n")

            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")


def test_comparator():
    """Test function for the comparator."""
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

    # Try plotting (should handle missing data gracefully)
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