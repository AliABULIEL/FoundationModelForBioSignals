# biosignal/compare.py
"""
Module to compare our results with Apple paper benchmarks
Generates comprehensive comparison reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional
from datetime import datetime


class ResultsComparator:
    """Compare experimental results with paper benchmarks."""

    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize comparator.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Paper benchmarks from Table 2
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

        # Store results
        self.our_results = {}
        self.comparison_df = None

    def load_results(
        self,
        results_path: str,
        modality: str = 'ppg'
    ):
        """
        Load our experimental results.

        Args:
            results_path: Path to results file (CSV or JSON)
            modality: Signal modality ('ppg' or 'ecg')
        """
        path = Path(results_path)

        if path.suffix == '.csv':
            df = pd.read_csv(path)
            # Convert to dictionary format
            self.our_results[modality] = {}
            for _, row in df.iterrows():
                task = row['task']
                metrics = row.to_dict()
                metrics.pop('task')
                metrics.pop('type', None)
                self.our_results[modality][task] = metrics

        elif path.suffix == '.json':
            with open(path, 'r') as f:
                results = json.load(f)
            self.our_results[modality] = results

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        print(f"Loaded {modality.upper()} results from {path}")

    def compare_modality(self, modality: str = 'ppg') -> pd.DataFrame:
        """
        Compare results for a specific modality.

        Args:
            modality: 'ppg' or 'ecg'

        Returns:
            DataFrame with comparison
        """
        if modality not in self.our_results:
            raise ValueError(f"No results loaded for {modality}")

        comparisons = []

        for task, paper_metrics in self.paper_benchmarks[modality].items():
            comparison = {
                'modality': modality.upper(),
                'task': task
            }

            # Get our results
            our_metrics = self.our_results[modality].get(task, {})

            # Compare each metric
            for metric, paper_value in paper_metrics.items():
                our_value = our_metrics.get(metric, None)

                if our_value is not None:
                    comparison[f'paper_{metric}'] = paper_value
                    comparison[f'our_{metric}'] = our_value

                    # Calculate difference
                    if metric in ['auc', 'pauc', 'r2']:
                        # Higher is better
                        diff = our_value - paper_value
                        pct_diff = (diff / paper_value) * 100
                    else:
                        # Lower is better (mae, rmse)
                        diff = paper_value - our_value
                        pct_diff = (diff / paper_value) * 100

                    comparison[f'diff_{metric}'] = diff
                    comparison[f'pct_diff_{metric}'] = pct_diff

            comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    def generate_report(
        self,
        save_dir: str = 'data/outputs/comparisons'
    ) -> Dict:
        """
        Generate comprehensive comparison report.

        Args:
            save_dir: Directory to save report

        Returns:
            Dictionary with comparison statistics
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'modalities': list(self.our_results.keys()),
            'comparisons': {}
        }

        all_comparisons = []

        for modality in self.our_results.keys():
            # Generate comparison DataFrame
            df = self.compare_modality(modality)
            all_comparisons.append(df)

            # Calculate summary statistics
            summary = {}

            # AUC comparisons
            auc_cols = [col for col in df.columns if 'our_auc' in col]
            if auc_cols:
                our_aucs = df[[col for col in auc_cols]].values.flatten()
                paper_aucs = df[[col.replace('our_', 'paper_') for col in auc_cols]].values.flatten()

                # Remove NaN values
                mask = ~np.isnan(our_aucs) & ~np.isnan(paper_aucs)
                our_aucs = our_aucs[mask]
                paper_aucs = paper_aucs[mask]

                if len(our_aucs) > 0:
                    summary['mean_auc_ours'] = float(np.mean(our_aucs))
                    summary['mean_auc_paper'] = float(np.mean(paper_aucs))
                    summary['mean_auc_diff'] = float(np.mean(our_aucs - paper_aucs))

            # MAE comparisons
            mae_cols = [col for col in df.columns if 'our_mae' in col]
            if mae_cols:
                our_maes = df[[col for col in mae_cols]].values.flatten()
                paper_maes = df[[col.replace('our_', 'paper_') for col in mae_cols]].values.flatten()

                # Remove NaN values
                mask = ~np.isnan(our_maes) & ~np.isnan(paper_maes)
                our_maes = our_maes[mask]
                paper_maes = paper_maes[mask]

                if len(our_maes) > 0:
                    summary['mean_mae_ours'] = float(np.mean(our_maes))
                    summary['mean_mae_paper'] = float(np.mean(paper_maes))
                    summary['mean_mae_diff'] = float(np.mean(paper_maes - our_maes))

            report['comparisons'][modality] = {
                'summary': summary,
                'details': df.to_dict('records')
            }

        # Combine all comparisons
        if all_comparisons:
            self.comparison_df = pd.concat(all_comparisons, ignore_index=True)

            # Save comparison table
            csv_path = save_dir / 'comparison_table.csv'
            self.comparison_df.to_csv(csv_path, index=False)
            print(f"Comparison table saved to {csv_path}")

        # Save JSON report
        json_path = save_dir / 'comparison_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Comparison report saved to {json_path}")

        return report

    def plot_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 8)
    ):
        """
        Create visualization comparing our results with paper.

        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.comparison_df is None or self.comparison_df.empty:
            print("No comparison data to plot")
            return

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Comparison with Apple Paper Results', fontsize=14, fontweight='bold')

        # 1. AUC Comparison (Classification tasks)
        ax = axes[0, 0]
        auc_data = []

        for _, row in self.comparison_df.iterrows():
            if 'our_auc' in row and not pd.isna(row.get('our_auc')):
                auc_data.append({
                    'Task': f"{row['modality']}\n{row['task'].replace('_', ' ').title()}",
                    'Paper': row.get('paper_auc', 0),
                    'Ours': row.get('our_auc', 0)
                })

        if auc_data:
            auc_df = pd.DataFrame(auc_data)
            auc_df_melted = auc_df.melt(id_vars='Task', var_name='Source', value_name='AUC')

            sns.barplot(data=auc_df_melted, x='Task', y='AUC', hue='Source', ax=ax)
            ax.set_title('AUC Comparison (Classification)')
            ax.set_ylim([0.5, 1.0])
            ax.set_ylabel('AUC')
            ax.set_xlabel('')
            ax.legend(title='')
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.3, label='Good (0.9)')

            # Rotate x labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. MAE Comparison (Regression tasks)
        ax = axes[0, 1]
        mae_data = []

        for _, row in self.comparison_df.iterrows():
            if 'our_mae' in row and not pd.isna(row.get('our_mae')):
                mae_data.append({
                    'Task': f"{row['modality']}\n{row['task'].replace('_', ' ').title()}",
                    'Paper': row.get('paper_mae', 0),
                    'Ours': row.get('our_mae', 0)
                })

        if mae_data:
            mae_df = pd.DataFrame(mae_data)
            mae_df_melted = mae_df.melt(id_vars='Task', var_name='Source', value_name='MAE')

            sns.barplot(data=mae_df_melted, x='Task', y='MAE', hue='Source', ax=ax)
            ax.set_title('MAE Comparison (Regression)')
            ax.set_ylabel('MAE')
            ax.set_xlabel('')
            ax.legend(title='')

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Performance Difference
        ax = axes[1, 0]
        diff_data = []

        for _, row in self.comparison_df.iterrows():
            task_name = f"{row['modality']} {row['task'].replace('_', ' ')}"

            if 'diff_auc' in row and not pd.isna(row.get('diff_auc')):
                diff_data.append({
                    'Task': task_name,
                    'Metric': 'AUC',
                    'Difference': row['diff_auc'] * 100  # Convert to percentage
                })

            if 'diff_mae' in row and not pd.isna(row.get('diff_mae')):
                diff_data.append({
                    'Task': task_name,
                    'Metric': 'MAE',
                    'Difference': row['diff_mae']
                })

        if diff_data:
            diff_df = pd.DataFrame(diff_data)

            # Create grouped bar chart
            metrics = diff_df['Metric'].unique()
            x = np.arange(len(diff_df['Task'].unique()))
            width = 0.35

            for i, metric in enumerate(metrics):
                metric_data = diff_df[diff_df['Metric'] == metric]
                ax.bar(x + i * width, metric_data['Difference'], width, label=metric)

            ax.set_title('Performance Difference (Ours - Paper)')
            ax.set_ylabel('Difference')
            ax.set_xlabel('Task')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend()

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')

        # 4. Summary Statistics
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate summary stats
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"

        for modality in self.our_results.keys():
            summary_text += f"{modality.upper()}:\n"

            # Get modality-specific data
            mod_df = self.comparison_df[self.comparison_df['modality'] == modality.upper()]

            # AUC stats
            auc_cols = [col for col in mod_df.columns if 'our_auc' in col]
            if auc_cols:
                our_aucs = mod_df[auc_cols].values.flatten()
                our_aucs = our_aucs[~np.isnan(our_aucs)]
                if len(our_aucs) > 0:
                    summary_text += f"  Mean AUC: {np.mean(our_aucs):.3f}\n"

            # MAE stats
            mae_cols = [col for col in mod_df.columns if 'our_mae' in col]
            if mae_cols:
                our_maes = mod_df[mae_cols].values.flatten()
                our_maes = our_maes[~np.isnan(our_maes)]
                if len(our_maes) > 0:
                    summary_text += f"  Mean MAE: {np.mean(our_maes):.2f}\n"

            summary_text += "\n"

        # Add dataset info
        summary_text += "Dataset: BUT PPG\n"
        summary_text += "Model: EfficientNet1D\n"
        summary_text += "SSL: RegularizedInfoNCE + KoLeo"

        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print summary comparison to console."""
        if not self.our_results:
            print("No results loaded")
            return

        print("\n" + "="*60)
        print("COMPARISON WITH APPLE PAPER RESULTS")
        print("="*60)

        for modality in self.our_results.keys():
            print(f"\n{modality.upper()} Results:")
            print("-"*40)

            df = self.compare_modality(modality)

            for _, row in df.iterrows():
                print(f"\n{row['task'].replace('_', ' ').title()}:")

                # AUC comparison
                if 'our_auc' in row and not pd.isna(row.get('our_auc')):
                    paper_auc = row.get('paper_auc', 0)
                    our_auc = row.get('our_auc', 0)
                    diff = our_auc - paper_auc

                    print(f"  AUC: {our_auc:.3f} (Paper: {paper_auc:.3f}, Diff: {diff:+.3f})")

                # MAE comparison
                if 'our_mae' in row and not pd.isna(row.get('our_mae')):
                    paper_mae = row.get('paper_mae', 0)
                    our_mae = row.get('our_mae', 0)
                    diff = paper_mae - our_mae

                    print(f"  MAE: {our_mae:.2f} (Paper: {paper_mae:.2f}, Diff: {diff:+.2f})")

        print("\n" + "="*60)


# ============= TEST FUNCTIONS =============

def test_comparison():
    """Test comparison functionality."""
    print("=" * 50)
    print("Testing Comparison Module")
    print("=" * 50)

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

    # Clean up
    import shutil
    if report_dir.exists():
        shutil.rmtree(report_dir)

    print("\n" + "=" * 50)
    print("All comparison tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_comparison()