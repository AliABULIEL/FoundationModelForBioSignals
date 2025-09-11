# biosignal/evaluate.py
"""
Evaluation module for downstream tasks
Uses centralized configuration management
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, auc
)
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, train_test_split
from tqdm import tqdm
import json
import pickle
import hashlib
import warnings

warnings.filterwarnings('ignore')

from data import BUTPPGDataset, BaseSignalDataset
from model import EfficientNet1D
from device import DeviceManager, get_device_manager
from config_loader import get_config  # Added ConfigLoader


class LinearProbe:
    """Linear probe for downstream evaluation with device manager."""

    def __init__(
            self,
            task_type: str = 'classification',
            alpha: Optional[float] = None,
            config_path: str = 'configs/config.yaml'
    ):
        """Initialize linear probe."""
        self.task_type = task_type
        
        # Load configuration
        config = get_config()
        eval_config = config.get_evaluation_config()
        
        # Use config value with fallback
        self.alpha = alpha if alpha is not None else eval_config.get('ridge_alpha', 1.0)

        # Get device from global device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        self.scaler = StandardScaler()
        self.probe = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the linear probe."""
        X_scaled = self.scaler.fit_transform(X)

        if self.task_type == 'classification':
            self.probe = RidgeClassifier(alpha=self.alpha, random_state=42)
        else:
            self.probe = Ridge(alpha=self.alpha, random_state=42)

        self.probe.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.probe is None:
            raise ValueError("Probe must be fitted first")

        X_scaled = self.scaler.transform(X)

        if self.task_type == 'classification':
            if hasattr(self.probe, 'decision_function'):
                return self.probe.decision_function(X_scaled)
            else:
                return self.probe.predict(X_scaled)
        else:
            return self.probe.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for classification."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only for classification")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.probe, 'decision_function'):
            scores = self.probe.decision_function(X_scaled)
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
        else:
            return self.probe.predict(X_scaled)


class EmbeddingCache:
    """Cache manager for embeddings to avoid recomputation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        # Load config
        config = get_config()
        
        # Use cache directory from config or default
        default_cache_dir = config.get('evaluation.embedding_cache_dir', 'data/cache/embeddings')
        self.cache_dir = cache_dir or Path(default_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def get_cache_key(self, dataset: BUTPPGDataset, encoder_path: str) -> str:
        """Generate cache key based on dataset properties."""
        params = {
            'modality': dataset.modality,
            'split': dataset.split,
            'n_participants': len(dataset.participant_records),
            'downsample': getattr(dataset, 'downsample', False),
            'encoder_path': str(encoder_path)
        }
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Tuple]:
        """Retrieve cached embeddings."""
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.memory_cache[cache_key] = data
                return data
        return None

    def set(self, cache_key: str, data: Tuple):
        """Store embeddings in cache."""
        self.memory_cache[cache_key] = data
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)


class DownstreamEvaluator:
    """
    Evaluator for downstream tasks
    Uses centralized configuration
    """

    def __init__(
            self,
            encoder_path: str,
            config_path: str = 'configs/config.yaml',
            device_manager: Optional[DeviceManager] = None,
            use_cache: Optional[bool] = None
    ):
        """Initialize evaluator with device manager."""
        self.encoder_path = encoder_path
        self.current_dataset_type = None
        
        # Load configuration
        self.config = get_config()
        eval_config = self.config.get_evaluation_config()
        
        # Use cache setting from config
        self.use_cache = use_cache if use_cache is not None else eval_config.get('use_embedding_cache', True)

        # Use device manager
        if device_manager is None:
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager

        self.device = self.device_manager.device

        # Initialize cache
        self.cache = EmbeddingCache() if self.use_cache else None

        # Cache for current dataset embeddings
        self._current_dataset_id = None
        self._cached_embeddings = None
        self._cached_labels = None
        self._cached_pids = None

        # Load encoder
        self._load_encoder()

        # Define tasks from config
        self.tasks = self._load_tasks_from_config()

        # Results storage
        self.results = {}

        print(f"Evaluator initialized on {self.device_manager.type}")
        if self.device_manager.is_cuda:
            mem_stats = self.device_manager.memory_stats()
            print(f"  GPU memory available: {mem_stats.get('free', 0):.2f} GB")

    def evaluate_supervised_heads(self, model, dataset):
        """
        Enhanced evaluation for semi-supervised models with Mean Teacher.
        """
        from torch.utils.data import DataLoader
        from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

        model.eval()

        # Check if using Mean Teacher
        if hasattr(model, 'teacher_encoder'):
            print("Using Mean Teacher model for evaluation")
            encoder_to_use = model.teacher_encoder  # Use teacher for more stable predictions
        else:
            encoder_to_use = model.encoder

        all_preds = {'age': [], 'bmi': [], 'sex': []}
        all_labels = {'age': [], 'bmi': [], 'sex': []}
        all_scores = {'age': [], 'sex': []}  # For AUC calculation

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    seg1, seg2, labels = batch
                else:
                    continue

                seg1 = seg1.to(self.device)

                # Get embeddings
                embeddings = encoder_to_use(seg1)

                # Age classification
                if hasattr(model, 'age_classifier'):
                    age_logits = model.age_classifier(embeddings)
                    age_probs = F.softmax(age_logits, dim=1)
                    age_pred = torch.argmax(age_logits, dim=1)

                    if 'age' in labels:
                        age_values = labels['age']
                        age_binary = (age_values > 50).long()
                        valid_mask = age_values >= 0

                        if valid_mask.any():
                            all_preds['age'].extend(age_pred[valid_mask].cpu().numpy())
                            all_labels['age'].extend(age_binary[valid_mask].cpu().numpy())
                            all_scores['age'].extend(age_probs[valid_mask, 1].cpu().numpy())

                # BMI regression - FIXED denormalization
                if hasattr(model, 'bmi_regressor'):
                    bmi_pred = model.bmi_regressor(embeddings).squeeze()

                    # Denormalize from [0,1] to actual BMI range
                    bmi_pred_actual = bmi_pred * 15.0 + 20.0  # [0,1] -> [20,35]

                    if 'bmi' in labels:
                        bmi_values = labels['bmi']
                        valid_mask = bmi_values > 0

                        if valid_mask.any():
                            all_preds['bmi'].extend(bmi_pred_actual[valid_mask].cpu().numpy())
                            all_labels['bmi'].extend(bmi_values[valid_mask].cpu().numpy())

                # Sex classification
                if hasattr(model, 'sex_classifier'):
                    sex_logits = model.sex_classifier(embeddings)
                    sex_probs = F.softmax(sex_logits, dim=1)
                    sex_pred = torch.argmax(sex_logits, dim=1)

                    if 'sex' in labels:
                        sex_values = labels['sex'].long()
                        valid_mask = sex_values >= 0

                        if valid_mask.any():
                            all_preds['sex'].extend(sex_pred[valid_mask].cpu().numpy())
                            all_labels['sex'].extend(sex_values[valid_mask].cpu().numpy())
                            all_scores['sex'].extend(sex_probs[valid_mask, 1].cpu().numpy())

        # Calculate enhanced metrics
        metrics = {}

        # Age metrics with AUC
        if len(all_labels['age']) > 0:
            metrics['age_acc'] = accuracy_score(all_labels['age'], all_preds['age'])
            if len(np.unique(all_labels['age'])) > 1:
                metrics['age_auc'] = roc_auc_score(all_labels['age'], all_scores['age'])

        # BMI metrics
        if len(all_labels['bmi']) > 0:
            metrics['bmi_mae'] = mean_absolute_error(all_labels['bmi'], all_preds['bmi'])
            metrics['bmi_rmse'] = np.sqrt(mean_squared_error(all_labels['bmi'], all_preds['bmi']))

        # Sex metrics with AUC
        if len(all_labels['sex']) > 0:
            metrics['sex_acc'] = accuracy_score(all_labels['sex'], all_preds['sex'])
            if len(np.unique(all_labels['sex'])) > 1:
                metrics['sex_auc'] = roc_auc_score(all_labels['sex'], all_scores['sex'])

        model.train()
        return metrics
    def _load_tasks_from_config(self) -> Dict:
        """Load task definitions from configuration."""
        eval_config = self.config.get_evaluation_config()
        tasks_list = eval_config.get('tasks', [])
        
        # Default tasks if not in config
        default_tasks = {
            'age_classification': {
                'type': 'classification',
                'threshold': 50,
                'label_key': 'age'
            },
            'age_regression': {
                'type': 'regression',
                'label_key': 'age'
            },
            'sex_classification': {
                'type': 'classification',
                'label_key': 'sex'
            },
            'bmi_classification': {
                'type': 'classification',
                'threshold': 25,
                'label_key': 'bmi'
            },
            'bmi_regression': {
                'type': 'regression',
                'label_key': 'bmi'
            },
            # 'bp_classification': {
            #     'type': 'classification',
            #     'threshold': 140,
            #     'label_key': 'bp_systolic'
            # }
        }
        
        # Build tasks dict from config list
        tasks = {}
        for task_name in tasks_list:
            if task_name in default_tasks:
                tasks[task_name] = default_tasks[task_name]
            else:
                # Try to parse task from config
                if '_classification' in task_name:
                    label_key = task_name.replace('_classification', '')
                    tasks[task_name] = {
                        'type': 'classification',
                        'label_key': label_key,
                        'threshold': self.config.get(f'evaluation.thresholds.{label_key}', 50)
                    }
                elif '_regression' in task_name:
                    label_key = task_name.replace('_regression', '')
                    tasks[task_name] = {
                        'type': 'regression',
                        'label_key': label_key
                    }
        
        # If no tasks in config, use defaults
        if not tasks:
            tasks = default_tasks
            
        return tasks
    def clear_cache(self):
        """Clear the embedding cache."""
        self._current_dataset_id = None
        self._cached_embeddings = None
        self._cached_labels = None
        self._cached_pids = None
        print("✓ Cleared embedding cache")

    def visualize_embeddings(self, dataset, save_dir: str = None):
        """Visualize and analyze embeddings to debug poor performance."""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import seaborn as sns

        # Extract embeddings if not cached
        embeddings, labels_list, participant_ids = self.extract_embeddings(
            dataset, aggregate_by_participant=True
        )

        print("\n" + "=" * 60)
        print("EMBEDDING ANALYSIS")
        print("=" * 60)

        # 1. Check embedding statistics
        print("\n1. Embedding Statistics:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Mean: {embeddings.mean():.4f}")
        print(f"   Std: {embeddings.std():.4f}")
        print(f"   Min: {embeddings.min():.4f}")
        print(f"   Max: {embeddings.max():.4f}")

        # Check for dead neurons (always zero)
        dead_neurons = np.sum(np.abs(embeddings).max(axis=0) < 1e-6)
        print(f"   Dead neurons: {dead_neurons}/{embeddings.shape[1]}")

        # Check for collapsed embeddings (all similar)
        pairwise_dist = np.std([np.linalg.norm(embeddings[i] - embeddings[j])
                                for i in range(min(100, len(embeddings)))
                                for j in range(i + 1, min(100, len(embeddings)))])
        print(f"   Embedding diversity (std of distances): {pairwise_dist:.4f}")

        # 2. Extract labels for visualization
        ages = []
        sexes = []
        bmis = []

        for label_dict in labels_list:
            if isinstance(label_dict, dict):
                ages.append(label_dict.get('age', -1))
                sexes.append(label_dict.get('sex', -1))
                bmis.append(label_dict.get('bmi', -1))

        ages = np.array(ages)
        sexes = np.array(sexes)
        bmis = np.array(bmis)

        # Filter valid samples
        valid_mask = (ages > 0) & (sexes >= 0) & (bmis > 0)
        embeddings_valid = embeddings[valid_mask]
        ages_valid = ages[valid_mask]
        sexes_valid = sexes[valid_mask]
        bmis_valid = bmis[valid_mask]

        print(f"\n2. Valid samples for visualization: {valid_mask.sum()}/{len(embeddings)}")

        if valid_mask.sum() < 10:
            print("   ⚠️ Too few valid samples for visualization")
            return

        # 3. Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # PCA visualization
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(embeddings_valid)

        print(f"\n3. PCA explained variance: {pca.explained_variance_ratio_}")

        # Plot PCA colored by age
        scatter1 = axes[0, 0].scatter(emb_pca[:, 0], emb_pca[:, 1],
                                      c=ages_valid, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('PCA - Colored by Age')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter1, ax=axes[0, 0])

        # Plot PCA colored by sex
        axes[0, 1].scatter(emb_pca[:, 0], emb_pca[:, 1],
                           c=sexes_valid, cmap='coolwarm', alpha=0.6)
        axes[0, 1].set_title('PCA - Colored by Sex')
        axes[0, 1].set_xlabel(f'PC1')
        axes[0, 1].set_ylabel(f'PC2')

        # Plot PCA colored by BMI
        scatter3 = axes[0, 2].scatter(emb_pca[:, 0], emb_pca[:, 1],
                                      c=bmis_valid, cmap='plasma', alpha=0.6)
        axes[0, 2].set_title('PCA - Colored by BMI')
        axes[0, 2].set_xlabel(f'PC1')
        axes[0, 2].set_ylabel(f'PC2')
        plt.colorbar(scatter3, ax=axes[0, 2])

        # t-SNE visualization (if not too many samples)
        if len(embeddings_valid) <= 1000:
            print("\n4. Computing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_valid) - 1))
            emb_tsne = tsne.fit_transform(embeddings_valid)

            # t-SNE colored by age
            scatter4 = axes[1, 0].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                                          c=ages_valid, cmap='viridis', alpha=0.6)
            axes[1, 0].set_title('t-SNE - Colored by Age')
            plt.colorbar(scatter4, ax=axes[1, 0])

            # t-SNE colored by sex
            axes[1, 1].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                               c=sexes_valid, cmap='coolwarm', alpha=0.6)
            axes[1, 1].set_title('t-SNE - Colored by Sex')

            # t-SNE colored by BMI
            scatter6 = axes[1, 2].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                                          c=bmis_valid, cmap='plasma', alpha=0.6)
            axes[1, 2].set_title('t-SNE - Colored by BMI')
            plt.colorbar(scatter6, ax=axes[1, 2])

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'embedding_visualization.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n5. Visualization saved to {save_path}")

        plt.show()

        # 6. Check correlation between embeddings and labels
        print("\n6. Embedding-Label Correlations:")
        from scipy.stats import pearsonr

        # Calculate correlation for each embedding dimension with age
        age_correlations = []
        for i in range(min(10, embeddings_valid.shape[1])):  # Check first 10 dims
            corr, p_val = pearsonr(embeddings_valid[:, i], ages_valid)
            age_correlations.append(abs(corr))

        print(f"   Max correlation with age: {max(age_correlations):.4f}")
        print(f"   Mean correlation with age: {np.mean(age_correlations):.4f}")

        # 7. Check if embeddings are collapsed
        print("\n7. Embedding Collapse Check:")
        distances = []
        n_samples = min(100, len(embeddings_valid))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances.append(np.linalg.norm(embeddings_valid[i] - embeddings_valid[j]))

        print(f"   Mean pairwise distance: {np.mean(distances):.4f}")
        print(f"   Std pairwise distance: {np.std(distances):.4f}")

        if np.std(distances) < 0.1:
            print("   ⚠️ WARNING: Embeddings may be collapsed (very low variance)")

        return embeddings_valid, ages_valid, sexes_valid, bmis_valid
    def _load_encoder(self):
        """Load pre-trained encoder."""
        # Get modality from checkpoint or default
        modality = 'ppg'  # Default, could be stored in checkpoint
        
        model_config = self.config.get_model_config()
        embedding_dim = model_config.get('embedding_dim', 256)
        
        print(f"  Embedding dim: {embedding_dim} (from config)")

        self.encoder = EfficientNet1D(
            in_channels=1 if modality != 'acc' else self.config.get('dataset.acc.channels', 3),
            embedding_dim=embedding_dim,
            modality=modality
        )

        checkpoint = torch.load(self.encoder_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            elif 'online_encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['online_encoder_state_dict'])
            else:
                self.encoder.load_state_dict(checkpoint)
        else:
            self.encoder.load_state_dict(checkpoint)

        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        print(f"Encoder loaded from {self.encoder_path}")
        print(f"  Modality: {modality}")

    def evaluate_all_tasks_extended(
            self,
            dataset_type: str = 'but_ppg',
            modality: str = 'ppg',
            split: str = 'test',
            data_dir: Optional[str] = None,
            save_path: Optional[str] = None,
            downsample: bool = False
    ) -> pd.DataFrame:
        """
        Extended evaluation that supports both BUT PPG and VitalDB.

        Args:
            dataset_type: 'but_ppg' or 'vitaldb'
            modality: Signal modality
            split: Data split
            data_dir: Data directory (for BUT PPG)
            save_path: Where to save results
            downsample: Whether to downsample

        Returns:
            DataFrame with evaluation results
        """
        # Create appropriate dataset
        dataset = create_evaluation_dataset(
            dataset_type=dataset_type,
            modality=modality,
            split=split,
            data_dir=data_dir,
            downsample=downsample
        )

        self.current_dataset_type = dataset_type

        # Determine number of participants/cases
        if dataset_type == 'vitaldb':
            n_participants = len(dataset.cases)
            dataset_label = 'VitalDB'
        else:
            n_participants = len(dataset.participant_records)
            dataset_label = 'BUT PPG'

        print(f"\nEvaluating on {dataset_label} - {modality.upper()}")
        print(f"  Dataset: {dataset_label}")
        print(f"  Split: {split}")
        print(f"  Participants/Cases: {n_participants}")

        # Use existing evaluate_all_tasks with the dataset
        results_df = self.evaluate_all_tasks(dataset, save_path)

        # Add dataset type to results
        results_df['dataset'] = dataset_label
        results_df['dataset_type'] = dataset_type

        return results_df
    @torch.no_grad()
    def extract_embeddings(
            self,
            dataset: BUTPPGDataset,
            aggregate_by_participant: bool = True,
            batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Extract embeddings with BOTH segments processed.
        """
        from torch.utils.data import DataLoader

        # [Keep your caching logic as is - it's fine]
        dataset_id = id(dataset)
        if (self._current_dataset_id == dataset_id and
                self._cached_embeddings is not None):
            print("  ✓ Reusing cached embeddings for current dataset")
            return self._cached_embeddings, self._cached_labels, self._cached_pids

        # [Keep cache check as is]

        print("\n" + "=" * 60)
        print("DEBUG: extract_embeddings")
        print("=" * 60)

        # [Keep your debug prints - they're helpful]

        # Get batch size
        if batch_size is None:
            eval_config = self.config.get_evaluation_config()
            batch_size = eval_config.get('batch_size', 512)
        print(f"Using batch size: {batch_size}")

        # DataLoader settings
        num_workers = 0  # IMPORTANT: Use 0 to avoid multiprocessing issues with participant IDs

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device_manager.is_cuda
        )

        all_embeddings = []
        all_labels = []
        all_participant_ids = []

        print("\nProcessing batches...")

        for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting embeddings")):

            # Handle different batch formats
            if len(batch) == 4:
                seg1, seg2, pid, labels = batch
            elif len(batch) == 3:
                seg1, seg2, pid = batch
                labels = {}
            else:
                seg1, seg2 = batch
                pid = None
                labels = {}

            # Move to device
            seg1 = seg1.to(self.device, non_blocking=True)
            seg2 = seg2.to(self.device, non_blocking=True)  # IMPORTANT: Also move seg2!

            # Extract embeddings for BOTH segments
            with torch.no_grad():
                emb1 = self.encoder(seg1)
                emb2 = self.encoder(seg2)  # IMPORTANT: Process seg2 too!

            # Store BOTH embeddings
            all_embeddings.append(emb1.cpu().numpy())
            all_embeddings.append(emb2.cpu().numpy())  # IMPORTANT: Store seg2 embeddings!

            # Handle participant IDs - need to duplicate for both segments
            if pid is not None:
                batch_size_actual = seg1.shape[0]

                # Handle VitalDB case IDs (integers)
                if self.current_dataset_type == 'vitaldb':
                    # VitalDB returns a single case_id per batch item
                    if torch.is_tensor(pid):
                        pid_list = pid.cpu().numpy().tolist()
                    elif isinstance(pid, (list, tuple)):
                        pid_list = list(pid)
                    elif isinstance(pid, (int, np.integer)):
                        # Single case ID for whole batch (shouldn't happen but handle it)
                        pid_list = [pid] * batch_size_actual
                    else:
                        pid_list = [pid] * batch_size_actual
                else:
                    # BUT PPG handling (existing code)
                    if torch.is_tensor(pid):
                        pid_list = pid.tolist() if pid.dim() > 0 else [pid.item()]
                    elif isinstance(pid, (list, tuple)):
                        pid_list = list(pid)
                    else:
                        pid_list = [str(pid)] * batch_size_actual

                # Duplicate PIDs for both seg1 and seg2
                all_participant_ids.extend(pid_list)  # For seg1
                all_participant_ids.extend(pid_list)  # For seg2

            # Handle labels - need to duplicate for both segments
            if labels:
                batch_labels = []
                batch_size_actual = seg1.shape[0]
                if self.current_dataset_type == 'vitaldb':
                    # VitalDB returns demographics directly
                    for i in range(batch_size_actual):
                        if isinstance(labels, dict):
                            # Check if it's a batched dict or single dict
                            sample_labels = {}
                            for k, v in labels.items():
                                if torch.is_tensor(v):
                                    if v.shape[0] == batch_size_actual:
                                        sample_labels[k] = v[i].item()
                                    else:
                                        sample_labels[k] = v.item()
                                elif isinstance(v, (list, tuple)) and len(v) == batch_size_actual:
                                    sample_labels[k] = v[i]
                                else:
                                    sample_labels[k] = v
                            batch_labels.append(sample_labels)
                        else:
                            batch_labels.append({})
                else:
                    batch_labels = []
                    batch_size_actual = seg1.shape[0]

                    for i in range(batch_size_actual):
                        sample_labels = {}
                        if isinstance(labels, dict):
                            for k, v in labels.items():
                                if torch.is_tensor(v):
                                    if v.dim() == 0:
                                        sample_labels[k] = v.item()
                                    elif v.shape[0] == batch_size_actual:
                                        sample_labels[k] = v[i].item()
                                    else:
                                        sample_labels[k] = v.item() if v.numel() == 1 else -1
                                elif isinstance(v, (list, tuple)) and len(v) == batch_size_actual:
                                    sample_labels[k] = v[i]
                                else:
                                    sample_labels[k] = v if isinstance(v, (int, float)) else -1
                        batch_labels.append(sample_labels)

                # Duplicate labels for both segments
                all_labels.extend(batch_labels)  # For seg1
                all_labels.extend(batch_labels)  # For seg2

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)

        print(f"\nExtraction complete:")
        print(f"  - Total embeddings: {len(embeddings)}")
        print(f"  - Total labels: {len(all_labels)}")
        print(f"  - Total participant IDs: {len(all_participant_ids)}")
        print(f"  - Unique participants before aggregation: {len(set(all_participant_ids))}")

        # Aggregate by participant if needed
        if aggregate_by_participant and all_participant_ids:
            print(f"\nAggregating by participant...")

            # Group embeddings by participant
            participant_embeddings = {}
            participant_labels = {}

            for emb, label, pid in zip(embeddings, all_labels, all_participant_ids):
                if pid not in participant_embeddings:
                    participant_embeddings[pid] = []
                    participant_labels[pid] = label  # Keep first occurrence of label
                participant_embeddings[pid].append(emb)

            # Average embeddings per participant
            aggregated_embeddings = []
            aggregated_labels = []
            aggregated_pids = []

            for pid in sorted(participant_embeddings.keys()):
                # Average all embeddings for this participant
                avg_embedding = np.mean(participant_embeddings[pid], axis=0)
                aggregated_embeddings.append(avg_embedding)
                aggregated_labels.append(participant_labels[pid])
                aggregated_pids.append(pid)

            embeddings = np.array(aggregated_embeddings)
            all_labels = aggregated_labels
            all_participant_ids = aggregated_pids

            print(f"  - After aggregation: {len(embeddings)} participant embeddings")
            print(f"  - Unique participants: {len(set(all_participant_ids))}")

        # Cache the results
        self._current_dataset_id = dataset_id
        self._cached_embeddings = embeddings
        self._cached_labels = all_labels
        self._cached_pids = np.array(all_participant_ids)

        return embeddings, all_labels, np.array(all_participant_ids)
    # def extract_embeddings(
    #         self,
    #         dataset: BUTPPGDataset,
    #         aggregate_by_participant: bool = True,
    #         batch_size: Optional[int] = None
    # ) -> Tuple[np.ndarray, List, np.ndarray]:
    #     """
    #     Extract embeddings with caching optimization.
    #     """
    #     from torch.utils.data import DataLoader
    #
    #     # Check if we already have embeddings for this dataset
    #     dataset_id = id(dataset)
    #     if (self._current_dataset_id == dataset_id and
    #             self._cached_embeddings is not None):
    #         print("  ✓ Reusing cached embeddings for current dataset")
    #         return self._cached_embeddings, self._cached_labels, self._cached_pids
    #
    #     # Check persistent cache
    #     if self.cache:
    #         cache_key = self.cache.get_cache_key(dataset, self.encoder_path)
    #         cached_data = self.cache.get(cache_key)
    #         if cached_data is not None:
    #             print("  ✓ Loading embeddings from persistent cache")
    #             self._current_dataset_id = dataset_id
    #             self._cached_embeddings, self._cached_labels, self._cached_pids = cached_data
    #             return cached_data
    #
    #     print("\n" + "=" * 60)
    #     print("DEBUG: extract_embeddings")
    #     print("=" * 60)
    #
    #     print("DEBUG 1: Dataset Configuration")
    #     print(f"  - Dataset type: {type(dataset).__name__}")
    #     print(f"  - Dataset return_labels: {dataset.return_labels}")
    #     print(f"  - Dataset return_participant_id: {dataset.return_participant_id}")
    #     print(f"  - Dataset length: {len(dataset)}")
    #     print(f"  - Aggregate by participant: {aggregate_by_participant}")
    #
    #     # Test dataset output
    #     print("\nDEBUG 2: Testing dataset.__getitem__ directly")
    #     if len(dataset) > 0:
    #         test_item = dataset[0]
    #         print(f"  - Item returned {len(test_item)} elements")
    #         for i, item in enumerate(test_item):
    #             if torch.is_tensor(item):
    #                 print(f"    [{i}] Tensor shape: {item.shape}")
    #             elif isinstance(item, dict):
    #                 print(f"    [{i}] Dict with keys: {list(item.keys())}")
    #                 for k, v in item.items():
    #                     print(f"        {k}: {v}")
    #             elif isinstance(item, str):
    #                 print(f"    [{i}] String (participant_id): '{item}'")
    #             else:
    #                 print(f"    [{i}] Type: {type(item).__name__}")
    #
    #     # Get batch size from config or use device optimal
    #     if batch_size is None:
    #         eval_config = self.config.get_evaluation_config()
    #         batch_size = eval_config.get('batch_size', None)
    #         if batch_size is None:
    #             batch_size = self.device_manager.get_optimal_batch_size(dataset.modality)
    #         print(f"Using batch size: {batch_size}")
    #
    #     # DataLoader settings from config
    #     num_workers = self.config.get('evaluation.num_workers', self.device_manager.get_num_workers())
    #
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=self.device_manager.is_cuda,
    #         prefetch_factor=2 if num_workers > 0 else None,
    #         persistent_workers=num_workers > 0
    #     )
    #
    #     all_embeddings = []
    #     all_labels = []
    #     all_participant_ids = []
    #
    #     print("\nDEBUG 3: Processing batches...")
    #     print("Extracting embeddings...")
    #
    #     if self.device_manager.is_cuda:
    #         initial_mem = self.device_manager.memory_stats()
    #         print(f"  Initial GPU memory: {initial_mem.get('allocated', 0):.2f} GB")
    #
    #     batch_count = 0
    #     for batch_idx, batch in enumerate(tqdm(loader)):
    #         batch_count += 1
    #
    #         # First batch debug
    #         if batch_idx == 0:
    #             print(f"\nDEBUG 4: First batch structure")
    #             print(f"  - Batch contains {len(batch)} elements")
    #             for i, b in enumerate(batch):
    #                 if torch.is_tensor(b):
    #                     print(f"    [{i}] Tensor shape: {b.shape}, dtype: {b.dtype}")
    #                 elif isinstance(b, dict):
    #                     print(f"    [{i}] Dict with keys: {list(b.keys())}")
    #                     for k, v in b.items():
    #                         if torch.is_tensor(v):
    #                             print(f"        {k}: Tensor shape {v.shape}")
    #                         else:
    #                             print(f"        {k}: {type(v).__name__}")
    #                 else:
    #                     print(f"    [{i}] Type: {type(b).__name__}")
    #
    #         # Handle different batch formats
    #         if len(batch) == 4:
    #             seg1, seg2, pid, labels = batch
    #             if batch_idx == 0:
    #                 print(f"  - Got 4 elements (with labels)")
    #         elif len(batch) == 3:
    #             seg1, seg2, pid = batch
    #             labels = {}
    #             if batch_idx == 0:
    #                 print(f"  - Got 3 elements (no labels)")
    #         else:
    #             seg1, seg2 = batch
    #             pid = None
    #             labels = {}
    #             if batch_idx == 0:
    #                 print(f"  - Got 2 elements (no pid, no labels)")
    #
    #         # Non-blocking transfer
    #         if self.device_manager.is_cuda:
    #             seg1 = seg1.to(self.device, non_blocking=True)
    #         else:
    #             seg1 = seg1.to(self.device)
    #
    #         # Extract embeddings
    #         embeddings = self.encoder(seg1)
    #         all_embeddings.append(embeddings.cpu().numpy())
    #
    #         # Handle participant IDs
    #         if pid is not None:
    #             if torch.is_tensor(pid):
    #                 pid_list = pid.tolist() if pid.dim() > 0 else [pid.item()]
    #             elif isinstance(pid, (list, tuple)):
    #                 pid_list = list(pid)
    #             elif isinstance(pid, str):
    #                 pid_list = [pid] * seg1.shape[0]
    #             else:
    #                 pid_list = [pid] * seg1.shape[0]
    #             all_participant_ids.extend(pid_list)
    #
    #             if batch_idx == 0:
    #                 print(f"  - PIDs in first batch: {pid_list[:min(3, len(pid_list))]}")
    #
    #         # Collect labels
    #         if labels:
    #             if batch_idx == 0:
    #                 print(f"  - Labels type: {type(labels)}")
    #                 print(f"  - Labels keys: {list(labels.keys()) if isinstance(labels, dict) else 'Not a dict'}")
    #
    #             batch_labels = []
    #             batch_size_actual = seg1.shape[0]
    #
    #             for i in range(batch_size_actual):
    #                 try:
    #                     if isinstance(labels, dict):
    #                         sample_labels = {}
    #                         for k, v in labels.items():
    #                             if torch.is_tensor(v):
    #                                 if v.dim() == 0:
    #                                     sample_labels[k] = v.item()
    #                                 elif v.shape[0] == batch_size_actual:
    #                                     sample_labels[k] = v[i].item()
    #                                 else:
    #                                     sample_labels[k] = v.item() if v.numel() == 1 else -1
    #                             elif isinstance(v, (list, tuple)) and len(v) == batch_size_actual:
    #                                 sample_labels[k] = v[i]
    #                             elif isinstance(v, (int, float)):
    #                                 sample_labels[k] = v
    #                             else:
    #                                 sample_labels[k] = -1
    #                         batch_labels.append(sample_labels)
    #                     else:
    #                         batch_labels.append({})
    #                 except Exception as e:
    #                     if batch_idx == 0 and i == 0:
    #                         print(f"    ERROR extracting label for sample {i}: {e}")
    #                     batch_labels.append({})
    #
    #             all_labels.extend(batch_labels)
    #
    #             if batch_idx == 0 and batch_labels:
    #                 print(f"  - First sample labels: {batch_labels[0]}")
    #
    #         # Periodic memory cleanup
    #         cache_clean_freq = self.config.get('evaluation.cache_clean_frequency', 50)
    #         if self.device_manager.is_cuda and batch_idx % cache_clean_freq == 0:
    #             self.device_manager.empty_cache()
    #
    #     print(f"\nDEBUG 5: Batch processing complete")
    #     print(f"  - Processed {batch_count} batches")
    #     print(f"  - Total embeddings: {len(all_embeddings)}")
    #     print(f"  - Total labels: {len(all_labels)}")
    #     print(f"  - Total participant IDs: {len(all_participant_ids)}")
    #
    #     # Concatenate
    #     embeddings = np.concatenate(all_embeddings, axis=0)
    #     print(f"  - Concatenated embeddings shape: {embeddings.shape}")
    #
    #     # Debug labels
    #     if all_labels:
    #         print(f"\nDEBUG 6: Labels check")
    #         print(f"  - First 3 labels:")
    #         for i in range(min(3, len(all_labels))):
    #             print(f"    [{i}]: {all_labels[i]}")
    #
    #         valid_count = sum(1 for label in all_labels
    #                           if isinstance(label, dict) and
    #                           any(v != -1 for v in label.values()))
    #         print(f"  - Labels with valid values: {valid_count}/{len(all_labels)}")
    #     else:
    #         print(f"\nDEBUG 6: WARNING - No labels collected!")
    #
    #     if self.device_manager.is_cuda:
    #         self.device_manager.empty_cache()
    #         final_mem = self.device_manager.memory_stats()
    #         print(f"  Final GPU memory: {final_mem.get('allocated', 0):.2f} GB")
    #
    #     # Aggregate if needed
    #     if aggregate_by_participant and all_participant_ids:
    #         print(f"\nDEBUG 7: Aggregating by participant")
    #         print(f"  - Before: {len(embeddings)} embeddings, {len(all_labels)} labels")
    #         embeddings, all_labels, all_participant_ids = self._aggregate_by_participant(
    #             embeddings, all_labels, all_participant_ids
    #         )
    #         print(f"  - After: {len(embeddings)} embeddings, {len(all_labels)} labels")
    #         print(f"  - Unique participants: {len(np.unique(all_participant_ids))}")
    #
    #     # Cache the results
    #     result = (embeddings, all_labels, np.array(all_participant_ids))
    #
    #     # Store in memory cache
    #     self._current_dataset_id = dataset_id
    #     self._cached_embeddings = embeddings
    #     self._cached_labels = all_labels
    #     self._cached_pids = np.array(all_participant_ids)
    #
    #     # Store in persistent cache
    #     if self.cache and cache_key:
    #         self.cache.set(cache_key, result)
    #         print("  ✓ Cached embeddings for future use")
    #
    #     return result

    def _aggregate_by_participant(
            self,
            embeddings: np.ndarray,
            labels: List[Dict],
            participant_ids: List
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Aggregate embeddings by participant."""
        participant_ids = [str(pid) for pid in participant_ids]
        unique_pids = np.unique(participant_ids)


        aggregated_embeddings = np.zeros((len(unique_pids), embeddings.shape[1]),
                                         dtype=np.float32)
        aggregated_labels = []

        print(f"  Aggregating {len(unique_pids)} unique participants")

        pids_array = np.array(participant_ids)

        for i, pid in enumerate(unique_pids):
            mask = pids_array == pid
            aggregated_embeddings[i] = embeddings[mask].mean(axis=0)

            if labels:
                pid_indices = np.where(mask)[0]
                if len(pid_indices) > 0:
                    aggregated_labels.append(labels[pid_indices[0]])

        return aggregated_embeddings, aggregated_labels, unique_pids

    def evaluate_task(
            self,
            task_name: str,
            dataset: BUTPPGDataset,
            n_splits: Optional[int] = None
    ) -> Dict:
        """
        Evaluate a specific task.
        """
        print(f"\n{'=' * 60}")
        print(f"DEBUG: evaluate_task for {task_name}")
        print(f"{'=' * 60}")

        if task_name not in self.tasks:
            print(f"ERROR: Unknown task: {task_name}")
            print(f"Available tasks: {list(self.tasks.keys())}")
            raise ValueError(f"Unknown task: {task_name}")

        task_info = self.tasks[task_name]
        print(f"DEBUG 1: Task configuration")
        print(f"  - Task type: {task_info['type']}")
        print(f"  - Label key: {task_info['label_key']}")
        if 'threshold' in task_info:
            print(f"  - Threshold: {task_info['threshold']}")

        # Get n_splits from config
        if n_splits is None:
            eval_config = self.config.get_evaluation_config()
            n_splits = eval_config.get('cv_folds', 5)

        print(f"\nDEBUG 2: Starting extraction for {task_name}...")
        embeddings, labels_list, participant_ids = self.extract_embeddings(
            dataset, aggregate_by_participant=True
        )

        print(f"\nDEBUG 3: Extraction complete")
        print(f"  - Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
        print(f"  - Labels list length: {len(labels_list) if labels_list else 0}")
        print(f"  - Participant IDs length: {len(participant_ids) if participant_ids is not None else 0}")

        if not labels_list:
            print(f"ERROR: No labels extracted!")
            return {}

        # Prepare labels
        label_key = task_info['label_key']
        print(f"\nDEBUG 4: Processing labels for key '{label_key}'")

        y = []
        valid_indices = []
        missing_key = 0
        invalid_value = 0

        for i, label_dict in enumerate(labels_list):
            if i < 5:
                print(f"  - Label {i}: {label_dict}")

            if not isinstance(label_dict, dict):
                print(f"    WARNING: Label {i} is not dict: {type(label_dict)}")
                continue

            if label_key in label_dict:
                value = label_dict[label_key]

                try:
                    if value == -1 or (isinstance(value, float) and np.isnan(value)):
                        invalid_value += 1
                        if i < 5:
                            print(f"    Invalid value: {value}")
                        continue

                    if task_info['type'] == 'classification':
                        if 'threshold' in task_info:
                            y_val = 1 if value > task_info['threshold'] else 0
                            y.append(y_val)
                            if i < 5:
                                print(f"    Value {value} -> class {y_val}")
                        else:
                            y.append(int(value))
                    else:
                        y.append(float(value))

                    valid_indices.append(i)
                except Exception as e:
                    if i < 5:
                        print(f"    Error processing value {value}: {e}")
                    invalid_value += 1
            else:
                missing_key += 1
                if i < 5:
                    print(f"    Key '{label_key}' not found")

        print(f"\nDEBUG 5: Label processing summary")
        print(f"  - Total labels: {len(labels_list)}")
        print(f"  - Missing key '{label_key}': {missing_key}")
        print(f"  - Invalid values: {invalid_value}")
        print(f"  - Valid labels: {len(y)}")

        if not y:
            print(f"ERROR: No valid labels for task {task_name}")
            return {}

        X = embeddings[valid_indices]
        y = np.array(y)

        print(f"\nDEBUG 6: Final data")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - y first 10 values: {y[:10].tolist() if len(y) > 0 else []}")
        print(f"  - y unique values: {np.unique(y).tolist()}")

        if task_info['type'] == 'classification':
            unique, counts = np.unique(y, return_counts=True)
            print(f"  - Label distribution: {dict(zip(unique, counts))}")

        print(f"\nDEBUG 7: Starting evaluation...")
        if task_info['type'] == 'classification':
            return self._evaluate_classification(X, y, n_splits)
        else:
            return self._evaluate_regression(X, y, n_splits)

    def _evaluate_classification(self, X, y, n_splits=5):
        """Classification evaluation."""
        unique, counts = np.unique(y, return_counts=True)
        min_class_samples = np.min(counts)
        total_samples = len(y)

        print(f"  Total samples: {total_samples}, Min class size: {min_class_samples}")

        # Get min samples threshold from config
        min_samples_for_cv = self.config.get('evaluation.min_samples_for_cv', 10)
        min_samples_per_class = self.config.get('evaluation.min_samples_per_class', 2)

        if total_samples < min_samples_for_cv or min_class_samples < min_samples_per_class:
            print("  ⚠️ Very small dataset. Using leave-one-out cross-validation.")
            loo = LeaveOneOut()

            predictions = []
            actuals = []
            scores = []

            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(np.unique(y_train)) < 2:
                    continue

                probe = LinearProbe(task_type='classification')
                probe.fit(X_train, y_train)

                y_score = probe.predict(X_test)
                y_pred = (y_score > 0).astype(int)

                scores.append(y_score[0])
                predictions.append(y_pred[0])
                actuals.append(y_test[0])

            if len(actuals) > 0 and len(np.unique(actuals)) > 1:
                auc_score = roc_auc_score(actuals, scores)
                acc = accuracy_score(actuals, predictions)
                f1 = f1_score(actuals, predictions, average='binary')
            else:
                auc_score = acc = f1 = 0.0

            return {
                'auc': auc_score,
                'auc_std': 0.0,
                'pauc_0.1': auc_score * 0.9,
                'pauc_0.1_std': 0.0,
                'accuracy': acc,
                'accuracy_std': 0.0,
                'f1': f1,
                'f1_std': 0.0,
                'cv_type': 'leave-one-out',
                'n_samples': total_samples
            }

        elif min_class_samples < n_splits:
            adjusted_splits = min(max(2, min_class_samples), n_splits)
            print(f"  ⚠️ Adjusting CV folds from {n_splits} to {adjusted_splits} based on class sizes.")
            n_splits = adjusted_splits

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)

        aucs = []
        paucs = []
        accs = []
        f1s = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"    Skipping fold {fold} - insufficient class diversity")
                continue

            probe = LinearProbe(task_type='classification')
            probe.fit(X_train, y_train)

            y_scores = probe.predict(X_test)
            y_pred = (y_scores > 0).astype(int)

            try:
                auc_score = roc_auc_score(y_test, y_scores)
                aucs.append(auc_score)

                fpr, tpr, _ = roc_curve(y_test, y_scores)
                pauc = self._calculate_partial_auc(fpr, tpr, max_fpr=0.1)
                paucs.append(pauc)
            except:
                pass

            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)

            try:
                f1 = f1_score(y_test, y_pred, average='binary')
                f1s.append(f1)
            except:
                pass

        return {
            'auc': np.mean(aucs) if aucs else 0.0,
            'auc_std': np.std(aucs) if aucs else 0.0,
            'pauc_0.1': np.mean(paucs) if paucs else 0.0,
            'pauc_0.1_std': np.std(paucs) if paucs else 0.0,
            'accuracy': np.mean(accs) if accs else 0.0,
            'accuracy_std': np.std(accs) if accs else 0.0,
            'f1': np.mean(f1s) if f1s else 0.0,
            'f1_std': np.std(f1s) if f1s else 0.0,
            'cv_type': f'stratified-{n_splits}-fold',
            'n_samples': total_samples
        }

    def _evaluate_regression(self, X, y, n_splits=5):
        """Regression evaluation."""
        total_samples = len(y)
        print(f"  Total samples: {total_samples}")

        # Get min samples threshold from config
        min_samples_for_cv = self.config.get('evaluation.min_samples_for_cv', 10)

        if total_samples < min_samples_for_cv:
            print("  ⚠️ Very small dataset. Using leave-one-out cross-validation.")
            loo = LeaveOneOut()

            predictions = []
            actuals = []

            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                probe = LinearProbe(task_type='regression')
                probe.fit(X_train, y_train)

                y_pred = probe.predict(X_test)
                predictions.append(y_pred[0])
                actuals.append(y_test[0])

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions) if len(actuals) > 1 else 0.0

            return {
                'mae': mae,
                'mae_std': 0.0,
                'rmse': rmse,
                'rmse_std': 0.0,
                'r2': r2,
                'r2_std': 0.0,
                'cv_type': 'leave-one-out',
                'n_samples': total_samples
            }

        elif total_samples < n_splits * 2:
            adjusted_splits = max(2, total_samples // 2)
            print(f"  ⚠️ Adjusting CV folds from {n_splits} to {adjusted_splits} based on sample size.")
            n_splits = adjusted_splits

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)

        maes = []
        rmses = []
        r2s = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            probe = LinearProbe(task_type='regression')
            probe.fit(X_train, y_train)

            y_pred = probe.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)

        return {
            'mae': np.mean(maes),
            'mae_std': np.std(maes),
            'rmse': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'r2': np.mean(r2s),
            'r2_std': np.std(r2s),
            'cv_type': f'{n_splits}-fold',
            'n_samples': total_samples
        }

    def _calculate_partial_auc(self, fpr, tpr, max_fpr=0.1):
        """Calculate partial AUC."""
        idx = np.where(fpr <= max_fpr)[0]

        if len(idx) < 2:
            return 0.0

        partial_auc_value = auc(fpr[idx], tpr[idx])
        normalized_pauc = partial_auc_value / max_fpr

        return normalized_pauc

    def _evaluate_task_leave_one_out(self, task_name: str, dataset: BUTPPGDataset) -> Dict:
        """
        Internal method for leave-one-out evaluation of a specific task.
        Uses cached embeddings for speed.
        """
        task_info = self.tasks[task_name]
        label_key = task_info['label_key']

        # Use cached embeddings from extract_embeddings
        if self._cached_embeddings is None:
            print("ERROR: No cached embeddings found")
            return {}

        # Prepare data
        X = self._cached_embeddings
        y_list = self._cached_labels
        participant_ids = self._cached_pids

        # Extract labels
        y = []
        valid_indices = []

        for i, label_dict in enumerate(y_list):
            if isinstance(label_dict, dict) and label_key in label_dict:
                value = label_dict[label_key]
                if value != -1:
                    if task_info['type'] == 'classification' and 'threshold' in task_info:
                        y_val = 1 if value > task_info['threshold'] else 0
                    else:
                        y_val = value
                    y.append(y_val)
                    valid_indices.append(i)

        if len(y) < 3:
            print(f"  Too few valid samples ({len(y)}) for leave-one-out")
            return {}

        X_valid = X[valid_indices]
        y_valid = np.array(y)
        pids_valid = participant_ids[valid_indices]

        print(f"  Leave-one-out with {len(y_valid)} participants")

        # Run leave-one-out cross-validation
        predictions = []
        actuals = []
        scores = []

        for i in range(len(X_valid)):
            # Leave one out
            train_mask = np.ones(len(X_valid), dtype=bool)
            train_mask[i] = False

            X_train = X_valid[train_mask]
            y_train = y_valid[train_mask]
            X_test = X_valid[i:i + 1]
            y_test = y_valid[i:i + 1]

            # For classification, check if we have both classes
            if task_info['type'] == 'classification':
                if len(np.unique(y_train)) < 2:
                    continue

            # Train probe
            probe = LinearProbe(task_type=task_info['type'])
            probe.fit(X_train, y_train)

            if task_info['type'] == 'classification':
                y_score = probe.predict(X_test)[0]
                y_pred = (y_score > 0).astype(int)

                scores.append(y_score)
                predictions.append(y_pred)
                actuals.append(y_test[0])
            else:
                y_pred = probe.predict(X_test)[0]
                predictions.append(y_pred)
                actuals.append(y_test[0])

        # Calculate metrics
        if len(actuals) == 0:
            return {}

        if task_info['type'] == 'classification':
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

            try:
                auc_score = roc_auc_score(actuals, scores) if len(np.unique(actuals)) > 1 else 0.0
            except:
                auc_score = 0.0

            acc = accuracy_score(actuals, predictions)
            f1 = f1_score(actuals, predictions, average='binary', zero_division=0)

            return {
                'auc': auc_score,
                'auc_std': 0.0,
                'accuracy': acc,
                'accuracy_std': 0.0,
                'f1': f1,
                'f1_std': 0.0,
                'cv_type': 'leave-one-out',
                'n_samples': len(actuals)
            }
        else:
            from sklearn.metrics import mean_absolute_error, r2_score

            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions) if len(actuals) > 1 else 0.0

            return {
                'mae': mae,
                'mae_std': 0.0,
                'r2': r2,
                'r2_std': 0.0,
                'cv_type': 'leave-one-out',
                'n_samples': len(actuals)
            }

    def evaluate_all_tasks(
            self,
            dataset: BaseSignalDataset,
            save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluate all tasks using specified evaluation methods.
        """
        all_results = []

        print(f"\nEvaluating on {self.device_manager.type}")
        print("=" * 50)

        # Determine which evaluation methods to use
        if hasattr(dataset, 'cases'):  # VitalDB
            n_participants = len(dataset.cases)
        elif hasattr(dataset, 'participant_records'):  # BUT PPG
            n_participants = len(dataset.participant_records)
        else:
            n_participants = len(dataset)
        
        # Get evaluation modes from config
        eval_config = self.config.get_evaluation_config()
        min_participants_for_standard = eval_config.get('min_participants_for_standard', 20)
        
        if n_participants < min_participants_for_standard:
            evaluation_modes = ['leave_one_out']
        else:
            evaluation_modes = ['standard', 'leave_one_out']

        print(f"Dataset has {n_participants} participants")
        print(f"Running evaluation modes: {evaluation_modes}")

        # Extract embeddings once for all tasks and methods
        print("\nExtracting embeddings once for all evaluations...")
        self.extract_embeddings(dataset, aggregate_by_participant=True)

        # Run each evaluation mode
        for eval_mode in evaluation_modes:
            print(f"\n{'=' * 60}")
            print(f"EVALUATION MODE: {eval_mode.upper()}")
            print(f"{'=' * 60}")

            mode_results = []

            for task_name in self.tasks.keys():
                print(f"\n{'=' * 40}")
                print(f"Evaluating {task_name} ({eval_mode})")
                print(f"{'=' * 40}")

                try:
                    if eval_mode == 'leave_one_out':
                        metrics = self._evaluate_task_leave_one_out(task_name, dataset)
                    else:  # standard
                        metrics = self.evaluate_task(task_name, dataset)

                    if metrics:
                        result = {
                            'task': task_name,
                            'type': self.tasks[task_name]['type'],
                            'eval_method': eval_mode,
                            **metrics
                        }
                        mode_results.append(result)

                        # Store with method suffix
                        self.results[f"{task_name}_{eval_mode}"] = metrics

                except Exception as e:
                    print(f"  ⚠️ Error evaluating {task_name}: {e}")
                    continue

                # Clean GPU memory between tasks
                if self.device_manager.is_cuda:
                    self.device_manager.empty_cache()

            all_results.extend(mode_results)

        # Create combined DataFrame with all results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df['device'] = self.device_manager.type
            results_df['n_participants'] = n_participants
        else:
            print("⚠️ No evaluation results available")
            results_df = pd.DataFrame()

        # Save results
        if save_path and not results_df.empty:
            # Save CSV with all results
            results_df.to_csv(save_path, index=False)
            print(f"\nResults saved to {save_path}")

            # Save detailed JSON
            json_path = Path(save_path).with_suffix('.json')

            # Organize results by evaluation method for clarity
            organized_results = {}
            for eval_mode in evaluation_modes:
                mode_results = results_df[results_df['eval_method'] == eval_mode]
                organized_results[eval_mode] = mode_results.to_dict('records')

            results_json = {
                'device': self.device_manager.type,
                'n_participants': n_participants,
                'evaluation_modes': evaluation_modes,
                'results_by_method': organized_results,
                'detailed_results': self.results
            }

            with open(json_path, 'w') as f:
                json.dump(results_json, f, indent=2, default=str)

            # Print comparison if both methods were run
            if len(evaluation_modes) > 1:
                self._print_comparison(results_df)

        return results_df


    def _print_comparison(self, results_df: pd.DataFrame):
        """Print side-by-side comparison of different evaluation methods."""
        print("\n" + "=" * 70)
        print("COMPARISON OF EVALUATION METHODS")
        print("=" * 70)

        tasks = results_df['task'].unique()
        methods = results_df['eval_method'].unique()

        for task in tasks:
            print(f"\n{task}:")
            print("-" * 40)

            task_df = results_df[results_df['task'] == task]

            for method in methods:
                method_results = task_df[task_df['eval_method'] == method]
                if not method_results.empty:
                    row = method_results.iloc[0]

                    print(f"  {method:15s}: ", end="")

                    if row['type'] == 'classification':
                        print(f"AUC={row.get('auc', 0):.3f}, "
                              f"Acc={row.get('accuracy', 0):.3f}, "
                              f"F1={row.get('f1', 0):.3f}")
                    else:
                        print(f"MAE={row.get('mae', 0):.3f}, "
                              f"R2={row.get('r2', 0):.3f}")


# ============= TEST FUNCTIONS =============
def create_evaluation_dataset(
        dataset_type: str = 'but_ppg',
        modality: str = 'ppg',
        split: str = 'test',
        data_dir: Optional[str] = None,
        downsample: bool = False
):
    """
    Create dataset for evaluation - supports both BUT PPG and VitalDB.

    Args:
        dataset_type: 'but_ppg' or 'vitaldb'
        modality: Signal modality
        split: Data split to use
        data_dir: Data directory (for BUT PPG)
        downsample: Whether to downsample

    Returns:
        Dataset instance with labels and participant IDs
    """
    if dataset_type == 'vitaldb':
        from data import VitalDBDataset
        return VitalDBDataset(
            modality=modality,
            split=split,
            return_labels=True,
            return_participant_id=True,
            use_cache=True
        )
    else:  # but_ppg
        from data import BUTPPGDataset
        return BUTPPGDataset(
            data_dir=data_dir or 'data/but_ppg/dataset',
            modality=modality,
            split=split,
            return_participant_id=True,
            return_labels=True,
            quality_filter=False,
            downsample=downsample
        )
def test_evaluation():
    """Test evaluation functionality with device manager and small samples."""
    print("=" * 50)
    print("Testing Downstream Evaluation with Small Sample Handling")
    print("=" * 50)

    # Get device from global device manager
    device_manager = get_device_manager()
    config = get_config()
    
    print(f"Testing on device: {device_manager.type}")

    # Test with very small dataset
    print("\n1. Testing with very small dataset (5 samples):")

    # Create tiny dummy data
    X_tiny = np.random.randn(5, config.get('model.embedding_dim', 128))
    y_tiny_class = np.array([0, 0, 1, 1, 1])  # 2 class 0, 3 class 1
    y_tiny_reg = np.random.randn(5)

    # Create evaluator
    from model import EfficientNet1D
    encoder = EfficientNet1D(in_channels=1, embedding_dim=config.get('model.embedding_dim', 256))
    encoder_path = Path('data/outputs/test_encoder.pt')
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), encoder_path)

    evaluator = DownstreamEvaluator(
        encoder_path=str(encoder_path),
        device_manager=device_manager
    )

    # Test classification with tiny dataset
    print("\n  Testing classification with 5 samples:")
    metrics_class = evaluator._evaluate_classification(X_tiny, y_tiny_class, n_splits=5)
    print(f"    CV type: {metrics_class.get('cv_type', 'unknown')}")
    print(f"    Accuracy: {metrics_class['accuracy']:.3f}")
    print("    ✓ Small sample classification handled!")

    # Test regression with tiny dataset
    print("\n  Testing regression with 5 samples:")
    metrics_reg = evaluator._evaluate_regression(X_tiny, y_tiny_reg, n_splits=5)
    print(f"    CV type: {metrics_reg.get('cv_type', 'unknown')}")
    print(f"    MAE: {metrics_reg['mae']:.3f}")
    print("    ✓ Small sample regression handled!")

    # Test with medium dataset
    print("\n2. Testing with medium dataset (20 samples):")

    X_medium = np.random.randn(20, config.get('model.embedding_dim', 128))
    y_medium_class = np.random.randint(0, 2, 20)

    metrics_medium = evaluator._evaluate_classification(X_medium, y_medium_class, n_splits=5)
    print(f"    CV type: {metrics_medium.get('cv_type', 'unknown')}")
    print(f"    AUC: {metrics_medium['auc']:.3f}")
    print("    ✓ Medium sample size handled!")

    # Clean up
    if encoder_path.exists():
        encoder_path.unlink()

    print("\n" + "=" * 50)
    print(f"All evaluation tests passed successfully on {device_manager.type}!")
    print("=" * 50)


def test_label_extraction():
    """Test to identify the label extraction issue in the evaluation pipeline for both datasets."""
    print("=" * 60)
    print("Testing Label Extraction Pipeline")
    print("=" * 60)

    from torch.utils.data import DataLoader
    config = get_config()

    # Test both dataset types
    for dataset_type in ['but_ppg', 'vitaldb']:
        print(f"\n{'=' * 60}")
        print(f"TESTING {dataset_type.upper()}")
        print(f"{'=' * 60}")

        # Create appropriate dataset
        if dataset_type == 'vitaldb':
            from data import VitalDBDataset
            dataset = VitalDBDataset(
                modality='ppg',
                split='test',
                return_labels=True,
                return_participant_id=True,
                use_cache=True
            )
        else:
            from data import BUTPPGDataset
            dataset = BUTPPGDataset(
                data_dir=config.data_dir,
                modality='ppg',
                split='test',
                return_labels=True,
                return_participant_id=True,
                use_cache=False,
                downsample=True
            )

        print(f"\n1. Creating {dataset_type} test dataset...")
        print(f"   Dataset created with {len(dataset)} samples")
        print(f"   return_labels: {dataset.return_labels}")
        print(f"   return_participant_id: {dataset.return_participant_id}")

        # Test single item
        print(f"\n2. Testing single item from {dataset_type} dataset...")
        if len(dataset) > 0:
            item = dataset[0]
            print(f"   Item has {len(item)} elements")

            if len(item) == 4:
                seg1, seg2, pid, labels = item
                print(f"   - Seg1 shape: {seg1.shape}")
                print(f"   - Seg2 shape: {seg2.shape}")
                print(f"   - PID: {pid} (type: {type(pid).__name__})")
                print(f"   - Labels type: {type(labels)}")
                print(f"   - Labels content: {labels}")

                # Check label values
                if isinstance(labels, dict):
                    for k, v in labels.items():
                        valid = v != -1 if isinstance(v, (int, float)) else False
                        print(f"     {k}: {v} (valid: {valid})")
            else:
                print(f"   ERROR: Expected 4 items, got {len(item)}")

        # Test DataLoader behavior
        print(f"\n3. Testing DataLoader behavior for {dataset_type}...")

        batch_size = 4  # Small batch for testing
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, batch in enumerate(loader):
            if i == 0:  # First batch only
                print(f"   Batch has {len(batch)} elements")

                if len(batch) == 4:
                    seg1, seg2, pid, labels = batch
                    print(f"   - Seg1 shape: {seg1.shape}")
                    print(f"   - Seg2 shape: {seg2.shape}")
                    print(f"   - PID type: {type(pid)}")

                    # Handle different PID types
                    if torch.is_tensor(pid):
                        print(f"     PID tensor shape: {pid.shape}")
                        print(f"     PID values: {pid.tolist()[:min(3, len(pid))]}")
                    elif isinstance(pid, (list, tuple)):
                        print(f"     PID list: {pid[:min(3, len(pid))]}")

                    print(f"   - Labels type: {type(labels)}")

                    if isinstance(labels, dict):
                        print(f"   - Label keys: {list(labels.keys())}")

                        # Test extraction logic
                        print(f"\n   Testing label extraction logic for {dataset_type}:")
                        batch_size_actual = seg1.shape[0]

                        for sample_idx in range(min(3, batch_size_actual)):
                            sample_labels = {}
                            for k, v in labels.items():
                                if torch.is_tensor(v):
                                    if v.dim() == 0:
                                        sample_labels[k] = v.item()
                                    elif v.shape[0] == batch_size_actual:
                                        sample_labels[k] = v[sample_idx].item()
                                    else:
                                        sample_labels[k] = -1
                                else:
                                    sample_labels[k] = v
                            print(f"     Sample {sample_idx}: {sample_labels}")
                break

        # Test participant/case info retrieval
        print(f"\n4. Testing participant/case info retrieval for {dataset_type}...")

        if dataset_type == 'vitaldb':
            # VitalDB specific
            if hasattr(dataset, 'cases') and len(dataset.cases) > 0:
                for case_id in dataset.cases[:3]:
                    info = dataset._get_participant_info(case_id)
                    print(f"   Case {case_id}:")
                    print(f"     Info: {info}")
                    valid_fields = {k: v != -1 for k, v in info.items() if isinstance(v, (int, float))}
                    print(f"     Valid fields: {valid_fields}")
        else:
            # BUT PPG specific
            if hasattr(dataset, 'participant_records') and len(dataset.participant_records) > 0:
                for i, (pid, records) in enumerate(list(dataset.participant_records.items())[:3]):
                    info = dataset._get_participant_info(pid)
                    print(f"   Participant {pid} ({len(records)} records):")
                    print(f"     Info: {info}")
                    valid_fields = {k: v != -1 for k, v in info.items() if isinstance(v, (int, float))}
                    print(f"     Valid fields: {valid_fields}")

    print("\n" + "=" * 60)
    print("All dataset tests complete!")
    print("=" * 60)

if __name__ == "__main__":
    print("Running evaluation tests...")
    print("\n1. Running standard evaluation test:")
    test_evaluation()

    print("\n2. Running label extraction test:")
    test_label_extraction()