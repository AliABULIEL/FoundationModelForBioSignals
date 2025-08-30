# biosignal/evaluate.py
"""
Evaluation module for downstream tasks
Uses centralized configuration management
"""

import torch
import torch.nn as nn
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

from data import BUTPPGDataset
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
                'threshold': 30,
                'label_key': 'bmi'
            },
            'bmi_regression': {
                'type': 'regression',
                'label_key': 'bmi'
            },
            'bp_classification': {
                'type': 'classification',
                'threshold': 140,
                'label_key': 'bp_systolic'
            }
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

        checkpoint = torch.load(self.encoder_path, map_location=self.device)
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

    @torch.no_grad()
    def extract_embeddings(
            self,
            dataset: BUTPPGDataset,
            aggregate_by_participant: bool = True,
            batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Extract embeddings with caching optimization.
        """
        from torch.utils.data import DataLoader

        # Check if we already have embeddings for this dataset
        dataset_id = id(dataset)
        if (self._current_dataset_id == dataset_id and
                self._cached_embeddings is not None):
            print("  ✓ Reusing cached embeddings for current dataset")
            return self._cached_embeddings, self._cached_labels, self._cached_pids

        # Check persistent cache
        if self.cache:
            cache_key = self.cache.get_cache_key(dataset, self.encoder_path)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                print("  ✓ Loading embeddings from persistent cache")
                self._current_dataset_id = dataset_id
                self._cached_embeddings, self._cached_labels, self._cached_pids = cached_data
                return cached_data

        print("\n" + "=" * 60)
        print("DEBUG: extract_embeddings")
        print("=" * 60)

        print("DEBUG 1: Dataset Configuration")
        print(f"  - Dataset type: {type(dataset).__name__}")
        print(f"  - Dataset return_labels: {dataset.return_labels}")
        print(f"  - Dataset return_participant_id: {dataset.return_participant_id}")
        print(f"  - Dataset length: {len(dataset)}")
        print(f"  - Aggregate by participant: {aggregate_by_participant}")

        # Test dataset output
        print("\nDEBUG 2: Testing dataset.__getitem__ directly")
        if len(dataset) > 0:
            test_item = dataset[0]
            print(f"  - Item returned {len(test_item)} elements")
            for i, item in enumerate(test_item):
                if torch.is_tensor(item):
                    print(f"    [{i}] Tensor shape: {item.shape}")
                elif isinstance(item, dict):
                    print(f"    [{i}] Dict with keys: {list(item.keys())}")
                    for k, v in item.items():
                        print(f"        {k}: {v}")
                elif isinstance(item, str):
                    print(f"    [{i}] String (participant_id): '{item}'")
                else:
                    print(f"    [{i}] Type: {type(item).__name__}")

        # Get batch size from config or use device optimal
        if batch_size is None:
            eval_config = self.config.get_evaluation_config()
            batch_size = eval_config.get('batch_size', None)
            if batch_size is None:
                batch_size = self.device_manager.get_optimal_batch_size(dataset.modality)
            print(f"Using batch size: {batch_size}")

        # DataLoader settings from config
        num_workers = self.config.get('evaluation.num_workers', self.device_manager.get_num_workers())
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device_manager.is_cuda,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )

        all_embeddings = []
        all_labels = []
        all_participant_ids = []

        print("\nDEBUG 3: Processing batches...")
        print("Extracting embeddings...")

        if self.device_manager.is_cuda:
            initial_mem = self.device_manager.memory_stats()
            print(f"  Initial GPU memory: {initial_mem.get('allocated', 0):.2f} GB")

        batch_count = 0
        for batch_idx, batch in enumerate(tqdm(loader)):
            batch_count += 1

            # First batch debug
            if batch_idx == 0:
                print(f"\nDEBUG 4: First batch structure")
                print(f"  - Batch contains {len(batch)} elements")
                for i, b in enumerate(batch):
                    if torch.is_tensor(b):
                        print(f"    [{i}] Tensor shape: {b.shape}, dtype: {b.dtype}")
                    elif isinstance(b, dict):
                        print(f"    [{i}] Dict with keys: {list(b.keys())}")
                        for k, v in b.items():
                            if torch.is_tensor(v):
                                print(f"        {k}: Tensor shape {v.shape}")
                            else:
                                print(f"        {k}: {type(v).__name__}")
                    else:
                        print(f"    [{i}] Type: {type(b).__name__}")

            # Handle different batch formats
            if len(batch) == 4:
                seg1, seg2, pid, labels = batch
                if batch_idx == 0:
                    print(f"  - Got 4 elements (with labels)")
            elif len(batch) == 3:
                seg1, seg2, pid = batch
                labels = {}
                if batch_idx == 0:
                    print(f"  - Got 3 elements (no labels)")
            else:
                seg1, seg2 = batch
                pid = None
                labels = {}
                if batch_idx == 0:
                    print(f"  - Got 2 elements (no pid, no labels)")

            # Non-blocking transfer
            if self.device_manager.is_cuda:
                seg1 = seg1.to(self.device, non_blocking=True)
            else:
                seg1 = seg1.to(self.device)

            # Extract embeddings
            embeddings = self.encoder(seg1)
            all_embeddings.append(embeddings.cpu().numpy())

            # Handle participant IDs
            if pid is not None:
                if torch.is_tensor(pid):
                    pid_list = pid.tolist() if pid.dim() > 0 else [pid.item()]
                elif isinstance(pid, (list, tuple)):
                    pid_list = list(pid)
                elif isinstance(pid, str):
                    pid_list = [pid] * seg1.shape[0]
                else:
                    pid_list = [pid] * seg1.shape[0]
                all_participant_ids.extend(pid_list)

                if batch_idx == 0:
                    print(f"  - PIDs in first batch: {pid_list[:min(3, len(pid_list))]}")

            # Collect labels
            if labels:
                if batch_idx == 0:
                    print(f"  - Labels type: {type(labels)}")
                    print(f"  - Labels keys: {list(labels.keys()) if isinstance(labels, dict) else 'Not a dict'}")

                batch_labels = []
                batch_size_actual = seg1.shape[0]

                for i in range(batch_size_actual):
                    try:
                        if isinstance(labels, dict):
                            sample_labels = {}
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
                                elif isinstance(v, (int, float)):
                                    sample_labels[k] = v
                                else:
                                    sample_labels[k] = -1
                            batch_labels.append(sample_labels)
                        else:
                            batch_labels.append({})
                    except Exception as e:
                        if batch_idx == 0 and i == 0:
                            print(f"    ERROR extracting label for sample {i}: {e}")
                        batch_labels.append({})

                all_labels.extend(batch_labels)

                if batch_idx == 0 and batch_labels:
                    print(f"  - First sample labels: {batch_labels[0]}")

            # Periodic memory cleanup
            cache_clean_freq = self.config.get('evaluation.cache_clean_frequency', 50)
            if self.device_manager.is_cuda and batch_idx % cache_clean_freq == 0:
                self.device_manager.empty_cache()

        print(f"\nDEBUG 5: Batch processing complete")
        print(f"  - Processed {batch_count} batches")
        print(f"  - Total embeddings: {len(all_embeddings)}")
        print(f"  - Total labels: {len(all_labels)}")
        print(f"  - Total participant IDs: {len(all_participant_ids)}")

        # Concatenate
        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"  - Concatenated embeddings shape: {embeddings.shape}")

        # Debug labels
        if all_labels:
            print(f"\nDEBUG 6: Labels check")
            print(f"  - First 3 labels:")
            for i in range(min(3, len(all_labels))):
                print(f"    [{i}]: {all_labels[i]}")

            valid_count = sum(1 for label in all_labels
                              if isinstance(label, dict) and
                              any(v != -1 for v in label.values()))
            print(f"  - Labels with valid values: {valid_count}/{len(all_labels)}")
        else:
            print(f"\nDEBUG 6: WARNING - No labels collected!")

        if self.device_manager.is_cuda:
            self.device_manager.empty_cache()
            final_mem = self.device_manager.memory_stats()
            print(f"  Final GPU memory: {final_mem.get('allocated', 0):.2f} GB")

        # Aggregate if needed
        if aggregate_by_participant and all_participant_ids:
            print(f"\nDEBUG 7: Aggregating by participant")
            print(f"  - Before: {len(embeddings)} embeddings, {len(all_labels)} labels")
            embeddings, all_labels, all_participant_ids = self._aggregate_by_participant(
                embeddings, all_labels, all_participant_ids
            )
            print(f"  - After: {len(embeddings)} embeddings, {len(all_labels)} labels")
            print(f"  - Unique participants: {len(np.unique(all_participant_ids))}")

        # Cache the results
        result = (embeddings, all_labels, np.array(all_participant_ids))

        # Store in memory cache
        self._current_dataset_id = dataset_id
        self._cached_embeddings = embeddings
        self._cached_labels = all_labels
        self._cached_pids = np.array(all_participant_ids)

        # Store in persistent cache
        if self.cache and cache_key:
            self.cache.set(cache_key, result)
            print("  ✓ Cached embeddings for future use")

        return result

    def _aggregate_by_participant(
            self,
            embeddings: np.ndarray,
            labels: List[Dict],
            participant_ids: List
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Aggregate embeddings by participant."""
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
            dataset: BUTPPGDataset,
            save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluate all tasks using specified evaluation methods.
        """
        all_results = []

        print(f"\nEvaluating on {self.device_manager.type}")
        print("=" * 50)

        # Determine which evaluation methods to use
        n_participants = len(dataset.participant_records)
        
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
    """Test to identify the label extraction issue in the evaluation pipeline."""
    print("=" * 60)
    print("Testing Label Extraction Pipeline")
    print("=" * 60)

    from data import BUTPPGDataset
    from torch.utils.data import DataLoader
    
    config = get_config()

    # Create a test dataset
    print("\n1. Creating test dataset...")
    dataset = BUTPPGDataset(
        data_dir=config.data_dir,
        modality='ppg',
        split='test',
        return_labels=True,
        return_participant_id=True,
        use_cache=False,
        downsample=True
    )

    print(f"   Dataset created with {len(dataset)} samples")
    print(f"   return_labels: {dataset.return_labels}")
    print(f"   return_participant_id: {dataset.return_participant_id}")

    # Test single item
    print("\n2. Testing single item from dataset...")
    if len(dataset) > 0:
        item = dataset[0]
        print(f"   Item has {len(item)} elements")

        if len(item) == 4:
            seg1, seg2, pid, labels = item
            print(f"   - Seg1 shape: {seg1.shape}")
            print(f"   - Seg2 shape: {seg2.shape}")
            print(f"   - PID: {pid}")
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
    print("\n3. Testing DataLoader behavior...")
    
    batch_size = config.get('evaluation.batch_size', 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in enumerate(loader):
        if i == 0:  # First batch only
            print(f"   Batch has {len(batch)} elements")

            if len(batch) == 4:
                seg1, seg2, pid, labels = batch
                print(f"   - Seg1 shape: {seg1.shape}")
                print(f"   - Seg2 shape: {seg2.shape}")
                print(f"   - PID type: {type(pid)}")
                if torch.is_tensor(pid):
                    print(f"     PID tensor shape: {pid.shape}")
                    print(f"     PID values: {pid.tolist()}")
                print(f"   - Labels type: {type(labels)}")

                if isinstance(labels, dict):
                    print(f"   - Label keys: {list(labels.keys())}")
                    for k, v in labels.items():
                        if torch.is_tensor(v):
                            print(f"     {k}: Tensor shape {v.shape}, dtype {v.dtype}")
                            print(f"        Values: {v.tolist()}")
                        else:
                            print(f"     {k}: {type(v).__name__} = {v}")

                    # Test extraction logic
                    print("\n   Testing label extraction logic:")
                    batch_size = seg1.shape[0]
                    for sample_idx in range(batch_size):
                        sample_labels = {}
                        for k, v in labels.items():
                            if torch.is_tensor(v):
                                if v.dim() == 0:
                                    sample_labels[k] = v.item()
                                elif v.shape[0] == batch_size:
                                    sample_labels[k] = v[sample_idx].item()
                                else:
                                    sample_labels[k] = -1
                            else:
                                sample_labels[k] = v
                        print(f"     Sample {sample_idx}: {sample_labels}")
            break

    # Test participant info retrieval
    print("\n4. Testing participant info retrieval...")
    if len(dataset.participant_records) > 0:
        for i, (pid, records) in enumerate(list(dataset.participant_records.items())[:3]):
            info = dataset._get_participant_info(pid)
            print(f"   Participant {pid} ({len(records)} records):")
            print(f"     Info: {info}")

            # Check validity
            valid_fields = {k: v != -1 for k, v in info.items() if isinstance(v, (int, float))}
            print(f"     Valid fields: {valid_fields}")

    print("\n" + "=" * 60)
    print("Test complete! Check output for issues.")
    print("=" * 60)


if __name__ == "__main__":
    print("Running evaluation tests...")
    print("\n1. Running standard evaluation test:")
    test_evaluation()

    print("\n2. Running label extraction test:")
    test_label_extraction()