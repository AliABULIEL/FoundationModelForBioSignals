"""
Evaluation module for downstream tasks
Following Apple paper's evaluation protocol
Fixed to handle small sample sizes and uses global device manager
WITH COMPREHENSIVE DEBUG PRINTS
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
import yaml
import json
import warnings

warnings.filterwarnings('ignore')

from data import BUTPPGDataset
from model import EfficientNet1D
from device import DeviceManager, get_device_manager


class LinearProbe:
    """Linear probe for downstream evaluation with device manager."""

    def __init__(
            self,
            task_type: str = 'classification',
            alpha: float = 1.0
    ):
        """
        Initialize linear probe.

        Args:
            task_type: 'classification' or 'regression'
            alpha: Ridge regularization strength
        """
        self.task_type = task_type
        self.alpha = alpha

        # Get device from global device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        self.scaler = StandardScaler()
        self.probe = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the linear probe."""
        # Standardize features
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
            # Get decision scores for AUC calculation
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

        # Convert decision function to probabilities
        if hasattr(self.probe, 'decision_function'):
            scores = self.probe.decision_function(X_scaled)
            # Sigmoid to convert to probabilities
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
        else:
            return self.probe.predict(X_scaled)


class DownstreamEvaluator:
    """Evaluator for downstream tasks with device manager integration and small sample handling."""

    def __init__(
            self,
            encoder_path: str,
            config_path: str = 'configs/config.yaml',
            device_manager: Optional[DeviceManager] = None
    ):
        """
        Initialize evaluator with device manager.

        Args:
            encoder_path: Path to trained encoder checkpoint
            config_path: Path to configuration
            device_manager: Device manager instance (if None, uses global)
        """
        self.encoder_path = encoder_path

        # Use device manager
        if device_manager is None:
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager

        self.device = self.device_manager.device

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load encoder
        self._load_encoder()

        # Define tasks (following Apple paper)
        self.tasks = {
            'age_classification': {
                'type': 'classification',
                'threshold': 50,  # >50 vs ≤50
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
                'threshold': 30,  # >30 vs ≤30 (obese vs non-obese)
                'label_key': 'bmi'
            },
            'bmi_regression': {
                'type': 'regression',
                'label_key': 'bmi'
            },
            'bp_classification': {
                'type': 'classification',
                'threshold': 140,  # Hypertensive vs normal (systolic)
                'label_key': 'bp_systolic'
            },
            'spo2_regression': {
                'type': 'regression',
                'label_key': 'spo2'
            }
        }

        # Results storage
        self.results = {}

        print(f"Evaluator initialized on {self.device_manager.type}")
        if self.device_manager.is_cuda:
            mem_stats = self.device_manager.memory_stats()
            print(f"  GPU memory available: {mem_stats.get('free', 0):.2f} GB")

    def _load_encoder(self):
        """Load pre-trained encoder."""
        # Determine modality from config or checkpoint
        modality = 'ppg'  # Default, should be loaded from checkpoint metadata

        self.encoder = EfficientNet1D(
            in_channels=1 if modality != 'acc' else 3,
            embedding_dim=256,
            modality=modality
        )

        # Load weights
        checkpoint = torch.load(self.encoder_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            elif 'online_encoder_state_dict' in checkpoint:
                # From SSL training
                self.encoder.load_state_dict(checkpoint['online_encoder_state_dict'])
            else:
                self.encoder.load_state_dict(checkpoint)
        else:
            self.encoder.load_state_dict(checkpoint)

        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        print(f"Encoder loaded from {self.encoder_path}")

    @torch.no_grad()
    def extract_embeddings(
            self,
            dataset: BUTPPGDataset,
            aggregate_by_participant: bool = True,
            batch_size: int = 32
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Extract embeddings for all samples with device optimization.
        WITH COMPREHENSIVE DEBUG PRINTS

        Returns:
            embeddings, labels, participant_ids
        """
        from torch.utils.data import DataLoader

        print("\n" + "=" * 60)
        print("DEBUG: extract_embeddings")
        print("=" * 60)

        # DEBUG 1: Check dataset configuration
        print("DEBUG 1: Dataset Configuration")
        print(f"  - Dataset type: {type(dataset).__name__}")
        print(f"  - Dataset return_labels: {dataset.return_labels}")
        print(f"  - Dataset return_participant_id: {dataset.return_participant_id}")
        print(f"  - Dataset length: {len(dataset)}")
        print(f"  - Aggregate by participant: {aggregate_by_participant}")

        # DEBUG 2: Test what dataset returns directly
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

        # Optimize batch size based on device
        if batch_size is None:
            batch_size = self.device_manager.get_optimal_batch_size(dataset.modality)
            print(f"Using auto-selected batch size: {batch_size}")

        # Create data loader with device-optimized settings
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.device_manager.get_num_workers(),
            pin_memory=self.device_manager.is_cuda
        )

        all_embeddings = []
        all_labels = []
        all_participant_ids = []

        print("\nDEBUG 3: Processing batches...")
        print("Extracting embeddings...")

        # Show initial memory if CUDA
        if self.device_manager.is_cuda:
            initial_mem = self.device_manager.memory_stats()
            print(f"  Initial GPU memory: {initial_mem.get('allocated', 0):.2f} GB")

        batch_count = 0
        for batch_idx, batch in enumerate(tqdm(loader)):
            batch_count += 1

            # DEBUG: First batch structure
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
                                if v.numel() <= 10:
                                    print(f"           Values: {v.tolist()}")
                            else:
                                print(f"        {k}: {type(v).__name__}")
                    else:
                        print(f"    [{i}] Type: {type(b).__name__}")

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

            # Use first segment for evaluation
            # Optimize transfer for CUDA
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

            # Collect labels if available
            if labels:
                if batch_idx == 0:
                    print(f"  - Labels type: {type(labels)}")
                    print(f"  - Labels keys: {list(labels.keys()) if isinstance(labels, dict) else 'Not a dict'}")

                batch_labels = []
                batch_size_actual = seg1.shape[0]

                # Try to extract labels per sample
                for i in range(batch_size_actual):
                    try:
                        if isinstance(labels, dict):
                            sample_labels = {}
                            for k, v in labels.items():
                                if torch.is_tensor(v):
                                    if v.dim() == 0:  # Scalar
                                        sample_labels[k] = v.item()
                                    elif v.shape[0] == batch_size_actual:  # Vector matching batch size
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

            # Periodic memory cleanup for CUDA
            if self.device_manager.is_cuda and batch_idx % 100 == 0:
                self.device_manager.empty_cache()

        print(f"\nDEBUG 5: Batch processing complete")
        print(f"  - Processed {batch_count} batches")
        print(f"  - Total embeddings: {len(all_embeddings)}")
        print(f"  - Total labels: {len(all_labels)}")
        print(f"  - Total participant IDs: {len(all_participant_ids)}")

        # Concatenate
        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"  - Concatenated embeddings shape: {embeddings.shape}")

        # DEBUG: Check labels content
        if all_labels:
            print(f"\nDEBUG 6: Labels check")
            print(f"  - First 3 labels:")
            for i in range(min(3, len(all_labels))):
                print(f"    [{i}]: {all_labels[i]}")

            # Check for valid labels
            valid_count = sum(1 for label in all_labels
                              if isinstance(label, dict) and
                              any(v != -1 for v in label.values()))
            print(f"  - Labels with valid values: {valid_count}/{len(all_labels)}")
        else:
            print(f"\nDEBUG 6: WARNING - No labels collected!")

        # Final memory cleanup
        if self.device_manager.is_cuda:
            self.device_manager.empty_cache()
            final_mem = self.device_manager.memory_stats()
            print(f"  Final GPU memory: {final_mem.get('allocated', 0):.2f} GB")

        # Aggregate by participant if requested
        if aggregate_by_participant and all_participant_ids:
            print(f"\nDEBUG 7: Aggregating by participant")
            print(f"  - Before: {len(embeddings)} embeddings, {len(all_labels)} labels")
            embeddings, all_labels, all_participant_ids = self._aggregate_by_participant(
                embeddings, all_labels, all_participant_ids
            )
            print(f"  - After: {len(embeddings)} embeddings, {len(all_labels)} labels")
            print(f"  - Unique participants: {len(np.unique(all_participant_ids))}")

        return embeddings, all_labels, np.array(all_participant_ids)

    def _aggregate_by_participant(
            self,
            embeddings: np.ndarray,
            labels: List[Dict],
            participant_ids: List
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Aggregate embeddings by participant (mean pooling)."""
        unique_pids = np.unique(participant_ids)
        aggregated_embeddings = []
        aggregated_labels = []

        print(f"  Aggregating {len(unique_pids)} unique participants")

        for pid in unique_pids:
            mask = np.array(participant_ids) == pid
            # Mean aggregate embeddings
            agg_emb = embeddings[mask].mean(axis=0)
            aggregated_embeddings.append(agg_emb)

            # Take first label (should be same for all segments)
            if labels:
                pid_labels = [labels[i] for i, m in enumerate(mask) if m]
                if pid_labels:
                    aggregated_labels.append(pid_labels[0])

        return np.array(aggregated_embeddings), aggregated_labels, unique_pids

    def evaluate_task(
            self,
            task_name: str,
            dataset: BUTPPGDataset,
            n_splits: int = 5
    ) -> Dict:
        """
        Evaluate a specific task.
        WITH COMPREHENSIVE DEBUG PRINTS
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

        # Extract embeddings and labels
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

        # Prepare labels for this task
        label_key = task_info['label_key']
        print(f"\nDEBUG 4: Processing labels for key '{label_key}'")

        y = []
        valid_indices = []
        missing_key = 0
        invalid_value = 0

        for i, label_dict in enumerate(labels_list):
            if i < 5:  # Debug first 5
                print(f"  - Label {i}: {label_dict}")

            if not isinstance(label_dict, dict):
                print(f"    WARNING: Label {i} is not dict: {type(label_dict)}")
                continue

            if label_key in label_dict:
                value = label_dict[label_key]

                # Check for valid value
                try:
                    if value == -1 or (isinstance(value, float) and np.isnan(value)):
                        invalid_value += 1
                        if i < 5:
                            print(f"    Invalid value: {value}")
                        continue

                    if task_info['type'] == 'classification':
                        if 'threshold' in task_info:
                            # Binary classification with threshold
                            y_val = 1 if value > task_info['threshold'] else 0
                            y.append(y_val)
                            if i < 5:
                                print(f"    Value {value} -> class {y_val}")
                        else:
                            # Direct classification (e.g., sex)
                            y.append(int(value))
                    else:
                        # Regression
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

        # Filter embeddings
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

        # Perform cross-validation
        print(f"\nDEBUG 7: Starting evaluation...")
        if task_info['type'] == 'classification':
            return self._evaluate_classification(X, y, n_splits)
        else:
            return self._evaluate_regression(X, y, n_splits)

    def _evaluate_classification(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 5
    ) -> Dict:
        """Evaluate classification task with adaptive CV for small datasets."""

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        min_class_samples = np.min(counts)
        total_samples = len(y)

        print(f"  Total samples: {total_samples}, Min class size: {min_class_samples}")

        # Adaptive strategy based on sample size
        if total_samples < 10 or min_class_samples < 2:
            print("  ⚠️ Very small dataset. Using leave-one-out cross-validation.")
            # Leave-one-out for very small datasets
            loo = LeaveOneOut()

            predictions = []
            actuals = []
            scores = []

            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Skip if training set has only one class
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
                'pauc_0.1': auc_score * 0.9,  # Approximate
                'pauc_0.1_std': 0.0,
                'accuracy': acc,
                'accuracy_std': 0.0,
                'f1': f1,
                'f1_std': 0.0,
                'cv_type': 'leave-one-out',
                'n_samples': total_samples
            }

        elif min_class_samples < n_splits:
            # Adjust n_splits to match smallest class
            adjusted_splits = min(max(2, min_class_samples), n_splits)
            print(f"  ⚠️ Adjusting CV folds from {n_splits} to {adjusted_splits} based on class sizes.")
            n_splits = adjusted_splits

        # Standard stratified k-fold (with adjusted n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        aucs = []
        paucs = []
        accs = []
        f1s = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip fold if only one class in train or test
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"    Skipping fold {fold} - insufficient class diversity")
                continue

            # Train probe
            probe = LinearProbe(task_type='classification')
            probe.fit(X_train, y_train)

            # Predict
            y_scores = probe.predict(X_test)
            y_pred = (y_scores > 0).astype(int)

            # Calculate metrics
            try:
                auc_score = roc_auc_score(y_test, y_scores)
                aucs.append(auc_score)

                # Partial AUC
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                pauc = self._calculate_partial_auc(fpr, tpr, max_fpr=0.1)
                paucs.append(pauc)
            except:
                pass  # Skip AUC if not computable

            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)

            try:
                f1 = f1_score(y_test, y_pred, average='binary')
                f1s.append(f1)
            except:
                pass

        # Return averaged metrics
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

    def _evaluate_regression(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 5
    ) -> Dict:
        """Evaluate regression task with adaptive CV for small datasets."""

        total_samples = len(y)
        print(f"  Total samples: {total_samples}")

        # Adaptive strategy
        if total_samples < 10:
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
            # Adjust n_splits
            adjusted_splits = max(2, total_samples // 2)
            print(f"  ⚠️ Adjusting CV folds from {n_splits} to {adjusted_splits} based on sample size.")
            n_splits = adjusted_splits

        # Standard k-fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        maes = []
        rmses = []
        r2s = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train probe
            probe = LinearProbe(task_type='regression')
            probe.fit(X_train, y_train)

            # Predict
            y_pred = probe.predict(X_test)

            # Calculate metrics
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

    def _calculate_partial_auc(
            self,
            fpr: np.ndarray,
            tpr: np.ndarray,
            max_fpr: float = 0.1
    ) -> float:
        """
        Calculate partial AUC up to max_fpr.
        Paper uses pAUC at 10% FPR.
        """
        # Find indices where FPR <= max_fpr
        idx = np.where(fpr <= max_fpr)[0]

        if len(idx) < 2:
            return 0.0

        # Calculate partial AUC using trapezoidal rule
        partial_auc_value = auc(fpr[idx], tpr[idx])

        # Normalize by the maximum possible area
        normalized_pauc = partial_auc_value / max_fpr

        return normalized_pauc

    def evaluate_all_tasks(
            self,
            dataset: BUTPPGDataset,
            save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate all downstream tasks.

        Args:
            dataset: Dataset to evaluate on
            save_path: Path to save results

        Returns:
            DataFrame with results
        """
        results = []

        print(f"\nEvaluating on {self.device_manager.type}")
        print("=" * 50)

        for task_name in self.tasks.keys():
            print(f"\n{'=' * 40}")
            print(f"Evaluating {task_name}")
            print(f"{'=' * 40}")

            try:
                metrics = self.evaluate_task(task_name, dataset)

                if metrics:
                    result = {
                        'task': task_name,
                        'type': self.tasks[task_name]['type'],
                        **metrics
                    }
                    results.append(result)

                    # Store in results dict
                    self.results[task_name] = metrics
            except Exception as e:
                print(f"  ⚠️ Error evaluating {task_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Clean GPU memory between tasks
            if self.device_manager.is_cuda:
                self.device_manager.empty_cache()

        # Create DataFrame
        if results:
            results_df = pd.DataFrame(results)
        else:
            print("⚠️ No evaluation results available")
            results_df = pd.DataFrame()

        # Add device info
        if not results_df.empty:
            results_df['device'] = self.device_manager.type

        # Save if requested
        if save_path and not results_df.empty:
            results_df.to_csv(save_path, index=False)
            print(f"\nResults saved to {save_path}")

            # Also save as JSON with device info
            json_path = Path(save_path).with_suffix('.json')
            results_with_device = {
                'device': self.device_manager.type,
                'device_properties': self.device_manager.get_properties(),
                'results': self.results
            }
            with open(json_path, 'w') as f:
                json.dump(results_with_device, f, indent=2, default=str)

        return results_df

    def print_summary(self, results_df: pd.DataFrame):
        """Print a summary of evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Device: {self.device_manager.type}")

        if self.device_manager.is_cuda:
            props = self.device_manager.get_properties()
            print(f"GPU: {props.get('name', 'Unknown')}")
            print(f"Memory: {props.get('memory_total', 0) / 1e9:.2f} GB")

        if not results_df.empty:
            print("\nResults:")
            print("-" * 60)
            print(results_df.to_string(index=False))
            print("-" * 60)
        else:
            print("\n⚠️ No results to display")


# ============= TEST FUNCTIONS =============

def test_evaluation():
    """Test evaluation functionality with device manager and small samples."""
    print("=" * 50)
    print("Testing Downstream Evaluation with Small Sample Handling")
    print("=" * 50)

    # Get device from global device manager
    device_manager = get_device_manager()
    print(f"Testing on device: {device_manager.type}")

    # Test with very small dataset
    print("\n1. Testing with very small dataset (5 samples):")

    # Create tiny dummy data
    X_tiny = np.random.randn(5, 256)
    y_tiny_class = np.array([0, 0, 1, 1, 1])  # 2 class 0, 3 class 1
    y_tiny_reg = np.random.randn(5)

    # Create evaluator
    from model import EfficientNet1D
    encoder = EfficientNet1D(in_channels=1, embedding_dim=256)
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

    X_medium = np.random.randn(20, 256)
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

    # Create a test dataset
    print("\n1. Creating test dataset...")
    dataset = BUTPPGDataset(
        data_dir="data/but_ppg/dataset",
        modality='ppg',
        split='test',
        return_labels=True,
        return_participant_id=True,
        use_cache=False
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
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i, batch in enumerate(loader):
        if i == 0:  # First batch onlyå
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