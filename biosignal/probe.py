# biosignal/evaluate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from .model import EfficientNet1D, BiosignalFoundationModel
from .data import MIMICWaveformDataset


class LinearProbe:
    """
    Linear probe for downstream evaluation following the paper's approach.
    Paper Section 4.2: "We perform linear probing for predicting self-reported age, 
    body mass index (BMI), and biological sex using ridge regression"
    """

    def __init__(
            self,
            encoder: Optional[nn.Module] = None,
            encoder_checkpoint: Optional[str] = None,
            task: str = 'classification',
            modality: str = 'ecg',
            embedding_dim: int = 256,
            alpha: float = 1.0,  # Ridge regularization
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize linear probe.

        Args:
            encoder: Pre-trained encoder model (if already loaded)
            encoder_checkpoint: Path to encoder checkpoint (if encoder not provided)
            task: 'classification' or 'regression'
            modality: 'ppg' or 'ecg'
            embedding_dim: Dimension of encoder output (256 in paper)
            alpha: Ridge regression regularization strength
            device: Device to use
        """
        self.task = task.lower()
        self.modality = modality
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.device = device

        # Load encoder
        if encoder is not None:
            self.encoder = encoder
        elif encoder_checkpoint is not None:
            self.encoder = self._load_encoder(encoder_checkpoint, modality)
        else:
            raise ValueError("Either encoder or encoder_checkpoint must be provided")

        # Move encoder to device and freeze
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Initialize sklearn models (will be fitted later)
        self.probe = None
        self.scaler = StandardScaler()

    def _load_encoder(self, checkpoint_path: str, modality: str) -> nn.Module:
        """Load pre-trained encoder from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Try different checkpoint formats
        if 'online_encoder_state_dict' in checkpoint:
            # From SSL training
            encoder = EfficientNet1D(modality=modality)
            encoder.load_state_dict(checkpoint['online_encoder_state_dict'])
        elif 'encoder_state_dict' in checkpoint:
            encoder = EfficientNet1D(modality=modality)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model = BiosignalFoundationModel(modality=modality)
            model.load_state_dict(checkpoint['model_state_dict'])
            encoder = model.encoder
        else:
            # Direct encoder weights
            encoder = EfficientNet1D(modality=modality)
            encoder.load_state_dict(checkpoint)

        return encoder

    @torch.no_grad()
    def extract_embeddings(
            self,
            dataloader: torch.utils.data.DataLoader,
            aggregate_by_participant: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract embeddings for all samples in dataloader.
        Paper: "mean-aggregate all embeddings associated to each participant"

        Returns:
            embeddings, labels, participant_ids (if available)
        """
        self.encoder.eval()

        all_embeddings = []
        all_labels = []
        all_participant_ids = []

        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Handle different batch formats
            if len(batch) == 4:
                # seg1, seg2, participant_id, labels_dict
                x1, x2, pids, labels_dict = batch
                # Use first segment for evaluation
                x = x1
                # Extract specific label if dict
                if isinstance(labels_dict, dict):
                    # For now, skip if labels are dict (handle separately)
                    continue
                else:
                    labels = labels_dict
                participant_ids = pids
            elif len(batch) == 3:
                # seg1, seg2, participant_id
                x1, x2, pids = batch
                x = x1
                labels = None
                participant_ids = pids
            elif len(batch) == 2:
                # Could be (seg1, seg2) or (x, y)
                if isinstance(batch[1], torch.Tensor) and batch[1].dtype == torch.float32:
                    # Assume it's two segments
                    x, _ = batch
                    labels = None
                    participant_ids = None
                else:
                    # Assume it's (x, y)
                    x, labels = batch
                    participant_ids = None
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            # Move to device
            x = x.to(self.device)

            # Extract embeddings
            embeddings = self.encoder(x)
            all_embeddings.append(embeddings.cpu().numpy())

            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    all_labels.append(labels.cpu().numpy())
                else:
                    all_labels.append(np.array(labels))

            if participant_ids is not None:
                if isinstance(participant_ids, torch.Tensor):
                    all_participant_ids.extend(participant_ids.cpu().numpy())
                else:
                    all_participant_ids.extend(participant_ids)

        # Concatenate all
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else None
        participant_ids = np.array(all_participant_ids) if all_participant_ids else None

        # Aggregate by participant if requested and possible
        if aggregate_by_participant and participant_ids is not None:
            embeddings, labels, participant_ids = self._aggregate_by_participant(
                embeddings, labels, participant_ids
            )

        return embeddings, labels, participant_ids

    def _aggregate_by_participant(
            self,
            embeddings: np.ndarray,
            labels: Optional[np.ndarray],
            participant_ids: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Aggregate embeddings by participant (mean pooling).
        Paper: "we mean-aggregate all the embeddings associated to each participant"
        """
        unique_pids = np.unique(participant_ids)
        aggregated_embeddings = []
        aggregated_labels = []

        for pid in unique_pids:
            mask = participant_ids == pid
            # Mean aggregate embeddings
            agg_emb = embeddings[mask].mean(axis=0)
            aggregated_embeddings.append(agg_emb)

            if labels is not None:
                # Take first label (should be same for all segments of participant)
                agg_label = labels[mask][0]
                aggregated_labels.append(agg_label)

        aggregated_embeddings = np.array(aggregated_embeddings)
        aggregated_labels = np.array(aggregated_labels) if labels is not None else None

        return aggregated_embeddings, aggregated_labels, unique_pids

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the linear probe using Ridge regression/classification.
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.task == 'regression':
            self.probe = Ridge(alpha=self.alpha, random_state=42)
        else:
            # For classification, use RidgeClassifier
            self.probe = RidgeClassifier(alpha=self.alpha, random_state=42)

        self.probe.fit(X_train_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using the fitted probe."""
        if self.probe is None:
            raise ValueError("Probe must be fitted before prediction")

        X_test_scaled = self.scaler.transform(X_test)

        if self.task == 'regression':
            return self.probe.predict(X_test_scaled)
        else:
            # For classification, get decision scores for AUC calculation
            if hasattr(self.probe, 'decision_function'):
                return self.probe.decision_function(X_test_scaled)
            else:
                # Fallback to predict if decision_function not available
                return self.probe.predict(X_test_scaled)


class DownstreamEvaluator:
    """
    Comprehensive downstream evaluation following the paper's protocol.
    Paper Table 2: Evaluates on demographics (age, BMI, sex) and health conditions.
    """

    def __init__(
            self,
            encoder: Optional[nn.Module] = None,
            encoder_checkpoint: Optional[str] = None,
            modality: str = 'ecg',
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.

        Args:
            encoder: Pre-trained encoder model
            encoder_checkpoint: Path to encoder checkpoint
            modality: 'ppg' or 'ecg'
            device: Device to use
        """
        self.encoder = encoder
        self.encoder_checkpoint = encoder_checkpoint
        self.modality = modality
        self.device = device

        # Define evaluation tasks as per paper (Table 2)
        self.demographic_tasks = {
            'age_classification': {
                'type': 'classification',
                'threshold': 50,  # Binary: >50 vs ≤50
                'label_key': 'age_at_admit'
            },
            'age_regression': {
                'type': 'regression',
                'label_key': 'age_at_admit'
            },
            'bmi_classification': {
                'type': 'classification',
                'threshold': 30,  # Binary: >30 vs ≤30 (obese vs non-obese)
                'label_key': 'bmi'
            },
            'bmi_regression': {
                'type': 'regression',
                'label_key': 'bmi'
            },
            'sex_classification': {
                'type': 'classification',
                'label_key': 'sex',
                'mapping': {'male': 1, 'female': 0}
            }
        }

        # Health conditions from MIMIC that map to paper's conditions
        self.health_conditions = [
            'has_chf',  # Congestive Heart Failure
            'has_afib',  # Atrial Fibrillation
            # Add more as available in labels
        ]

    def evaluate_demographics(
            self,
            dataloader: torch.utils.data.DataLoader,
            labels_df: pd.DataFrame,
            task_name: str = 'age_classification',
            n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate on demographic prediction tasks.

        Args:
            dataloader: DataLoader with signals
            labels_df: DataFrame with participant labels
            task_name: Name of the task to evaluate
            n_splits: Number of cross-validation splits

        Returns:
            Dictionary with evaluation metrics
        """
        if task_name not in self.demographic_tasks:
            raise ValueError(f"Unknown task: {task_name}")

        task_info = self.demographic_tasks[task_name]

        # Create probe
        probe = LinearProbe(
            encoder=self.encoder,
            encoder_checkpoint=self.encoder_checkpoint,
            task=task_info['type'],
            modality=self.modality,
            device=self.device
        )

        # Extract embeddings
        print(f"Extracting embeddings for {task_name}...")
        embeddings, _, participant_ids = probe.extract_embeddings(
            dataloader, aggregate_by_participant=True
        )

        # Get labels for participants
        labels = self._prepare_labels(
            participant_ids, labels_df, task_info
        )

        # Filter out participants with missing labels
        valid_mask = ~np.isnan(labels)
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]

        print(f"Evaluating {task_name} with {len(labels)} participants...")

        # Perform cross-validation evaluation
        if task_info['type'] == 'classification':
            return self._evaluate_classification(embeddings, labels, probe, n_splits)
        else:
            return self._evaluate_regression(embeddings, labels, probe, n_splits)

    def _prepare_labels(
            self,
            participant_ids: np.ndarray,
            labels_df: pd.DataFrame,
            task_info: Dict
    ) -> np.ndarray:
        """Prepare labels for the specific task."""
        labels = []

        for pid in participant_ids:
            participant_data = labels_df[labels_df['subject_id'] == pid]

            if participant_data.empty:
                labels.append(np.nan)
                continue

            value = participant_data.iloc[0][task_info['label_key']]

            # Handle missing values
            if pd.isna(value):
                labels.append(np.nan)
                continue

            # Process based on task type
            if task_info['type'] == 'classification':
                if 'threshold' in task_info:
                    # Binary classification with threshold
                    labels.append(1 if value > task_info['threshold'] else 0)
                elif 'mapping' in task_info:
                    # Categorical classification
                    labels.append(task_info['mapping'].get(value, np.nan))
                else:
                    labels.append(value)
            else:
                # Regression
                labels.append(float(value))

        return np.array(labels)

    def _evaluate_classification(
            self,
            X: np.ndarray,
            y: np.ndarray,
            probe: LinearProbe,
            n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate classification task using cross-validation.
        Paper reports AUC and partial AUC at 10% FPR.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        aucs = []
        paucs = []
        accs = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit probe
            probe.fit(X_train, y_train)

            # Predict
            y_scores = probe.predict(X_test)

            # Calculate metrics
            auc = roc_auc_score(y_test, y_scores)

            # Partial AUC at 10% FPR (paper's metric)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            pauc = self._calculate_partial_auc(fpr, tpr, max_fpr=0.1)

            # Accuracy (for reference)
            y_pred = (y_scores > 0).astype(int)
            acc = accuracy_score(y_test, y_pred)

            aucs.append(auc)
            paucs.append(pauc)
            accs.append(acc)

        return {
            'auc': np.mean(aucs),
            'auc_std': np.std(aucs),
            'pauc_0.1': np.mean(paucs),
            'pauc_0.1_std': np.std(paucs),
            'accuracy': np.mean(accs),
            'accuracy_std': np.std(accs)
        }

    def _evaluate_regression(
            self,
            X: np.ndarray,
            y: np.ndarray,
            probe: LinearProbe,
            n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate regression task using cross-validation.
        Paper reports MAE (Mean Absolute Error).
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        maes = []
        r2_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit probe
            probe.fit(X_train, y_train)

            # Predict
            y_pred = probe.predict(X_test)

            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            maes.append(mae)

            # Calculate R² for reference
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)

        return {
            'mae': np.mean(maes),
            'mae_std': np.std(maes),
            'r2': np.mean(r2_scores),
            'r2_std': np.std(r2_scores)
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
        partial_auc = np.trapz(tpr[idx], fpr[idx])

        # Normalize by the maximum possible area
        normalized_pauc = partial_auc / max_fpr

        return normalized_pauc

    def evaluate_all_demographics(
            self,
            dataloader: torch.utils.data.DataLoader,
            labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate all demographic tasks and return results as DataFrame.
        Reproduces paper's Table 2.
        """
        results = []

        for task_name in self.demographic_tasks.keys():
            print(f"\nEvaluating {task_name}...")
            metrics = self.evaluate_demographics(
                dataloader, labels_df, task_name
            )

            # Format results
            if 'classification' in task_name:
                results.append({
                    'Task': task_name,
                    'Modality': self.modality.upper(),
                    'AUC': f"{metrics['auc']:.3f}",
                    'pAUC@0.1': f"{metrics['pauc_0.1']:.3f}",
                    'Accuracy': f"{metrics['accuracy']:.3f}"
                })
            else:
                results.append({
                    'Task': task_name,
                    'Modality': self.modality.upper(),
                    'MAE': f"{metrics['mae']:.2f}",
                    'R²': f"{metrics['r2']:.3f}"
                })

        return pd.DataFrame(results)


# ============= TEST FUNCTIONS =============

def test_evaluation():
    """Test evaluation functionality."""
    print("=" * 50)
    print("Testing Downstream Evaluation")
    print("=" * 50)

    # Create dummy encoder
    encoder = EfficientNet1D(modality='ecg')

    # Create dummy data
    print("\n1. Creating dummy dataset...")
    n_participants = 50
    n_segments_per = 3

    # Create dummy embeddings
    embeddings = np.random.randn(n_participants * n_segments_per, 256)
    participant_ids = np.repeat(np.arange(n_participants), n_segments_per)

    # Create dummy labels
    ages = np.random.randint(20, 80, n_participants)
    bmis = np.random.uniform(18, 40, n_participants)
    sexes = np.random.randint(0, 2, n_participants)

    print(f"   Created {len(embeddings)} embeddings from {n_participants} participants")

    # Test LinearProbe
    print("\n2. Testing LinearProbe:")
    probe = LinearProbe(
        encoder=encoder,
        task='classification',
        modality='ecg'
    )

    # Test aggregation
    agg_embeddings, _, agg_pids = probe._aggregate_by_participant(
        embeddings, None, participant_ids
    )

    print(f"   Aggregated to {len(agg_embeddings)} participant embeddings")
    assert len(agg_embeddings) == n_participants
    print("   ✓ Aggregation test passed!")

    # Test fitting and prediction
    print("\n3. Testing probe fitting:")

    # Binary classification task
    binary_labels = (ages > 50).astype(int)

    # Split data
    n_train = int(0.8 * n_participants)
    X_train = agg_embeddings[:n_train]
    y_train = binary_labels[:n_train]
    X_test = agg_embeddings[n_train:]
    y_test = binary_labels[n_train:]

    # Fit probe
    probe.fit(X_train, y_train)

    # Predict
    y_scores = probe.predict(X_test)

    print(f"   Predictions shape: {y_scores.shape}")
    print("   ✓ Probe fitting test passed!")

    # Test DownstreamEvaluator
    print("\n4. Testing DownstreamEvaluator:")
    evaluator = DownstreamEvaluator(
        encoder=encoder,
        modality='ecg'
    )

    # Test partial AUC calculation
    fpr = np.array([0, 0.05, 0.1, 0.2, 1.0])
    tpr = np.array([0, 0.4, 0.6, 0.8, 1.0])
    pauc = evaluator._calculate_partial_auc(fpr, tpr, max_fpr=0.1)

    print(f"   Partial AUC: {pauc:.3f}")
    assert 0 <= pauc <= 1
    print("   ✓ Partial AUC test passed!")

    # Test cross-validation evaluation
    print("\n5. Testing cross-validation:")

    probe_cv = LinearProbe(encoder=encoder, task='classification', modality='ecg')
    metrics = evaluator._evaluate_classification(
        agg_embeddings, binary_labels, probe_cv, n_splits=3
    )

    print(f"   AUC: {metrics['auc']:.3f} ± {metrics['auc_std']:.3f}")
    print(f"   pAUC@0.1: {metrics['pauc_0.1']:.3f} ± {metrics['pauc_0.1_std']:.3f}")
    print(f"   Accuracy: {metrics['accuracy']:.3f} ± {metrics['accuracy_std']:.3f}")
    print("   ✓ Cross-validation test passed!")

    # Test regression task
    print("\n6. Testing regression evaluation:")

    probe_reg = LinearProbe(encoder=encoder, task='regression', modality='ecg')
    metrics_reg = evaluator._evaluate_regression(
        agg_embeddings, ages, probe_reg, n_splits=3
    )

    print(f"   MAE: {metrics_reg['mae']:.2f} ± {metrics_reg['mae_std']:.2f}")
    print(f"   R²: {metrics_reg['r2']:.3f} ± {metrics_reg['r2_std']:.3f}")
    print("   ✓ Regression evaluation test passed!")

    print("\n" + "=" * 50)
    print("All evaluation tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    # Run tests
    test_evaluation()