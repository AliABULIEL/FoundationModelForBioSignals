import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import warnings

warnings.filterwarnings('ignore')

from .model import EfficientNet1D, BiosignalFoundationModel


class LinearProbe(nn.Module):
    """
    Linear probe for downstream evaluation following the paper's approach.
    Paper uses Ridge regression for both classification and regression tasks.
    """

    def __init__(
            self,
            encoder_checkpoint: str,
            task: str = 'classification',
            num_classes: int = 2,
            freeze_encoder: bool = True,
            modality: str = 'ecg',
            use_sklearn: bool = True,  # Paper uses sklearn Ridge
            alpha: float = 1.0  # Ridge regularization
    ):
        super().__init__()

        self.task = task.lower()
        self.num_classes = num_classes
        self.modality = modality
        self.use_sklearn = use_sklearn
        self.alpha = alpha

        # Load pre-trained encoder
        self.encoder = self._load_encoder(encoder_checkpoint, modality)

        # Freeze encoder if specified (paper freezes for linear probing)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Create probe head
        if not use_sklearn:
            # PyTorch linear layer (for end-to-end fine-tuning if needed)
            out_dim = 1 if task == 'regression' else num_classes
            self.probe = nn.Linear(256, out_dim)
        else:
            # Use sklearn Ridge (paper's approach)
            self.probe = None  # Will be created during fit

    def _load_encoder(self, checkpoint_path: str, modality: str) -> nn.Module:
        """Load pre-trained encoder from checkpoint."""
        # Check if checkpoint is full model or just encoder
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'encoder_state_dict' in checkpoint:
            # Full training checkpoint
            encoder = EfficientNet1D(modality=modality)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Alternative format
            model = BiosignalFoundationModel(modality=modality)
            model.load_state_dict(checkpoint['model_state_dict'])
            encoder = model.encoder
        else:
            # Direct encoder weights
            encoder = EfficientNet1D(modality=modality)
            encoder.load_state_dict(checkpoint)

        return encoder

    @torch.no_grad()
    def extract_embeddings(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for all samples in dataloader.
        Paper aggregates embeddings at participant level.
        """
        self.encoder.eval()

        embeddings = []
        labels = []
        participant_ids = []

        for batch in dataloader:
            if len(batch) == 3:
                # With participant IDs
                x, y, pids = batch
                participant_ids.extend(pids)
            else:
                # Without participant IDs
                x, y = batch
                participant_ids = None

            # Move to device
            x = x.to(next(self.encoder.parameters()).device)

            # Extract embeddings
            emb = self.encoder(x)
            embeddings.append(emb.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else y)

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Aggregate by participant if IDs available (paper's approach)
        if participant_ids is not None:
            embeddings, labels = self._aggregate_by_participant(
                embeddings, labels, participant_ids
            )

        return embeddings, labels

    def _aggregate_by_participant(
            self,
            embeddings: np.ndarray,
            labels: np.ndarray,
            participant_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate embeddings by participant (mean pooling).
        Paper: "mean-aggregate all embeddings associated to each participant"
        """
        df = pd.DataFrame({
            'pid': participant_ids,
            'label': labels
        })

        # Add embedding columns
        for i in range(embeddings.shape[1]):
            df[f'emb_{i}'] = embeddings[:, i]

        # Group by participant and aggregate
        emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]

        # Mean aggregate embeddings, take first label (should be same per participant)
        agg_df = df.groupby('pid').agg({
            **{col: 'mean' for col in emb_cols},
            'label': 'first'
        }).reset_index()

        # Extract aggregated embeddings and labels
        agg_embeddings = agg_df[emb_cols].values
        agg_labels = agg_df['label'].values

        return agg_embeddings, agg_labels

    def fit_sklearn_probe(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray
    ):
        """
        Fit sklearn Ridge regression/classifier (paper's approach).
        """
        if self.task == 'regression':
            self.probe = Ridge(alpha=self.alpha)
        else:
            # For classification, paper uses Ridge regression with binarized targets
            self.probe = RidgeClassifier(alpha=self.alpha)

        # Standardize features (common practice)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit the probe
        self.probe.fit(X_train_scaled, y_train)

    def predict_sklearn(self, X: np.ndarray) -> np.ndarray:
        """Predict using sklearn probe."""
        X_scaled = self.scaler.transform(X)

        if self.task == 'regression':
            return self.probe.predict(X_scaled)
        else:
            # For classification, get decision scores
            if hasattr(self.probe, 'decision_function'):
                return self.probe.decision_function(X_scaled)
            else:
                return self.probe.predict_proba(X_scaled)[:, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch-based evaluation.
        """
        with torch.no_grad():
            embeddings = self.encoder(x)

        if self.use_sklearn:
            # Convert to numpy and use sklearn
            emb_np = embeddings.cpu().numpy()
            preds = self.predict_sklearn(emb_np)
            return torch.from_numpy(preds).to(x.device)
        else:
            # Use PyTorch linear layer
            return self.probe(embeddings)


class DownstreamEvaluator:
    """
    Comprehensive downstream evaluation following the paper's protocol.
    Evaluates on demographics (age, BMI, sex) and health conditions.
    """

    def __init__(
            self,
            encoder_checkpoint: str,
            modality: str = 'ecg'
    ):
        self.encoder_checkpoint = encoder_checkpoint
        self.modality = modality

        # Paper's evaluation tasks
        self.demographic_tasks = {
            'age_classification': {'type': 'classification', 'threshold': 50},
            'age_regression': {'type': 'regression'},
            'bmi_classification': {'type': 'classification', 'threshold': 30},
            'bmi_regression': {'type': 'regression'},
            'sex_classification': {'type': 'classification'}
        }

    def evaluate_demographics(
            self,
            dataloader,
            task_name: str = 'age_classification'
    ) -> Dict[str, float]:
        """
        Evaluate on demographic prediction tasks.

        Returns:
            Dictionary with evaluation metrics
        """
        task_info = self.demographic_tasks.get(task_name)
        if not task_info:
            raise ValueError(f"Unknown task: {task_name}")

        # Create probe
        probe = LinearProbe(
            encoder_checkpoint=self.encoder_checkpoint,
            task=task_info['type'],
            modality=self.modality,
            use_sklearn=True
        )

        # Extract embeddings
        X, y = probe.extract_embeddings(dataloader)

        # Split data for evaluation
        if task_info['type'] == 'classification':
            return self._evaluate_classification(X, y, probe)
        else:
            return self._evaluate_regression(X, y, probe)

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

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit probe
            probe.fit_sklearn_probe(X_train, y_train)

            # Predict
            y_scores = probe.predict_sklearn(X_test)

            # Calculate metrics
            auc = roc_auc_score(y_test, y_scores)

            # Partial AUC at 10% FPR (paper's metric)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            pauc = self._calculate_partial_auc(fpr, tpr, max_fpr=0.1)

            aucs.append(auc)
            paucs.append(pauc)

        return {
            'auc': np.mean(aucs),
            'auc_std': np.std(aucs),
            'pauc_0.1': np.mean(paucs),
            'pauc_std': np.std(paucs)
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

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit probe
            probe.fit_sklearn_probe(X_train, y_train)

            # Predict
            y_pred = probe.predict_sklearn(X_test)

            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            maes.append(mae)

        return {
            'mae': np.mean(maes),
            'mae_std': np.std(maes)
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

    def evaluate_health_conditions(
            self,
            dataloader,
            condition_labels: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on health condition prediction tasks.
        Paper evaluates on 29+ medical conditions.

        Args:
            dataloader: DataLoader for the dataset
            condition_labels: Dict mapping condition names to binary labels

        Returns:
            Dict mapping condition names to evaluation metrics
        """
        # Extract embeddings once
        probe = LinearProbe(
            encoder_checkpoint=self.encoder_checkpoint,
            task='classification',
            modality=self.modality,
            use_sklearn=True
        )

        X, _ = probe.extract_embeddings(dataloader)

        results = {}

        for condition_name, labels in condition_labels.items():
            print(f"Evaluating {condition_name}...")

            # Create new probe for this condition
            condition_probe = LinearProbe(
                encoder_checkpoint=self.encoder_checkpoint,
                task='classification',
                modality=self.modality,
                use_sklearn=True
            )

            # Evaluate
            metrics = self._evaluate_classification(X, labels, condition_probe)
            results[condition_name] = metrics

        return results


def create_evaluation_dataloaders(
        data_path: str,
        csv_path: str,
        batch_size: int = 32,
        modality: str = 'ecg'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/test dataloaders for downstream evaluation.

    Args:
        data_path: Path to signal data
        csv_path: Path to CSV with labels
        batch_size: Batch size
        modality: 'ecg' or 'ppg'

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader, Dataset

    class LabeledDataset(Dataset):
        def __init__(self, filepaths, labels, transform=None):
            self.filepaths = filepaths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            # Load signal
            signal = np.load(self.filepaths[idx]).astype(np.float32)
            if signal.ndim == 1:
                signal = signal[np.newaxis, :]

            # Apply transform if provided
            if self.transform:
                signal = self.transform(signal)

            label = self.labels[idx]

            return torch.from_numpy(signal).float(), torch.tensor(label).float()

    # Load CSV
    df = pd.read_csv(csv_path)
    filepaths = [Path(p) for p in df['filepath']]
    labels = df['label'].values

    # Split data (80/20)
    n_train = int(0.8 * len(filepaths))

    train_dataset = LabeledDataset(filepaths[:n_train], labels[:n_train])
    test_dataset = LabeledDataset(filepaths[n_train:], labels[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader