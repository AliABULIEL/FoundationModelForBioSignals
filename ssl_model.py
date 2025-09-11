# ssl_model.py - PROPERLY FIXED VERSION
"""
SSL model implementation supporting InfoNCE and SimSiam
Uses centralized configuration management
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pathlib import Path
from copy import deepcopy

from device import get_device_manager
from config_loader import get_config  # Added ConfigLoader


# ssl_model.py - Enhanced but backward-compatible version

# Add these classes to your ssl_model.py

class MixMatchHelper:
    """MixMatch implementation for biosignals."""

    def __init__(self, config):
        self.config = config
        self.K = config.get('semi_supervised.mixmatch.K', 2)  # Number of augmentations
        self.T = config.get('semi_supervised.mixmatch.temperature', 0.5)  # Sharpening temperature
        self.alpha = config.get('semi_supervised.mixmatch.alpha', 0.75)  # MixUp alpha

    def sharpen(self, probs, temperature):
        """Sharpen probability distribution."""
        sharpened = probs ** (1.0 / temperature)
        return sharpened / sharpened.sum(dim=1, keepdim=True)

    def mixup_data(self, x, y, alpha=0.75):
        """MixUp augmentation for 1D biosignals."""
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha)

        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]

        if y.dim() == 1:  # Classification labels
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        else:  # Soft labels or regression
            mixed_y = lam * y + (1 - lam) * y[index]
            return mixed_x, mixed_y, None, lam

    def process_batch(self, labeled_data, unlabeled_data, labels, model, augmentation):
        """Process batch with MixMatch."""
        with torch.no_grad():
            # Generate pseudo-labels with K augmentations
            pseudo_probs = []
            for _ in range(self.K):
                aug_data, _ = augmentation(unlabeled_data, unlabeled_data)
                embeddings = model.encoder(aug_data)

                # Get predictions for each task
                task_probs = {}
                if hasattr(model, 'age_classifier'):
                    task_probs['age'] = F.softmax(model.age_classifier(embeddings), dim=1)
                if hasattr(model, 'sex_classifier'):
                    task_probs['sex'] = F.softmax(model.sex_classifier(embeddings), dim=1)

                pseudo_probs.append(task_probs)

            # Average and sharpen
            avg_probs = {}
            for task in pseudo_probs[0].keys():
                task_prob_list = [p[task] for p in pseudo_probs]
                avg_probs[task] = torch.mean(torch.stack(task_prob_list), dim=0)
                avg_probs[task] = self.sharpen(avg_probs[task], self.T)

        return avg_probs


class TemporalEnsembling:
    """Temporal ensembling for semi-supervised learning."""

    def __init__(self, config, num_samples, num_classes=2):
        self.config = config
        self.alpha = config.get('semi_supervised.temporal.alpha', 0.6)
        self.rampup_epochs = config.get('semi_supervised.temporal.rampup', 40)

        # Initialize ensemble predictions for each task
        self.ensemble_predictions = {
            'age': torch.zeros(num_samples, 2),
            'sex': torch.zeros(num_samples, 2),
            'bmi': torch.zeros(num_samples, 1)
        }
        self.epoch = 0

    def update(self, predictions, indices, task='age'):
        """Update ensemble predictions with EMA."""
        if task not in self.ensemble_predictions:
            return predictions

        device = predictions.device
        ensemble = self.ensemble_predictions[task].to(device)

        # Update with EMA
        ensemble[indices] = (
                self.alpha * ensemble[indices] +
                (1 - self.alpha) * predictions.detach()
        )

        self.ensemble_predictions[task] = ensemble.cpu()
        return ensemble[indices]

    def get_weight(self, epoch):
        """Ramp up weight over epochs."""
        if epoch < self.rampup_epochs:
            return np.exp(-5.0 * (1.0 - epoch / self.rampup_epochs) ** 2)
        return 1.0


class MeanTeacherHelper:
    """Mean Teacher implementation for biosignals."""

    def __init__(self, config):
        self.config = config
        self.ema_decay = config.get('semi_supervised.mean_teacher.ema_decay', 0.999)
        self.consistency_weight = config.get('semi_supervised.mean_teacher.consistency_weight', 1.0)
        self.rampup_epochs = config.get('semi_supervised.mean_teacher.rampup', 40)

    def create_teacher(self, model):
        """Create teacher model from student."""
        teacher = deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def update_teacher(self, student, teacher):
        """Update teacher with EMA of student weights."""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                    teacher.parameters(),
                    student.parameters()
            ):
                teacher_param.data = (
                        self.ema_decay * teacher_param.data +
                        (1 - self.ema_decay) * student_param.data
                )

    def consistency_loss(self, student_pred, teacher_pred):
        """Consistency loss between student and teacher."""
        if isinstance(student_pred, dict):
            losses = []
            for task in student_pred:
                if task in teacher_pred:
                    if 'class' in task or 'sex' in task or 'age' in task:
                        # KL divergence for classification
                        student_log = F.log_softmax(student_pred[task], dim=1)
                        teacher_soft = F.softmax(teacher_pred[task].detach(), dim=1)
                        losses.append(F.kl_div(student_log, teacher_soft, reduction='batchmean'))
                    else:
                        # MSE for regression
                        losses.append(F.mse_loss(student_pred[task], teacher_pred[task].detach()))

            return torch.mean(torch.stack(losses)) if losses else 0
        else:
            return F.mse_loss(student_pred, teacher_pred.detach())


# Enhanced SemiSupervisedLoss with all methods
class SemiSupervisedLoss(nn.Module):
    def __init__(self, alpha=0.01, age_threshold=50):
        super().__init__()
        self.config = get_config()
        self.alpha = self.config.config.get('semi_supervised', {}).get('supervised_weight', alpha)

        self.age_threshold = age_threshold
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()

        # Method selection
        self.method = self.config.config.get('semi_supervised', {}).get('method', 'standard')

        # Initialize helpers based on method
        if self.method == 'mixmatch':
            self.mixmatch_helper = MixMatchHelper(self.config.config)
        elif self.method == 'temporal':
            # Will be initialized with dataset size
            self.temporal_helper = None
        elif self.method == 'mean_teacher':
            self.mean_teacher_helper = MeanTeacherHelper(self.config.config)

        # Standard options (always available)
        self.use_pseudo = self.config.config.get('semi_supervised', {}).get('use_pseudo', False)
        self.pseudo_threshold = self.config.config.get('semi_supervised', {}).get('pseudo_threshold', 0.95)
        self.use_consistency = self.config.config.get('semi_supervised', {}).get('use_consistency', False)
        self.consistency_weight = 0.0
        self.confidence_weighting = self.config.config.get('semi_supervised', {}).get('confidence_weighting', False)

    def init_temporal_ensemble(self, num_samples):
        """Initialize temporal ensemble with dataset size."""
        if self.method == 'temporal':
            self.temporal_helper = TemporalEnsembling(
                self.config.config,
                num_samples
            )

    def forward(self, ssl_loss, predictions, labels,
                predictions_aug=None, teacher_predictions=None,
                sample_indices=None, epoch=0):
        """
        Enhanced forward with all semi-supervised methods.

        Args:
            ssl_loss: Base SSL loss
            predictions: Student predictions
            labels: Ground truth labels (can be None)
            predictions_aug: Second augmentation predictions (for consistency)
            teacher_predictions: Teacher model predictions (for mean teacher)
            sample_indices: Sample indices (for temporal ensemble)
            epoch: Current epoch (for ramp-up schedules)
        """

        total_supervised_loss = 0
        method_loss = 0

        # Standard supervised loss (if labels available)
        supervised_losses = []
        weights = []

        if labels is not None:
            # Age classification
            if 'age' in labels and 'age_class' in predictions:
                age_values = labels['age']
                valid_mask = age_values >= 0
                if valid_mask.any():
                    age_binary = (age_values > self.age_threshold).long()

                    if self.method == 'mixmatch' and hasattr(self, 'mixmatch_helper'):
                        # Use MixUp for labeled data
                        mixed_input = predictions['age_class'][valid_mask]
                        mixed_labels = age_binary[valid_mask]
                        age_loss = self.ce_loss(mixed_input, mixed_labels)
                    else:
                        age_loss = self.ce_loss(
                            predictions['age_class'][valid_mask],
                            age_binary[valid_mask]
                        )

                    supervised_losses.append(age_loss)

                    if self.confidence_weighting:
                        with torch.no_grad():
                            probs = F.softmax(predictions['age_class'][valid_mask], dim=1)
                            confidence = probs.max(dim=1)[0].mean()
                            weights.append(confidence)
                    else:
                        weights.append(1.0)

            # BMI regression
            if 'bmi' in labels and 'bmi' in predictions:
                valid_mask = labels['bmi'] > 0
                if valid_mask.any():
                    bmi_normalized = (labels['bmi'][valid_mask] - 25.0) / 5.0
                    bmi_normalized = torch.clamp(bmi_normalized, -2, 2)

                    bmi_pred = predictions['bmi'][valid_mask].squeeze()
                    bmi_loss = self.mse_loss(bmi_pred, bmi_normalized)
                    supervised_losses.append(bmi_loss * 0.5)
                    weights.append(1.0)

            # Sex classification
            if 'sex' in labels and 'sex' in predictions:
                valid_mask = labels['sex'] >= 0
                if valid_mask.any():
                    sex_loss = self.ce_loss(
                        predictions['sex'][valid_mask],
                        labels['sex'][valid_mask].long()
                    )
                    supervised_losses.append(sex_loss)

                    if self.confidence_weighting:
                        with torch.no_grad():
                            probs = F.softmax(predictions['sex'][valid_mask], dim=1)
                            confidence = probs.max(dim=1)[0].mean()
                            weights.append(confidence)
                    else:
                        weights.append(1.0)

        # Calculate supervised loss
        if supervised_losses:
            if self.confidence_weighting and len(weights) == len(supervised_losses):
                weights = torch.tensor(weights, device=ssl_loss.device)
                weights = weights / weights.sum()
                total_supervised_loss = sum(w * l for w, l in zip(weights, supervised_losses))
            else:
                total_supervised_loss = torch.mean(torch.stack(supervised_losses))

            total_supervised_loss = torch.clamp(total_supervised_loss, max=10.0)

        # Method-specific losses
        if self.method == 'temporal' and self.temporal_helper and sample_indices is not None:
            # Temporal ensemble consistency
            ensemble_losses = []
            weight = self.temporal_helper.get_weight(epoch)

            for task in ['age_class', 'sex']:
                if task in predictions:
                    probs = F.softmax(predictions[task], dim=1)
                    ensemble_pred = self.temporal_helper.update(probs, sample_indices, task.replace('_class', ''))
                    ensemble_losses.append(F.mse_loss(probs, ensemble_pred) * weight)

            if ensemble_losses:
                method_loss = torch.mean(torch.stack(ensemble_losses))

        elif self.method == 'mean_teacher' and teacher_predictions is not None:
            # Mean teacher consistency
            consistency = self.mean_teacher_helper.consistency_loss(predictions, teacher_predictions)
            weight = min(epoch / self.mean_teacher_helper.rampup_epochs, 1.0)
            method_loss = consistency * weight * self.mean_teacher_helper.consistency_weight

        elif self.use_consistency and predictions_aug is not None:
            # Standard consistency regularization
            cons_losses = []

            for task in ['age_class', 'sex']:
                if task in predictions and task in predictions_aug:
                    p1 = F.log_softmax(predictions[task], dim=1)
                    p2 = F.softmax(predictions_aug[task].detach(), dim=1)
                    cons_losses.append(F.kl_div(p1, p2, reduction='batchmean'))

            if 'bmi' in predictions and 'bmi' in predictions_aug:
                cons_losses.append(
                    F.mse_loss(predictions['bmi'], predictions_aug['bmi'].detach()) * 0.5
                )

            if cons_losses:
                self.update_consistency_weight(epoch)
                method_loss = torch.mean(torch.stack(cons_losses)) * self.consistency_weight

        # Pseudo-labeling (for unlabeled data)
        pseudo_loss = 0
        if self.use_pseudo and not supervised_losses:
            pseudo_losses = []

            for task in ['age_class', 'sex']:
                if task in predictions:
                    with torch.no_grad():
                        probs = F.softmax(predictions[task], dim=1)
                        max_probs, pseudo_labels = probs.max(dim=1)
                        confident_mask = max_probs > self.pseudo_threshold

                    if confident_mask.any():
                        loss = self.ce_loss(
                            predictions[task][confident_mask],
                            pseudo_labels[confident_mask]
                        )
                        pseudo_losses.append(loss * 0.1)

            if pseudo_losses:
                pseudo_loss = torch.mean(torch.stack(pseudo_losses))

        # Combine all losses
        total_loss = ssl_loss

        if supervised_losses:
            total_loss += self.alpha * total_supervised_loss

        if method_loss > 0:
            total_loss += method_loss

        if pseudo_loss > 0:
            total_loss += pseudo_loss * 0.01

        return total_loss, {
            'supervised': total_supervised_loss.item() if supervised_losses else 0,
            'method': method_loss.item() if isinstance(method_loss, torch.Tensor) else 0,
            'pseudo': pseudo_loss.item() if isinstance(pseudo_loss, torch.Tensor) else 0
        }

    def update_consistency_weight(self, epoch, max_epochs=100):
        """Update consistency weight with ramp-up."""
        if self.use_consistency:
            rampup = self.config.config.get('semi_supervised', {}).get('consistency_rampup', 20)
            if epoch < rampup:
                self.consistency_weight = (epoch / rampup) * 0.1
            else:
                self.consistency_weight = 0.1
class RegularizedInfoNCE(nn.Module):
    """InfoNCE loss with KoLeo regularization."""

    def __init__(
            self,
            temperature: Optional[float] = None,
            lambda_koleo: Optional[float] = None,
            normalize: bool = True,
            config_path: str = 'configs/config.yaml'
    ):
        super().__init__()

        # Load configuration
        config = get_config()
        ssl_config = config.get_ssl_config('infonce')

        # Use config values with fallbacks
        self.temperature = temperature if temperature is not None else ssl_config.get('temperature', 0.07)
        self.lambda_koleo = lambda_koleo if lambda_koleo is not None else ssl_config.get('lambda_koleo', 0.05)
        self.normalize = normalize

        # Get variance loss weight from config
        self.lambda_variance = config.get('ssl.lambda_variance', 0.1)

    def infonce_loss(self, queries, keys):
        batch_size = queries.shape[0]

        if self.normalize:
            queries = F.normalize(queries, dim=1)
            keys = F.normalize(keys, dim=1)

        sim_matrix = torch.matmul(queries, keys.T) / self.temperature
        labels = torch.arange(batch_size, device=queries.device)

        return F.cross_entropy(sim_matrix, labels)

    def koleo_loss(self, embeddings):
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)

        distances = torch.cdist(embeddings, embeddings, p=2)
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        distances = distances.masked_fill(mask, float('inf'))
        min_distances = distances.min(dim=1)[0]

        # Get min distance clamp from config
        config = get_config()
        min_dist_clamp = config.get('ssl.koleo_min_distance_clamp', 1e-3)
        min_distances = torch.clamp(min_distances, min=min_dist_clamp)
        embedding_std = embeddings.std(dim=0).mean()
        if embedding_std < 0.01:
            print("WARNING: Possible collapse detected!")

        return -torch.log(min_distances).mean()

    def forward(self, z1, z2, z1_momentum=None, z2_momentum=None):
        keys1 = z1_momentum if z1_momentum is not None else z1
        keys2 = z2_momentum if z2_momentum is not None else z2

        loss_12 = self.infonce_loss(z1, keys2)
        loss_21 = self.infonce_loss(z2, keys1)
        loss_contrastive = (loss_12 + loss_21) / 2

        koleo_1 = self.koleo_loss(z1)
        koleo_2 = self.koleo_loss(z2)
        loss_koleo = (koleo_1 + koleo_2) / 2

        var_loss = 0
        z_concat = torch.cat([z1, z2], dim=0)

        # Get variance parameters from config
        config = get_config()
        variance_epsilon = config.get('ssl.variance_epsilon', 1e-4)
        variance_threshold = config.get('ssl.variance_threshold', 1.0)

        std = torch.sqrt(z_concat.var(dim=0) + variance_epsilon)
        var_loss = torch.mean(F.relu(variance_threshold - std))

        total_loss = loss_contrastive + self.lambda_koleo * loss_koleo + self.lambda_variance * var_loss

        metrics = {
            'loss': total_loss.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_koleo': loss_koleo.item(),
            'loss_12': loss_12.item(),
            'loss_21': loss_21.item(),
            'loss_variance': var_loss.item()
        }

        return total_loss, metrics


class SimSiamLoss(nn.Module):
    """SimSiam loss - CORRECTED implementation."""

    def __init__(self, config_path: str = 'configs/config.yaml'):
        super().__init__()
        self.config = get_config()

    def forward(self, p1, p2, z1, z2):
        """
        SimSiam loss computation.
        p1, p2: outputs from predictor
        z1, z2: outputs from projector (with stop-gradient)
        """
        # Normalize the projector outputs (z), not predictor outputs (p)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # Negative cosine similarity (the paper uses negative cosine)
        loss = -(F.cosine_similarity(p1, z2.detach()).mean() +
                 F.cosine_similarity(p2, z1.detach()).mean()) * 0.5

        metrics = {
            'loss': loss.item(),
            'loss_contrastive': loss.item(),
            'loss_koleo': 0.0,
            'loss_12': -F.cosine_similarity(p1, z2.detach()).mean().item(),
            'loss_21': -F.cosine_similarity(p2, z1.detach()).mean().item(),
            'loss_variance': 0.0
        }

        return loss, metrics


class SSLModel(nn.Module):
    """Unified SSL model supporting both InfoNCE and SimSiam."""

    def __init__(
            self,
            encoder: nn.Module,
            projection_head: nn.Module,
            config_path: str = 'configs/config.yaml',
            ssl_method: str = 'infonce',
            use_supervised: bool = False
    ):
        super().__init__()
        self.ssl_method = ssl_method
        self.use_supervised = use_supervised

        # Load configuration
        self.config = get_config()
        model_config = self.config.get_model_config()

        self.encoder = encoder


        # Check actual input size using config
        with torch.no_grad():
            # Use segment size from config
            if self.config.get('downsample.segment_length_sec'):
                segment_length_sec = self.config.get('downsample.segment_length_sec')
                # Assume PPG for default
                target_fs = self.config.get('dataset.ppg.target_fs', 64)
                dummy_size = int(segment_length_sec * target_fs)
            else:
                # Use default segment lengths from config
                segment_length = self.config.get('dataset.ppg.segment_length', 30)
                target_fs = self.config.get('dataset.ppg.target_fs', 64)
                dummy_size = segment_length * target_fs

            device = next(encoder.parameters()).device
            dummy = torch.randn(1, 1, dummy_size).to(device)
            embedding_dim = encoder(dummy).shape[1]
        ssl_config = self.config.get_ssl_config('infonce')
        if self.use_supervised:
            # Initialize heads with better architecture
            hidden_dim = 128  # Larger hidden layer

            self.age_classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            )

            self.bmi_regressor = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output in [0,1] for normalized BMI
            )

            self.sex_classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            )
        if ssl_method == 'simsiam':
            simsiam_config = self.config.get_ssl_config('simsiam')
            proj_dim = simsiam_config.get('projection_dim', 2048)
            pred_dim = simsiam_config.get('prediction_dim', 512)

            # Correct 3-layer projector with BN
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim, affine=False)  # No learnable affine parameters
            )

            # Correct 2-layer predictor
            self.predictor = nn.Sequential(
                nn.Linear(proj_dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pred_dim, proj_dim)  # No BN or activation at the end
            )

            self.criterion = SimSiamLoss(config_path=config_path)
            self.momentum_encoder = deepcopy(encoder)
            self.momentum_projection = deepcopy(projection_head)
            self.momentum_rate = ssl_config.get('momentum_rate', 0.99)

        else:  # infonce
            ssl_config = self.config.get_ssl_config('infonce')
            self.projection_head = projection_head

            self.momentum_encoder = deepcopy(encoder)
            self.momentum_projection = deepcopy(projection_head)

            for param in self.momentum_encoder.parameters():
                param.requires_grad = False
            for param in self.momentum_projection.parameters():
                param.requires_grad = False

            self.criterion = RegularizedInfoNCE(
                temperature=ssl_config.get('temperature', 0.07),
                lambda_koleo=ssl_config.get('lambda_koleo', 0.05),
                config_path=config_path
            )

            self.momentum_rate = ssl_config.get('momentum_rate', 0.99)

        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        self.to(self.device)

        print(f"\nSSL Model initialized:")
        print(f"  Method: {ssl_method}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Device: {self.device}")
        if ssl_method == 'infonce':
            print(f"  Temperature: {self.criterion.temperature}")
            print(f"  Lambda KoLeo: {self.criterion.lambda_koleo}")
            print(f"  Lambda Variance: {self.criterion.lambda_variance}")
            print(f"  Momentum rate: {self.momentum_rate}")
        else:
            print(f"  Projection dim: {proj_dim}")
            print(f"  Prediction dim: {pred_dim}")

    @torch.no_grad()
    def update_momentum_networks(self):
        if self.ssl_method != 'infonce':
            return

        for param_m, param in zip(
                self.momentum_encoder.parameters(),
                self.encoder.parameters()
        ):
            param_m.data = (
                    self.momentum_rate * param_m.data +
                    (1 - self.momentum_rate) * param.data
            )

        for param_m, param in zip(
                self.momentum_projection.parameters(),
                self.projection_head.parameters()
        ):
            param_m.data = (
                    self.momentum_rate * param_m.data +
                    (1 - self.momentum_rate) * param.data
            )

    def forward(self, x1, x2, return_predictions=False, x1_aug2=None, x2_aug2=None):
        """
        Forward pass with optional consistency regularization.

        Args:
            x1, x2: Primary augmented views for SSL
            return_predictions: Whether to return supervised predictions
            x1_aug2, x2_aug2: Optional second augmentations for consistency
        """
        # Get embeddings for primary augmentations
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Calculate SSL loss (existing code)
        if self.ssl_method == 'simsiam':
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            loss, metrics = self.criterion(p1, p2, z1, z2)
        else:  # infonce
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

            with torch.no_grad():
                h1_m = self.momentum_encoder(x1)
                h2_m = self.momentum_encoder(x2)
                z1_m = self.momentum_projection(h1_m)
                z2_m = self.momentum_projection(h2_m)

            loss, metrics = self.criterion(z1, z2, z1_m, z2_m)

        # Add supervised predictions if requested
        if return_predictions and self.use_supervised:
            predictions = {}
            predictions_aug = None

            # Get predictions for first augmentation
            if hasattr(self, 'age_classifier'):
                predictions['age_class'] = self.age_classifier(h1)
            if hasattr(self, 'bmi_regressor'):
                predictions['bmi'] = self.bmi_regressor(h1)
            if hasattr(self, 'sex_classifier'):
                predictions['sex'] = self.sex_classifier(h1)

            # Get predictions for second augmentation (for consistency)
            if x1_aug2 is not None:
                h1_aug2 = self.encoder(x1_aug2)
                predictions_aug = {}

                if hasattr(self, 'age_classifier'):
                    predictions_aug['age_class'] = self.age_classifier(h1_aug2)
                if hasattr(self, 'bmi_regressor'):
                    predictions_aug['bmi'] = self.bmi_regressor(h1_aug2)
                if hasattr(self, 'sex_classifier'):
                    predictions_aug['sex'] = self.sex_classifier(h1_aug2)

            # Return all predictions
            return loss, metrics, predictions, predictions_aug

        # Standard SSL return (backward compatible)
        return loss, metrics

    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def save_checkpoint(self, path, optimizer=None, epoch=0, best_loss=float('inf')):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_state_dict': self.projection_head.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
            'ssl_method': self.ssl_method
        }

        if self.ssl_method == 'simsiam':
            checkpoint['predictor_state_dict'] = self.predictor.state_dict()
        else:
            checkpoint['momentum_encoder_state_dict'] = self.momentum_encoder.state_dict()
            checkpoint['momentum_projection_state_dict'] = self.momentum_projection.state_dict()

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if self.use_supervised:
            checkpoint['age_classifier_state_dict'] = self.age_classifier.state_dict()
            checkpoint['bmi_regressor_state_dict'] = self.bmi_regressor.state_dict()
            checkpoint['sex_classifier_state_dict'] = self.sex_classifier.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_state_dict'])

        if self.ssl_method == 'simsiam' and 'predictor_state_dict' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        elif self.ssl_method == 'infonce':
            if 'momentum_encoder_state_dict' in checkpoint:
                self.momentum_encoder.load_state_dict(checkpoint['momentum_encoder_state_dict'])
            if 'momentum_projection_state_dict' in checkpoint:
                self.momentum_projection.load_state_dict(checkpoint['momentum_projection_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('use_supervised', False) and self.use_supervised:
            if 'age_classifier_state_dict' in checkpoint:
                self.age_classifier.load_state_dict(checkpoint['age_classifier_state_dict'])
            if 'bmi_regressor_state_dict' in checkpoint:
                self.bmi_regressor.load_state_dict(checkpoint['bmi_regressor_state_dict'])
            if 'sex_classifier_state_dict' in checkpoint:
                self.sex_classifier.load_state_dict(checkpoint['sex_classifier_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))


def create_ssl_model(
        encoder: nn.Module,
        projection_head: nn.Module,
        config_path: str = 'configs/config.yaml',
        ssl_method: str = 'infonce',
        use_supervised: bool = False
) -> SSLModel:
    return SSLModel(encoder, projection_head, config_path, ssl_method,use_supervised)


def test_ssl():
    """Test both InfoNCE and SimSiam."""
    print("=" * 50)
    print("Testing SSL Models")
    print("=" * 50)

    device_manager = get_device_manager()
    device = device_manager.device

    from model import EfficientNet1D, ProjectionHead

    # Load config
    config = get_config()

    # Get segment parameters from config
    if config.get('downsample.segment_length_sec'):
        segment_length_sec = config.get('downsample.segment_length_sec')
        target_fs = config.get('dataset.ppg.target_fs', 64)
        test_length = int(segment_length_sec * target_fs)
    else:
        segment_length = config.get('dataset.ppg.segment_length', 30)
        target_fs = config.get('dataset.ppg.target_fs', 64)
        test_length = segment_length * target_fs

    # Test InfoNCE
    print("\n1. Testing InfoNCE Model:")

    embedding_dim = config.get('model.embedding_dim', 256)
    projection_dim = config.get('model.projection_dim', 128)

    encoder = EfficientNet1D(in_channels=1, embedding_dim=embedding_dim)
    projection_head = ProjectionHead(input_dim=embedding_dim, output_dim=projection_dim)

    infonce_model = create_ssl_model(encoder, projection_head, ssl_method='infonce')

    x1 = torch.randn(8, 1, test_length).to(device)
    x2 = torch.randn(8, 1, test_length).to(device)

    loss, metrics = infonce_model(x1, x2)
    print(f"   InfoNCE loss: {loss:.4f}")
    print(f"   Contrastive: {metrics['loss_contrastive']:.4f}")
    print(f"   KoLeo: {metrics['loss_koleo']:.4f}")
    print(f"   Variance: {metrics.get('loss_variance', 0):.4f}")
    assert loss > 0, "InfoNCE loss must be positive"
    print("   ✓ InfoNCE test passed!")

    # Test SimSiam
    print("\n2. Testing SimSiam Model:")
    encoder2 = EfficientNet1D(in_channels=1, embedding_dim=embedding_dim)
    projection_head2 = ProjectionHead(input_dim=embedding_dim, output_dim=projection_dim)

    simsiam_model = create_ssl_model(encoder2, projection_head2, ssl_method='simsiam')

    loss2, metrics2 = simsiam_model(x1, x2)
    print(f"   SimSiam loss: {loss2:.4f}")
    assert loss2 < 0, "SimSiam loss must be positive (fixed)"
    print("   ✓ SimSiam test passed!")

    # Test with different batch sizes
    print("\n3. Testing different batch sizes:")
    batch_sizes = config.get('test.batch_sizes', [1, 4, 8, 16])
    for batch_size in batch_sizes:
        test_x1 = torch.randn(batch_size, 1, test_length).to(device)
        test_x2 = torch.randn(batch_size, 1, test_length).to(device)

        loss, _ = infonce_model(test_x1, test_x2)
        assert loss > 0
        print(f"   Batch size {batch_size}: ✓")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    test_ssl()