"""
TabPFN v2 Foundation Model Integration
Provides unified interface for TabPFN Classifier and Regressor
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class TabPFNFoundation:
    """
    Foundation model wrapper for TabPFN v2
    Supports both classification (IOH) and regression (BP) tasks
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 device: str = 'cuda',
                 N_ensemble_configurations: int = 4,
                 max_features: int = 500,
                 use_auto_optimizer: bool = False,
                 **kwargs):
        """
        Initialize TabPFN v2 model
        
        Args:
            task_type: 'classification' for IOH, 'regression' for BP
            device: 'cuda' or 'cpu'
            N_ensemble_configurations: Number of ensemble members (≤8 recommended)
            max_features: Maximum features (TabPFN v2 supports ≤500)
            use_auto_optimizer: Use AutoTabPFNClassifier for hyperparameter optimization
            **kwargs: Additional TabPFN parameters
        """
        self.task_type = task_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.N_ensemble_configurations = min(N_ensemble_configurations, 8)
        self.max_features = min(max_features, 500)
        self.use_auto_optimizer = use_auto_optimizer
        
        # Import TabPFN v2
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            self.tabpfn_available = True
            
            # Optional: AutoTabPFN for optimization
            if use_auto_optimizer:
                try:
                    from tabpfn_extensions import AutoTabPFNClassifier
                    self.auto_available = True
                except ImportError:
                    self.auto_available = False
                    if use_auto_optimizer:
                        print("Warning: AutoTabPFNClassifier not available. Install tabpfn_extensions.")
                        self.use_auto_optimizer = False
        except ImportError:
            raise ImportError("TabPFN v2 not installed. Run: pip install tabpfn")
        
        # Initialize model based on task
        if task_type == 'classification':
            if self.use_auto_optimizer and self.auto_available:
                from tabpfn_extensions import AutoTabPFNClassifier
                self.model = AutoTabPFNClassifier(
                    device=self.device,
                    N_ensemble_configurations=self.N_ensemble_configurations,
                    **kwargs
                )
            else:
                self.model = TabPFNClassifier(
                    device=self.device,
                    N_ensemble_configurations=self.N_ensemble_configurations,
                    **kwargs
                )
        elif task_type == 'regression':
            self.model = TabPFNRegressor(
                device=self.device,
                N_ensemble_configurations=self.N_ensemble_configurations,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Context storage
        self.context_X = None
        self.context_y = None
        self.is_fitted = False
        
        # Feature selection if > max_features
        self.selected_features = None
        
    def fit_context(self, X_context: np.ndarray, y_context: np.ndarray,
                    feature_names: Optional[List[str]] = None) -> 'TabPFNFoundation':
        """
        Fit the model with in-context learning data
        
        Args:
            X_context: Context features of shape (n_context, n_features)
            y_context: Context labels of shape (n_context,)
            feature_names: Optional feature names for interpretability
            
        Returns:
            Self for chaining
        """
        # Validate input
        if X_context.shape[0] != y_context.shape[0]:
            raise ValueError(f"Shape mismatch: X_context {X_context.shape} vs y_context {y_context.shape}")
        
        # Handle NaN features (TabPFN v2 supports NaN)
        n_samples, n_features = X_context.shape
        
        # Check feature limit
        if n_features > self.max_features:
            print(f"Warning: {n_features} features exceeds limit of {self.max_features}")
            print(f"Selecting top {self.max_features} features based on variance...")
            
            # Select features with highest variance (ignoring NaN)
            variances = np.nanvar(X_context, axis=0)
            self.selected_features = np.argsort(variances)[-self.max_features:]
            X_context = X_context[:, self.selected_features]
            
            if feature_names:
                feature_names = [feature_names[i] for i in self.selected_features]
        
        # Check sample limit (~10,000 for TabPFN v2)
        max_samples = 10000
        if n_samples > max_samples:
            print(f"Warning: {n_samples} samples exceeds recommended {max_samples}")
            print(f"Subsampling to {max_samples} samples...")
            
            # Stratified sampling for classification
            if self.task_type == 'classification':
                from sklearn.model_selection import train_test_split
                _, X_context, _, y_context = train_test_split(
                    X_context, y_context, 
                    test_size=max_samples/n_samples,
                    stratify=y_context,
                    random_state=42
                )
            else:
                # Random sampling for regression
                indices = np.random.choice(n_samples, max_samples, replace=False)
                X_context = X_context[indices]
                y_context = y_context[indices]
        
        # Store context
        self.context_X = X_context
        self.context_y = y_context
        self.feature_names = feature_names
        
        # Fit the model
        self.model.fit(X_context, y_context)
        self.is_fitted = True
        
        # Print context statistics
        print(f"TabPFN v2 context fitted:")
        print(f"  Samples: {X_context.shape[0]}")
        print(f"  Features: {X_context.shape[1]}")
        if self.task_type == 'classification':
            unique, counts = np.unique(y_context, return_counts=True)
            print(f"  Class distribution: {dict(zip(unique, counts))}")
        else:
            print(f"  Target range: [{y_context.min():.2f}, {y_context.max():.2f}]")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only)
        
        Args:
            X: Query features of shape (n_query, n_features)
            
        Returns:
            Probabilities of shape (n_query, n_classes)
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_context first.")
        
        # Apply feature selection if needed
        if self.selected_features is not None:
            X = X[:, self.selected_features]
        
        # Get predictions
        proba = self.model.predict_proba(X)
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (classification or regression)
        
        Args:
            X: Query features of shape (n_query, n_features)
            
        Returns:
            Predictions of shape (n_query,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_context first.")
        
        # Apply feature selection if needed
        if self.selected_features is not None:
            X = X[:, self.selected_features]
        
        # Get predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_context first.")
        
        # Apply feature selection if needed
        if self.selected_features is not None:
            X_test = X_test[:, self.selected_features]
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        metrics = {}
        
        if self.task_type == 'classification':
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            
            # ROC-AUC for binary classification
            if len(np.unique(y_test)) == 2:
                proba = self.predict_proba(X_test)
                metrics['roc_auc'] = roc_auc_score(y_test, proba[:, 1])
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if available)
        
        Returns:
            Feature importance scores or None
        """
        # TabPFN doesn't provide direct feature importance
        # Could implement permutation importance
        return None
    
    def save_context(self, filepath: str):
        """Save context data for reproducibility"""
        np.savez_compressed(
            filepath,
            context_X=self.context_X,
            context_y=self.context_y,
            selected_features=self.selected_features,
            feature_names=self.feature_names if hasattr(self, 'feature_names') else None
        )
    
    def load_context(self, filepath: str):
        """Load saved context data"""
        data = np.load(filepath, allow_pickle=True)
        self.context_X = data['context_X']
        self.context_y = data['context_y']
        self.selected_features = data['selected_features']
        
        if data['feature_names'] is not None:
            self.feature_names = data['feature_names'].tolist()
        
        # Re-fit model
        self.model.fit(self.context_X, self.context_y)
        self.is_fitted = True
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of metric lists across folds
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Choose splitter
        if self.task_type == 'classification':
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        all_metrics = defaultdict(list)
        
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit and evaluate
            self.fit_context(X_train, y_train)
            metrics = self.evaluate(X_test, y_test)
            
            # Store metrics
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            print(f"Fold {fold_idx + 1}/{cv}: {metrics}")
        
        # Add mean and std
        for key in list(all_metrics.keys()):
            values = all_metrics[key]
            all_metrics[f"{key}_mean"] = np.mean(values)
            all_metrics[f"{key}_std"] = np.std(values)
        
        return dict(all_metrics)
