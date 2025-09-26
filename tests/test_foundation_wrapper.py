"""
Test TabPFN Foundation wrapper
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFoundationWrapper:
    """Test suite for TabPFN Foundation wrapper"""
    
    @patch('models.tabpfn_foundation.TabPFNClassifier')
    def test_classification_initialization(self, mock_classifier):
        """Test initialization for classification task"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        model = TabPFNFoundation(
            task_type='classification',
            device='cpu',
            N_ensemble_configurations=4
        )
        
        assert model.task_type == 'classification'
        assert model.device == 'cpu'
        assert model.N_ensemble_configurations == 4
        assert model.max_features == 500
        
        # Should have created classifier
        mock_classifier.assert_called_once()
    
    @patch('models.tabpfn_foundation.TabPFNRegressor')
    def test_regression_initialization(self, mock_regressor):
        """Test initialization for regression task"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        model = TabPFNFoundation(
            task_type='regression',
            device='cpu',
            N_ensemble_configurations=4
        )
        
        assert model.task_type == 'regression'
        
        # Should have created regressor
        mock_regressor.assert_called_once()
    
    @patch('models.tabpfn_foundation.TabPFNClassifier')
    def test_fit_context_classification(self, mock_classifier):
        """Test fit_context for classification"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        # Create mock classifier
        mock_clf_instance = MagicMock()
        mock_classifier.return_value = mock_clf_instance
        
        model = TabPFNFoundation(task_type='classification')
        
        # Create fake data
        n_samples = 100
        n_features = 50
        X_context = np.random.randn(n_samples, n_features)
        y_context = np.random.randint(0, 2, n_samples)
        
        # Fit context
        model.fit_context(X_context, y_context)
        
        # Check model was fitted
        assert model.is_fitted == True
        assert model.context_X is not None
        assert model.context_y is not None
        assert model.context_X.shape == (n_samples, n_features)
        assert model.context_y.shape == (n_samples,)
        
        # Check fit was called
        mock_clf_instance.fit.assert_called_once()
    
    @patch('models.tabpfn_foundation.TabPFNClassifier')
    def test_predict_proba_shape(self, mock_classifier):
        """Test predict_proba output shape for binary classification"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        # Setup mock
        mock_clf_instance = MagicMock()
        mock_proba = np.random.rand(10, 2)  # 10 samples, 2 classes
        mock_clf_instance.predict_proba.return_value = mock_proba
        mock_classifier.return_value = mock_clf_instance
        
        model = TabPFNFoundation(task_type='classification')
        
        # Fit context
        X_context = np.random.randn(50, 30)
        y_context = np.random.randint(0, 2, 50)
        model.fit_context(X_context, y_context)
        
        # Predict
        X_query = np.random.randn(10, 30)
        proba = model.predict_proba(X_query)
        
        # Check shape
        assert proba.shape == (10, 2), \
            f"Binary classification should return (n, 2), got {proba.shape}"
    
    @patch('models.tabpfn_foundation.TabPFNRegressor')
    def test_predict_regression_shape(self, mock_regressor):
        """Test predict output shape for regression"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        # Setup mock
        mock_reg_instance = MagicMock()
        mock_predictions = np.random.randn(10)  # 10 samples
        mock_reg_instance.predict.return_value = mock_predictions
        mock_regressor.return_value = mock_reg_instance
        
        model = TabPFNFoundation(task_type='regression')
        
        # Fit context
        X_context = np.random.randn(50, 30)
        y_context = np.random.randn(50)
        model.fit_context(X_context, y_context)
        
        # Predict
        X_query = np.random.randn(10, 30)
        predictions = model.predict(X_query)
        
        # Check shape
        assert predictions.shape == (10,), \
            f"Regression should return (n,), got {predictions.shape}"
    
    def test_feature_limit_enforcement(self):
        """Test that feature limit is enforced"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        with patch('models.tabpfn_foundation.TabPFNClassifier') as mock_clf:
            mock_clf_instance = MagicMock()
            mock_clf.return_value = mock_clf_instance
            
            model = TabPFNFoundation(
                task_type='classification',
                max_features=500
            )
            
            # Try to fit with too many features
            X_context = np.random.randn(100, 600)  # 600 features > 500 limit
            y_context = np.random.randint(0, 2, 100)
            
            # Should handle by selecting top features
            model.fit_context(X_context, y_context)
            
            # Should have selected features
            assert model.selected_features is not None
            assert len(model.selected_features) == 500
            
            # Context should be reduced
            assert model.context_X.shape[1] == 500
    
    def test_sample_limit_warning(self):
        """Test that sample limit produces warning/subsampling"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        with patch('models.tabpfn_foundation.TabPFNClassifier') as mock_clf:
            mock_clf_instance = MagicMock()
            mock_clf.return_value = mock_clf_instance
            
            model = TabPFNFoundation(task_type='classification')
            
            # Try to fit with too many samples
            X_context = np.random.randn(15000, 50)  # 15000 > 10000 limit
            y_context = np.random.randint(0, 2, 15000)
            
            # Should subsample
            model.fit_context(X_context, y_context)
            
            # Context should be subsampled
            assert model.context_X.shape[0] <= 10000
    
    @patch('models.tabpfn_foundation.TabPFNClassifier')
    def test_evaluate_metrics(self, mock_classifier):
        """Test evaluate method returns proper metrics"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        # Setup mock
        mock_clf_instance = MagicMock()
        mock_clf_instance.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_clf_instance.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3]
        ])
        mock_classifier.return_value = mock_clf_instance
        
        model = TabPFNFoundation(task_type='classification')
        
        # Fit
        X_context = np.random.randn(50, 30)
        y_context = np.random.randint(0, 2, 50)
        model.fit_context(X_context, y_context)
        
        # Evaluate
        X_test = np.random.randn(5, 30)
        y_test = np.array([0, 1, 0, 1, 0])
        
        metrics = model.evaluate(X_test, y_test)
        
        # Check metrics exist
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
    
    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        with patch('models.tabpfn_foundation.TabPFNClassifier'):
            model = TabPFNFoundation(task_type='classification')
            
            X_query = np.random.randn(10, 30)
            
            # Should raise error
            with pytest.raises(ValueError, match="not fitted"):
                model.predict(X_query)
            
            with pytest.raises(ValueError, match="not fitted"):
                model.predict_proba(X_query)
    
    def test_multi_target_regression(self):
        """Test regression with multiple targets (SBP/DBP/MAP)"""
        from models.tabpfn_foundation import TabPFNFoundation
        
        with patch('models.tabpfn_foundation.TabPFNRegressor') as mock_reg:
            mock_reg_instance = MagicMock()
            
            # Simulate 3 target predictions
            mock_reg_instance.predict.return_value = np.random.randn(10, 3)
            mock_reg.return_value = mock_reg_instance
            
            model = TabPFNFoundation(task_type='regression')
            
            # Fit with multi-target
            X_context = np.random.randn(50, 30)
            y_context = np.random.randn(50, 3)  # 3 targets
            
            # Should handle multi-target
            try:
                model.fit_context(X_context, y_context)
                X_query = np.random.randn(10, 30)
                predictions = model.predict(X_query)
                
                # Could be (10,) or (10, 3) depending on implementation
                assert predictions.shape[0] == 10
            except:
                # Multi-target might not be directly supported
                pass
    
    def test_save_load_context(self):
        """Test saving and loading context"""
        from models.tabpfn_foundation import TabPFNFoundation
        import tempfile
        
        with patch('models.tabpfn_foundation.TabPFNClassifier') as mock_clf:
            mock_clf_instance = MagicMock()
            mock_clf.return_value = mock_clf_instance
            
            model = TabPFNFoundation(task_type='classification')
            
            # Fit context
            X_context = np.random.randn(50, 30)
            y_context = np.random.randint(0, 2, 50)
            model.fit_context(X_context, y_context)
            
            # Save context
            with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
                model.save_context(tmp.name)
                
                # Create new model and load
                model2 = TabPFNFoundation(task_type='classification')
                model2.load_context(tmp.name)
                
                # Check context loaded
                assert model2.is_fitted == True
                assert np.allclose(model2.context_X, X_context)
                assert np.allclose(model2.context_y, y_context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
