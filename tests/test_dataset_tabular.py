"""
Test VitalDB dataset in tabular mode
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatasetTabular:
    """Test suite for VitalDB dataset in tabular mode"""
    
    @patch('data.VitalDBFeatureExtractor')
    @patch('data.VitalDBLabelCreator')
    @patch('vitaldb.find_cases')
    @patch('vitaldb.load_case')
    def test_tabular_mode_output_format(self, mock_load, mock_find, mock_label, mock_feature):
        """Test that dataset in tabular mode returns correct format"""
        from data import VitalDBDataset
        
        # Mock VitalDB API
        mock_find.return_value = [1, 2, 3, 4, 5]  # 5 cases
        
        # Create mock signal data
        mock_signal = np.random.randn(10000)
        mock_load.return_value = mock_signal
        
        # Mock feature extractor
        mock_extractor = MagicMock()
        mock_extractor.extract_windows.return_value = [
            {'ppg': np.random.randn(1250), 'ecg': np.random.randn(1250)}
            for _ in range(10)
        ]
        mock_extractor.compute_features_v1_basic.return_value = np.random.randn(50)
        mock_feature.return_value = mock_extractor
        
        # Mock label creator
        mock_creator = MagicMock()
        mock_creator.create_ioh_labels.return_value = {
            'labels': {'ioh_5min': 1},
            'metadata': {}
        }
        mock_label.return_value = mock_creator
        
        # Create dataset
        dataset = VitalDBDataset(
            mode='tabular',
            feature_set='v1_basic',
            window_sec=10,
            overlap=0.5,
            target_task='ioh',
            horizon_min=5,
            split='train'
        )
        
        # Get an item
        features, label, context = dataset[0]
        
        # Check output format
        assert isinstance(features, torch.Tensor), "Features should be torch.Tensor"
        assert features.shape == (50,), f"Features shape should be (50,), got {features.shape}"
        
        assert isinstance(label, torch.Tensor), "Label should be torch.Tensor"
        assert label.shape == (1,), f"Label shape should be (1,), got {label.shape}"
        
        assert isinstance(context, dict), "Context should be dictionary"
        assert 'patient_id' in context, "Context should contain patient_id"
        assert 'window_idx' in context, "Context should contain window_idx"
        assert 'split' in context, "Context should contain split"
    
    @patch('vitaldb.find_cases')
    def test_no_patient_overlap_splits(self, mock_find):
        """Test that there's no patient overlap across train/val/test splits"""
        from data import VitalDBDataset
        
        # Mock 100 cases
        all_cases = list(range(100))
        mock_find.return_value = all_cases
        
        # Create datasets for each split
        train_dataset = VitalDBDataset(
            mode='tabular',
            split='train',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        val_dataset = VitalDBDataset(
            mode='tabular', 
            split='val',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_dataset = VitalDBDataset(
            mode='tabular',
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        # Get cases for each split
        train_cases = set(train_dataset.cases)
        val_cases = set(val_dataset.cases)
        test_cases = set(test_dataset.cases)
        
        # Check no overlap
        assert len(train_cases & val_cases) == 0, \
            "Train and val sets should not overlap"
        assert len(train_cases & test_cases) == 0, \
            "Train and test sets should not overlap"
        assert len(val_cases & test_cases) == 0, \
            "Val and test sets should not overlap"
        
        # Check split sizes (approximately)
        total = len(all_cases)
        assert abs(len(train_cases) - 0.6 * total) <= 1, \
            f"Train set should be ~60% of data, got {len(train_cases)}/{total}"
        assert abs(len(val_cases) - 0.2 * total) <= 1, \
            f"Val set should be ~20% of data, got {len(val_cases)}/{total}"
        assert abs(len(test_cases) - 0.2 * total) <= 1, \
            f"Test set should be ~20% of data, got {len(test_cases)}/{total}"
    
    @patch('vitaldb.find_cases')
    def test_context_patient_tracking(self, mock_find):
        """Test that context properly tracks patient IDs"""
        from data import VitalDBDataset
        
        # Mock cases
        mock_find.return_value = [101, 102, 103]
        
        with patch('vitaldb.load_case') as mock_load:
            # Mock signal data
            mock_load.return_value = np.random.randn(10000)
            
            dataset = VitalDBDataset(
                mode='tabular',
                split='train'
            )
            
            # Track patient IDs seen
            patient_ids_seen = set()
            
            # Sample multiple items
            for i in range(min(30, len(dataset))):
                try:
                    features, label, context = dataset[i]
                    patient_ids_seen.add(context['patient_id'])
                except:
                    # Handle mock limitations
                    pass
            
            # Should see patient IDs from our cases
            assert len(patient_ids_seen) > 0, "Should track some patient IDs"
            for pid in patient_ids_seen:
                assert pid in [101, 102, 103], \
                    f"Patient ID {pid} not in expected cases"
    
    def test_feature_extractor_initialized(self):
        """Test that feature extractor is properly initialized in tabular mode"""
        from data import VitalDBDataset
        
        with patch('vitaldb.find_cases') as mock_find:
            mock_find.return_value = [1, 2, 3]
            
            dataset = VitalDBDataset(
                mode='tabular',
                feature_set='v1_basic',
                window_sec=10,
                overlap=0.5
            )
            
            assert hasattr(dataset, 'feature_extractor'), \
                "Dataset should have feature_extractor in tabular mode"
            assert hasattr(dataset, 'label_creator'), \
                "Dataset should have label_creator in tabular mode"
            
            # Check feature extractor config
            assert dataset.feature_extractor.feature_set == 'v1_basic'
            assert dataset.feature_extractor.total_features == 50
    
    def test_window_indices_structure(self):
        """Test window indices structure in tabular mode"""
        from data import VitalDBDataset
        
        with patch('vitaldb.find_cases') as mock_find:
            cases = [201, 202, 203, 204, 205]
            mock_find.return_value = cases
            
            dataset = VitalDBDataset(
                mode='tabular',
                split='train',
                train_ratio=0.6  # Should get 3 cases
            )
            
            # Check window indices created
            assert hasattr(dataset, 'window_indices'), \
                "Dataset should have window_indices in tabular mode"
            
            # Should have indices for train cases
            assert len(dataset.window_indices) > 0, \
                "Should have window indices"
    
    @patch('vitaldb.find_cases')
    @patch('vitaldb.load_case')
    def test_missing_channel_handling(self, mock_load, mock_find):
        """Test handling of missing channels"""
        from data import VitalDBDataset
        
        mock_find.return_value = [1]
        
        # Simulate missing ECG
        def load_case_side_effect(case_id, tracks):
            if 'ECG_II' in tracks:
                return None  # No ECG
            elif 'PLETH' in tracks:
                return np.random.randn(10000)  # PPG exists
            elif 'ABP' in tracks:
                return np.random.randn(10000)  # ABP exists
            return None
        
        mock_load.side_effect = load_case_side_effect
        
        dataset = VitalDBDataset(
            mode='tabular',
            split='train'
        )
        
        # Should handle missing channel gracefully
        try:
            features, label, context = dataset[0]
            
            # Features should still be 50-dimensional
            assert features.shape == (50,), \
                "Should return 50 features even with missing channel"
            
            # Some features should be NaN (ECG features)
            assert torch.any(torch.isnan(features)), \
                "Should have NaN features for missing ECG"
        except Exception as e:
            pytest.fail(f"Should handle missing channel gracefully: {e}")
    
    def test_mode_parameter_switches_behavior(self):
        """Test that mode parameter correctly switches dataset behavior"""
        from data import VitalDBDataset
        
        with patch('vitaldb.find_cases') as mock_find:
            mock_find.return_value = [1, 2, 3]
            
            # Timeseries mode
            ts_dataset = VitalDBDataset(
                mode='timeseries',
                split='train'
            )
            
            # Should have segment pairs for timeseries
            assert hasattr(ts_dataset, 'segment_pairs'), \
                "Timeseries mode should have segment_pairs"
            
            # Tabular mode
            tab_dataset = VitalDBDataset(
                mode='tabular',
                split='train'
            )
            
            # Should have window indices for tabular
            assert hasattr(tab_dataset, 'window_indices'), \
                "Tabular mode should have window_indices"
            
            # Should have feature extractor only in tabular
            assert hasattr(tab_dataset, 'feature_extractor'), \
                "Tabular mode should have feature_extractor"
            assert not hasattr(ts_dataset, 'feature_extractor'), \
                "Timeseries mode should not have feature_extractor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
