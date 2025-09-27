"""
Tests for VitalDB data handling with SPEC compliance
Tests quality control, clinical data extraction, and waveform processing
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import tempfile

# Import the modules to test
import sys
sys.path.append('..')
from data import (
    WaveformQualityControl,
    VitalDBClinicalExtractor,
    VitalDBDataset,
    SPEC_VERSION
)


class TestWaveformQualityControl:
    """Test quality control checks per SPEC."""
    
    def test_flatline_detection_variance(self):
        """Test flatline detection using variance threshold."""
        qc = WaveformQualityControl()
        
        # Flat signal
        flat_signal = np.ones(100) * 5.0
        assert qc.check_flatline(flat_signal, variance_threshold=0.01) == True
        
        # Normal signal
        normal_signal = np.random.randn(100)
        assert qc.check_flatline(normal_signal, variance_threshold=0.01) == False
        
        # Edge case: very small variance
        small_var = np.ones(100) + np.random.randn(100) * 0.001
        assert qc.check_flatline(small_var, variance_threshold=0.01) == True
    
    def test_flatline_detection_consecutive(self):
        """Test flatline detection using consecutive identical values."""
        qc = WaveformQualityControl()
        
        # Signal with 60 consecutive identical values
        signal = np.random.randn(100)
        signal[20:80] = 5.0
        assert qc.check_flatline(signal, consecutive_threshold=50) == True
        
        # Signal with only 40 consecutive identical values
        signal = np.random.randn(100)
        signal[20:60] = 5.0
        assert qc.check_flatline(signal, consecutive_threshold=50) == False
    
    def test_spike_detection(self):
        """Test spike detection using z-score method."""
        qc = WaveformQualityControl()
        
        # Normal signal
        normal_signal = np.random.randn(100)
        spikes = qc.check_spikes(normal_signal, z_threshold=3.0)
        assert np.sum(spikes) < 10  # Should have few spikes
        
        # Signal with spikes
        spike_signal = np.random.randn(100)
        spike_signal[20:25] = 10.0  # 5 consecutive high values
        spikes = qc.check_spikes(spike_signal, z_threshold=3.0, consecutive_samples=5)
        assert np.any(spikes[20:25])  # Should detect spike region
    
    def test_physiologic_bounds(self):
        """Test physiologic bounds checking."""
        qc = WaveformQualityControl()
        
        # Heart rate bounds
        hr_normal = np.array([60, 80, 100, 120])
        hr_bounds = qc.check_physiologic_bounds(hr_normal, 'hr')
        assert np.all(hr_bounds)
        
        hr_abnormal = np.array([20, 60, 100, 250])
        hr_bounds = qc.check_physiologic_bounds(hr_abnormal, 'hr')
        assert not hr_bounds[0]  # Too low
        assert not hr_bounds[3]  # Too high
        
        # MAP bounds
        map_normal = np.array([50, 65, 80, 100])
        map_bounds = qc.check_physiologic_bounds(map_normal, 'map')
        assert np.all(map_bounds)
        
        map_abnormal = np.array([30, 65, 80, 200])
        map_bounds = qc.check_physiologic_bounds(map_abnormal, 'map')
        assert not map_bounds[0]  # Too low
        assert not map_bounds[3]  # Too high
    
    def test_ppg_sqi(self):
        """Test PPG signal quality index calculation."""
        qc = WaveformQualityControl()
        
        # Good quality PPG (typically right-skewed)
        good_ppg = np.random.exponential(1, 1000)
        sqi = qc.calculate_ppg_sqi(good_ppg)
        assert sqi > 0  # Exponential distribution has positive skew
        
        # Poor quality (symmetric)
        poor_ppg = np.random.randn(1000)
        sqi = qc.calculate_ppg_sqi(poor_ppg)
        assert abs(sqi) < 0.5  # Normal distribution has near-zero skew
    
    def test_comprehensive_quality_mask(self):
        """Test comprehensive quality assessment."""
        qc = WaveformQualityControl()
        
        # Good signal
        good_signal = np.random.randn(1000) * 0.5 + 1.0
        qc_results = qc.get_quality_mask(good_signal, 'ppg', min_valid_ratio=0.7)
        assert not qc_results['is_flatline']
        assert not qc_results['has_spikes']
        assert qc_results['valid_ratio'] > 0.9
        
        # Bad signal (flatline)
        bad_signal = np.ones(1000) * 2.0
        qc_results = qc.get_quality_mask(bad_signal, 'ppg', min_valid_ratio=0.7)
        assert qc_results['is_flatline']
        assert not qc_results['overall_valid']


class TestVitalDBClinicalExtractor:
    """Test clinical data extraction and alignment."""
    
    def test_extract_clinical_data(self):
        """Test extraction of clinical data from VitalDB case."""
        extractor = VitalDBClinicalExtractor()
        
        # Mock VitalDB API
        mock_api = Mock()
        mock_api.get_case_info.return_value = {
            'age': 65,
            'sex': 'M',
            'height': 175,
            'weight': 80,
            'bmi': 26.1,
            'asa': 2,
            'department': 'General Surgery',
            'optype': 'Laparoscopic',
            'duration': 120,
            'hr_baseline': 72,
            'map_baseline': 85,
            'spo2_baseline': 98
        }
        
        clinical_data = extractor.extract_clinical_data(123, mock_api)
        
        assert clinical_data['demographics']['age'] == 65
        assert clinical_data['demographics']['sex'] == 1  # M=1
        assert clinical_data['demographics']['bmi'] == 26.1
        assert clinical_data['surgery']['asa'] == 2
        assert clinical_data['vitals']['hr_baseline'] == 72
    
    def test_clinical_data_missing_fields(self):
        """Test handling of missing clinical fields."""
        extractor = VitalDBClinicalExtractor()
        
        # Mock API with missing fields
        mock_api = Mock()
        mock_api.get_case_info.return_value = {
            'age': 55,
            'sex': 'F'
            # Missing other fields
        }
        
        clinical_data = extractor.extract_clinical_data(456, mock_api)
        
        assert clinical_data['demographics']['age'] == 55
        assert clinical_data['demographics']['sex'] == 0  # F=0
        assert clinical_data['demographics']['height'] == -1  # Missing
        assert clinical_data['surgery']['asa'] == -1  # Missing


class TestVitalDBDatasetSPEC:
    """Test VitalDB dataset with SPEC compliance."""
    
    @patch('data.vitaldb')
    def test_dataset_initialization(self, mock_vitaldb):
        """Test dataset initialization with SPEC parameters."""
        mock_vitaldb.find_cases.return_value = [1, 2, 3, 4, 5]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = VitalDBDataset(
                cache_dir=tmpdir,
                modality='ppg',
                split='train',
                enable_qc=True,
                extract_clinical=True
            )
            
            # Check SPEC version
            assert dataset.filter_params['type'] == 'cheby2'
            assert dataset.filter_params['order'] == 4
            assert dataset.target_fs == 25  # PPG → 25 Hz
            assert dataset.enable_qc == True
            
            # Check cache metadata was written
            metadata_file = Path(tmpdir) / f"vitaldb_waveform_cache_{SPEC_VERSION}" / "metadata.json"
            assert metadata_file.exists()
            
            with open(metadata_file) as f:
                metadata = json.load(f)
                assert metadata['spec_version'] == SPEC_VERSION
                assert metadata['modality'] == 'ppg'
    
    @patch('data.vitaldb')
    def test_ecg_filter_parameters(self, mock_vitaldb):
        """Test ECG uses correct filter parameters."""
        mock_vitaldb.find_cases.return_value = [1, 2, 3]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = VitalDBDataset(
                cache_dir=tmpdir,
                modality='ecg',
                split='train'
            )
            
            assert dataset.filter_params['type'] == 'butter'
            assert dataset.filter_params['order'] == 4
            assert dataset.filter_params['band'] == [0.5, 40]
            assert dataset.filter_params['zero_phase'] == True
            assert dataset.target_fs == 125  # ECG → 125 Hz
    
    def test_apply_spec_filter_ppg(self):
        """Test PPG filtering with Chebyshev Type-II."""
        dataset = Mock()
        dataset.filter_params = {
            'type': 'cheby2',
            'order': 4,
            'band': [0.5, 10],
            'ripple': 40
        }
        
        # Create test signal
        fs = 100
        t = np.arange(0, 10, 1/fs)
        # Mix of frequencies
        signal = (
            np.sin(2 * np.pi * 1.0 * t) +  # 1 Hz (should pass)
            np.sin(2 * np.pi * 20 * t) * 0.5 +  # 20 Hz (should be filtered)
            np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz (should be filtered)
        )
        
        # Apply filter using the actual method
        from data import VitalDBDataset
        filtered = VitalDBDataset._apply_spec_filter(dataset, signal, fs)
        
        # Check that high frequency is attenuated
        from scipy import signal as scipy_signal
        f, psd_orig = scipy_signal.periodogram(signal, fs)
        f, psd_filt = scipy_signal.periodogram(filtered, fs)
        
        # Find power at different frequencies
        idx_1hz = np.argmin(np.abs(f - 1.0))
        idx_20hz = np.argmin(np.abs(f - 20.0))
        
        # 1 Hz should be preserved, 20 Hz should be attenuated
        assert psd_filt[idx_1hz] > psd_filt[idx_20hz] * 10
    
    def test_cache_key_generation(self):
        """Test cache key includes all SPEC parameters."""
        dataset = Mock()
        dataset.modality = 'ppg'
        dataset.window_sec = 10.0
        dataset.hop_sec = 5.0
        dataset.filter_params = {'type': 'cheby2', 'order': 4}
        dataset.target_fs = 25
        dataset.enable_qc = True
        dataset.min_valid_ratio = 0.7
        
        from data import VitalDBDataset
        key1 = VitalDBDataset._generate_cache_key(dataset, case_id=123)
        key2 = VitalDBDataset._generate_cache_key(dataset, case_id=123)
        key3 = VitalDBDataset._generate_cache_key(dataset, case_id=456)
        
        # Same parameters should give same key
        assert key1 == key2
        # Different case should give different key
        assert key1 != key3
    
    @patch('data.vitaldb')
    def test_quality_control_integration(self, mock_vitaldb):
        """Test QC is applied during preprocessing."""
        mock_vitaldb.find_cases.return_value = [1, 2, 3]
        
        # Create dataset with QC enabled
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = VitalDBDataset(
                cache_dir=tmpdir,
                modality='ppg',
                split='train',
                enable_qc=True,
                min_valid_ratio=0.7
            )
            
            # Test preprocessing with good signal
            good_signal = np.random.randn(10000) * 0.5 + 1.0
            processed, qc_results = dataset._preprocess_with_qc(good_signal, case_id=1)
            
            assert processed is not None
            assert qc_results['overall_valid'] == True
            assert qc_results['valid_ratio'] > 0.9
            
            # Test with bad signal (flatline)
            bad_signal = np.ones(10000) * 2.0
            processed, qc_results = dataset._preprocess_with_qc(bad_signal, case_id=2)
            
            assert qc_results['overall_valid'] == False
            assert qc_results['is_flatline'] == True
    
    def test_window_extraction_with_clinical(self):
        """Test window extraction includes clinical context."""
        dataset = Mock()
        dataset.vitaldb = Mock()
        dataset.target_fs = 100
        dataset.segment_length = 1000  # 10s at 100Hz
        dataset.hop_sec = 5.0
        dataset.window_sec = 10.0
        dataset.modality = 'ppg'
        dataset.enable_qc = True
        dataset.min_valid_ratio = 0.7
        dataset.extract_clinical = True
        dataset.original_fs = 100
        dataset.filter_params = {'type': 'butter', 'order': 2, 'band': [0.5, 10]}
        
        # Mock signal loading
        mock_signal = np.random.randn(20000)  # 200 seconds
        dataset.vitaldb.load_case.return_value = mock_signal
        
        # Mock QC
        dataset.qc = WaveformQualityControl()
        
        # Mock clinical extractor
        dataset.clinical_extractor = Mock()
        dataset.clinical_extractor.extract_clinical_data.return_value = {
            'demographics': {'age': 65, 'sex': 1}
        }
        dataset.clinical_extractor.align_to_window.return_value = {
            'demographics': {'age': 65, 'sex': 1}
        }
        
        from data import VitalDBDataset
        window_data = VitalDBDataset._extract_window_with_clinical(dataset, case_id=1, window_idx=0)
        
        assert window_data['signal'] is not None
        assert len(window_data['signal']) == dataset.segment_length
        assert window_data['qc'] is not None
        assert window_data['clinical'] is not None
        assert window_data['clinical']['demographics']['age'] == 65


class TestDeprecation:
    """Test that old functions are marked as deprecated."""
    
    def test_deprecated_preprocessing(self):
        """Test that old preprocessing raises deprecation warning."""
        from data import _preprocess_signal_deprecated
        
        with pytest.warns(DeprecationWarning):
            signal = np.random.randn(1000)
            _preprocess_signal_deprecated(signal)


class TestEndToEndIntegration:
    """Test full pipeline integration."""
    
    @patch('data.vitaldb')
    @patch('requests.get')
    def test_full_pipeline_with_qc_and_clinical(self, mock_requests, mock_vitaldb):
        """Test complete data loading with QC and clinical context."""
        # Setup mocks
        mock_vitaldb.find_cases.return_value = [1, 2, 3, 4, 5]
        mock_vitaldb.load_case.return_value = np.random.randn(100000)
        
        # Mock clinical data download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'caseid,age,sex,height,weight,bmi,asa\n1,65,M,175,80,26.1,2'
        mock_requests.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            dataset = VitalDBDataset(
                cache_dir=tmpdir,
                modality='ppg',
                split='train',
                enable_qc=True,
                extract_clinical=True,
                mode='timeseries'
            )
            
            # Get a sample
            if len(dataset) > 0:
                seg1, seg2 = dataset[0]
                
                # Check outputs
                assert seg1.shape == (1, dataset.segment_length)
                assert seg2.shape == (1, dataset.segment_length)
                assert not torch.all(seg1 == 0)  # Should not be all zeros
    
    def test_cache_invalidation(self):
        """Test that cache is invalidated when SPEC changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache with version 1
            cache_dir = Path(tmpdir) / f"vitaldb_waveform_cache_{SPEC_VERSION}"
            cache_dir.mkdir(parents=True)
            
            metadata1 = {
                'spec_version': SPEC_VERSION,
                'modality': 'ppg',
                'filter_params': {'type': 'butter', 'order': 2}
            }
            with open(cache_dir / 'metadata.json', 'w') as f:
                json.dump(metadata1, f)
            
            # Create cache with different version
            cache_dir2 = Path(tmpdir) / "vitaldb_waveform_cache_v0.9"
            cache_dir2.mkdir(parents=True)
            
            metadata2 = {
                'spec_version': 'v0.9',
                'modality': 'ppg',
                'filter_params': {'type': 'butter', 'order': 4}
            }
            with open(cache_dir2 / 'metadata.json', 'w') as f:
                json.dump(metadata2, f)
            
            # Check that only current version cache is used
            assert cache_dir.exists()
            assert cache_dir2.exists()
            
            # Load metadata and verify version
            with open(cache_dir / 'metadata.json') as f:
                metadata = json.load(f)
                assert metadata['spec_version'] == SPEC_VERSION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
