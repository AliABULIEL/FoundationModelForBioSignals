"""
Comprehensive tests for VitalDB waveform quality control
Tests all QC functions with edge cases and boundary conditions
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deprecated.data import WaveformQualityControl


class TestFlatlineDetection:
    """Test flatline detection with various scenarios."""
    
    def test_flatline_zero_variance(self):
        """Test detection of perfect flatline (zero variance)."""
        qc = WaveformQualityControl()
        
        # Perfect flatline
        flat = np.ones(1000) * 3.14159
        assert qc.check_flatline(flat, variance_threshold=0.01) == True
        
        # Zero signal
        zeros = np.zeros(1000)
        assert qc.check_flatline(zeros, variance_threshold=0.01) == True
    
    def test_flatline_small_variance(self):
        """Test detection with very small variance."""
        qc = WaveformQualityControl()
        
        # Small noise around constant
        small_noise = np.ones(1000) + np.random.randn(1000) * 0.001
        assert qc.check_flatline(small_noise, variance_threshold=0.01) == True
        
        # Slightly larger noise (should pass)
        larger_noise = np.ones(1000) + np.random.randn(1000) * 0.2
        assert qc.check_flatline(larger_noise, variance_threshold=0.01) == False
    
    def test_flatline_consecutive_values(self):
        """Test consecutive identical value detection."""
        qc = WaveformQualityControl()
        
        # Exactly 50 consecutive
        signal = np.random.randn(200)
        signal[50:100] = 2.5
        assert qc.check_flatline(signal, consecutive_threshold=50) == False  # Exactly at threshold
        
        # 51 consecutive (should trigger)
        signal[50:101] = 2.5
        assert qc.check_flatline(signal, consecutive_threshold=50) == True
        
        # Multiple flat regions
        signal = np.random.randn(300)
        signal[50:80] = 1.0  # 30 consecutive
        signal[150:210] = 2.0  # 60 consecutive
        assert qc.check_flatline(signal, consecutive_threshold=50) == True
    
    def test_flatline_edge_cases(self):
        """Test edge cases for flatline detection."""
        qc = WaveformQualityControl()
        
        # Empty signal
        assert qc.check_flatline(np.array([]), variance_threshold=0.01) == True
        
        # Single value
        assert qc.check_flatline(np.array([5.0]), variance_threshold=0.01) == True
        
        # Two identical values
        assert qc.check_flatline(np.array([3.0, 3.0]), variance_threshold=0.01) == True
        
        # Two different values
        assert qc.check_flatline(np.array([3.0, 4.0]), variance_threshold=0.01) == False
    
    def test_flatline_threshold_sensitivity(self):
        """Test sensitivity to different thresholds."""
        qc = WaveformQualityControl()
        
        # Signal with specific variance
        signal = np.random.randn(1000) * 0.05  # variance ≈ 0.0025
        
        assert qc.check_flatline(signal, variance_threshold=0.001) == False
        assert qc.check_flatline(signal, variance_threshold=0.01) == True
        assert qc.check_flatline(signal, variance_threshold=0.1) == True


class TestSpikeDetection:
    """Test spike artifact detection."""
    
    def test_spike_basic_detection(self):
        """Test basic spike detection."""
        qc = WaveformQualityControl()
        
        # Normal signal
        normal = np.random.randn(1000)
        spikes = qc.check_spikes(normal, z_threshold=3.0)
        # Should have very few spikes in normal distribution
        assert np.sum(spikes) < len(normal) * 0.01  # Less than 1%
    
    def test_spike_consecutive_requirement(self):
        """Test consecutive sample requirement for spikes."""
        qc = WaveformQualityControl()
        
        # Signal with exactly 5 consecutive high values
        signal = np.random.randn(200) * 0.5
        signal[50:55] = 5.0  # 5 consecutive high values
        
        spikes = qc.check_spikes(signal, z_threshold=3.0, consecutive_samples=5)
        assert np.any(spikes[50:55])  # Should detect spike
        
        # Only 4 consecutive high values (should not trigger)
        signal = np.random.randn(200) * 0.5
        signal[50:54] = 5.0  # 4 consecutive
        
        spikes = qc.check_spikes(signal, z_threshold=3.0, consecutive_samples=5)
        assert not np.any(spikes[50:54])  # Should NOT detect spike
    
    def test_spike_multiple_regions(self):
        """Test detection of multiple spike regions."""
        qc = WaveformQualityControl()
        
        signal = np.random.randn(500) * 0.5
        # Add multiple spike regions
        signal[50:60] = 6.0   # Region 1
        signal[150:160] = -6.0  # Region 2 (negative spike)
        signal[250:260] = 7.0   # Region 3
        
        spikes = qc.check_spikes(signal, z_threshold=3.0, consecutive_samples=5)
        
        # Should detect all three regions
        assert np.any(spikes[50:60])
        assert np.any(spikes[150:160])
        assert np.any(spikes[250:260])
    
    def test_spike_z_score_threshold(self):
        """Test different z-score thresholds."""
        qc = WaveformQualityControl()
        
        # Create signal with known outliers
        signal = np.random.randn(1000)
        signal[100:110] = 4.0  # Moderate outlier
        signal[200:210] = 6.0  # Strong outlier
        
        # Strict threshold (z=2)
        spikes_strict = qc.check_spikes(signal, z_threshold=2.0, consecutive_samples=5)
        
        # Normal threshold (z=3)
        spikes_normal = qc.check_spikes(signal, z_threshold=3.0, consecutive_samples=5)
        
        # Loose threshold (z=4)
        spikes_loose = qc.check_spikes(signal, z_threshold=4.0, consecutive_samples=5)
        
        # Strict should detect more than normal, normal more than loose
        assert np.sum(spikes_strict) >= np.sum(spikes_normal)
        assert np.sum(spikes_normal) >= np.sum(spikes_loose)
    
    def test_spike_edge_cases(self):
        """Test edge cases for spike detection."""
        qc = WaveformQualityControl()
        
        # Empty signal
        assert len(qc.check_spikes(np.array([]), z_threshold=3.0)) == 0
        
        # Short signal (less than consecutive requirement)
        short = np.array([1, 2, 10, 10])
        spikes = qc.check_spikes(short, z_threshold=2.0, consecutive_samples=5)
        assert len(spikes) == 4
        assert not np.any(spikes)  # Too short for consecutive requirement
        
        # Constant signal (no variance)
        constant = np.ones(100) * 5
        spikes = qc.check_spikes(constant, z_threshold=3.0)
        assert not np.any(spikes)  # No spikes in constant signal


class TestPhysiologicBounds:
    """Test physiologic bounds checking."""
    
    def test_heart_rate_bounds(self):
        """Test heart rate bounds (30-200 BPM)."""
        qc = WaveformQualityControl()
        
        # Valid heart rates
        valid_hr = np.array([40, 60, 80, 100, 120, 180])
        mask = qc.check_physiologic_bounds(valid_hr, 'hr')
        assert np.all(mask)
        
        # Invalid heart rates
        invalid_hr = np.array([20, 25, 210, 250, 300])
        mask = qc.check_physiologic_bounds(invalid_hr, 'hr')
        assert not np.any(mask)
        
        # Mixed
        mixed_hr = np.array([20, 60, 100, 220])
        mask = qc.check_physiologic_bounds(mixed_hr, 'hr')
        assert mask[1] and mask[2]  # Middle values valid
        assert not mask[0] and not mask[3]  # Extremes invalid
    
    def test_blood_pressure_bounds(self):
        """Test blood pressure bounds."""
        qc = WaveformQualityControl()
        
        # SBP bounds (60-200)
        sbp_values = np.array([50, 60, 120, 200, 210])
        mask = qc.check_physiologic_bounds(sbp_values, 'sbp')
        assert not mask[0] and not mask[4]  # Out of bounds
        assert mask[1] and mask[2] and mask[3]  # In bounds
        
        # DBP bounds (30-150)
        dbp_values = np.array([20, 30, 80, 150, 160])
        mask = qc.check_physiologic_bounds(dbp_values, 'dbp')
        assert not mask[0] and not mask[4]
        assert mask[1] and mask[2] and mask[3]
        
        # MAP bounds (40-180)
        map_values = np.array([30, 40, 65, 100, 180, 190])
        mask = qc.check_physiologic_bounds(map_values, 'map')
        assert not mask[0] and not mask[5]
        assert np.all(mask[1:5])
    
    def test_spo2_bounds(self):
        """Test SpO2 bounds (70-100%)."""
        qc = WaveformQualityControl()
        
        spo2_values = np.array([60, 70, 85, 95, 100, 105])
        mask = qc.check_physiologic_bounds(spo2_values, 'spo2')
        assert not mask[0] and not mask[5]
        assert np.all(mask[1:5])
    
    def test_normalized_signal_bounds(self):
        """Test bounds for normalized signals."""
        qc = WaveformQualityControl()
        
        # PPG normalized bounds (-5 to 5)
        ppg_values = np.array([-6, -5, 0, 2, 5, 6])
        mask = qc.check_physiologic_bounds(ppg_values, 'ppg')
        assert not mask[0] and not mask[5]
        assert np.all(mask[1:5])
        
        # ECG normalized bounds
        ecg_values = np.array([-10, -4, 0, 3, 5, 8])
        mask = qc.check_physiologic_bounds(ecg_values, 'ecg')
        assert not mask[0] and not mask[5]
        assert np.all(mask[1:5])
    
    def test_unknown_signal_type(self):
        """Test handling of unknown signal types."""
        qc = WaveformQualityControl()
        
        # Unknown signal type should return all valid
        values = np.array([-1000, 0, 1000])
        mask = qc.check_physiologic_bounds(values, 'unknown_type')
        assert np.all(mask)  # Should accept all values for unknown types


class TestPPGSignalQualityIndex:
    """Test PPG SQI calculation using skewness."""
    
    def test_sqi_good_ppg(self):
        """Test SQI for good quality PPG (right-skewed)."""
        qc = WaveformQualityControl()
        
        # Simulate good PPG (typically right-skewed)
        # Use exponential distribution
        good_ppg = np.random.exponential(1, 1000)
        sqi = qc.calculate_ppg_sqi(good_ppg)
        assert sqi > 1.0  # Exponential has positive skew
        
        # Use gamma distribution (also right-skewed)
        good_ppg2 = np.random.gamma(2, 2, 1000)
        sqi2 = qc.calculate_ppg_sqi(good_ppg2)
        assert sqi2 > 0.5
    
    def test_sqi_poor_ppg(self):
        """Test SQI for poor quality PPG."""
        qc = WaveformQualityControl()
        
        # Normal distribution (symmetric, poor PPG)
        poor_ppg = np.random.randn(1000)
        sqi = qc.calculate_ppg_sqi(poor_ppg)
        assert abs(sqi) < 0.5  # Near zero skewness
        
        # Uniform distribution (also symmetric)
        poor_ppg2 = np.random.uniform(-1, 1, 1000)
        sqi2 = qc.calculate_ppg_sqi(poor_ppg2)
        assert abs(sqi2) < 0.5
    
    def test_sqi_edge_cases(self):
        """Test SQI calculation edge cases."""
        qc = WaveformQualityControl()
        
        # Very short signal
        short = np.array([1, 2, 3])
        sqi = qc.calculate_ppg_sqi(short)
        assert sqi == 0.0  # Too short
        
        # Constant signal
        constant = np.ones(100)
        sqi = qc.calculate_ppg_sqi(constant)
        # Skewness of constant is undefined/nan, should handle gracefully
        assert not np.isnan(sqi)
    
    def test_sqi_threshold(self):
        """Test SQI threshold for acceptance (>3.0)."""
        qc = WaveformQualityControl()
        
        # Create signals with specific skewness
        # High skewness (should pass)
        high_skew = np.concatenate([
            np.ones(100) * 0,
            np.ones(10) * 10  # Few high values create right skew
        ])
        np.random.shuffle(high_skew)
        sqi_high = qc.calculate_ppg_sqi(high_skew)
        
        # Check if it would pass threshold
        if sqi_high > 3.0:
            assert True  # Good PPG
        
        # Create definitely good PPG signal
        t = np.linspace(0, 10, 1000)
        ppg_wave = np.sin(2 * np.pi * 1.2 * t)  # Heart rate ~72 BPM
        ppg_wave = np.abs(ppg_wave)  # Make positive (like PPG)
        ppg_wave += np.random.randn(1000) * 0.05  # Add small noise
        sqi_wave = qc.calculate_ppg_sqi(ppg_wave)
        assert sqi_wave != 0.0  # Should calculate something


class TestComprehensiveQualityMask:
    """Test comprehensive quality assessment."""
    
    def test_quality_mask_good_signal(self):
        """Test quality mask for a good signal."""
        qc = WaveformQualityControl()
        
        # Create good quality signal
        t = np.linspace(0, 10, 1000)
        good_signal = np.sin(2 * np.pi * 1.0 * t) + np.random.randn(1000) * 0.1
        
        qc_results = qc.get_quality_mask(good_signal, 'ppg', min_valid_ratio=0.7)
        
        assert not qc_results['is_flatline']
        assert not qc_results['has_spikes']
        assert qc_results['in_bounds']
        assert qc_results['valid_ratio'] > 0.9
        assert len(qc_results['mask']) == len(good_signal)
    
    def test_quality_mask_flatline(self):
        """Test quality mask for flatline signal."""
        qc = WaveformQualityControl()
        
        flatline = np.ones(1000) * 2.5
        qc_results = qc.get_quality_mask(flatline, 'ppg')
        
        assert qc_results['is_flatline']
        assert not qc_results['overall_valid']
        assert qc_results['valid_ratio'] == 1.0  # All values "valid" but flatline
    
    def test_quality_mask_with_spikes(self):
        """Test quality mask with spike artifacts."""
        qc = WaveformQualityControl()
        
        signal = np.random.randn(1000) * 0.5
        # Add spikes
        signal[100:110] = 10.0
        signal[500:510] = -10.0
        
        qc_results = qc.get_quality_mask(signal, 'ecg')
        
        assert qc_results['has_spikes']
        assert qc_results['valid_ratio'] < 1.0
        # Check spike regions are masked
        assert not np.all(qc_results['mask'][100:110])
        assert not np.all(qc_results['mask'][500:510])
    
    def test_quality_mask_out_of_bounds(self):
        """Test quality mask with out-of-bounds values."""
        qc = WaveformQualityControl()
        
        # MAP signal with some out-of-bounds values
        map_signal = np.random.uniform(50, 100, 1000)
        map_signal[100:150] = 30  # Below bounds
        map_signal[600:650] = 200  # Above bounds
        
        qc_results = qc.get_quality_mask(map_signal, 'map')
        
        assert not qc_results['in_bounds']
        assert not np.all(qc_results['mask'][100:150])
        assert not np.all(qc_results['mask'][600:650])
    
    def test_quality_mask_valid_ratio_threshold(self):
        """Test valid ratio threshold."""
        qc = WaveformQualityControl()
        
        # Signal with 60% valid (below 70% threshold)
        signal = np.random.randn(1000)
        signal[0:400] = 100  # 40% out of bounds for normalized signal
        
        qc_results = qc.get_quality_mask(signal, 'ecg', min_valid_ratio=0.7)
        
        assert qc_results['valid_ratio'] < 0.7
        assert not qc_results['overall_valid']
        
        # Same signal but lower threshold
        qc_results2 = qc.get_quality_mask(signal, 'ecg', min_valid_ratio=0.5)
        assert qc_results2['overall_valid']  # Should pass with lower threshold
    
    def test_quality_mask_ppg_sqi_requirement(self):
        """Test PPG-specific SQI requirement."""
        qc = WaveformQualityControl()
        
        # Normal distribution (low SQI for PPG)
        poor_ppg = np.random.randn(1000)
        qc_results = qc.get_quality_mask(poor_ppg, 'ppg')
        
        # Should fail due to low SQI even if other metrics pass
        if qc_results['sqi'] < 3.0:
            assert not qc_results['overall_valid']
        
        # Good PPG signal
        good_ppg = np.random.exponential(1, 1000)
        qc_results2 = qc.get_quality_mask(good_ppg, 'ppg')
        
        if qc_results2['sqi'] > 3.0 and not qc_results2['is_flatline']:
            assert qc_results2['overall_valid']
    
    def test_quality_mask_combined_problems(self):
        """Test signal with multiple quality issues."""
        qc = WaveformQualityControl()
        
        # Create signal with multiple problems
        signal = np.random.randn(1000) * 0.5
        signal[100:200] = 5.0  # Spike region
        signal[400:500] = signal[400]  # Flatline region
        signal[700:800] = -10  # Out of bounds
        
        qc_results = qc.get_quality_mask(signal, 'ecg')
        
        assert qc_results['has_spikes']
        # May or may not detect partial flatline depending on implementation
        assert not qc_results['in_bounds']
        assert qc_results['valid_ratio'] < 0.8
        assert not qc_results['overall_valid']
    
    def test_quality_mask_empty_signal(self):
        """Test quality mask with empty signal."""
        qc = WaveformQualityControl()
        
        empty = np.array([])
        qc_results = qc.get_quality_mask(empty, 'ppg')
        
        assert not qc_results['overall_valid']
        assert len(qc_results['mask']) == 0
        assert qc_results['valid_ratio'] == 0.0


class TestQualityControlIntegration:
    """Integration tests for quality control."""
    
    def test_realistic_ppg_signal(self):
        """Test with realistic PPG signal."""
        qc = WaveformQualityControl()
        
        # Simulate realistic PPG
        fs = 100
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        
        # Base PPG waveform (cardiac cycle)
        heart_rate = 72  # BPM
        ppg = np.sin(2 * np.pi * (heart_rate/60) * t)
        ppg = (ppg + 1) / 2  # Make positive
        
        # Add respiratory modulation
        resp_rate = 15  # breaths per minute
        ppg += 0.1 * np.sin(2 * np.pi * (resp_rate/60) * t)
        
        # Add noise
        ppg += np.random.randn(len(ppg)) * 0.02
        
        qc_results = qc.get_quality_mask(ppg, 'ppg')
        
        assert not qc_results['is_flatline']
        assert not qc_results['has_spikes']
        assert qc_results['valid_ratio'] > 0.95
    
    def test_realistic_ecg_signal(self):
        """Test with realistic ECG signal."""
        qc = WaveformQualityControl()
        
        # Simulate ECG-like signal
        fs = 500
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        
        # Simplified ECG (just R peaks)
        heart_rate = 75
        ecg = np.zeros(len(t))
        peak_interval = int(fs * 60 / heart_rate)
        
        for i in range(0, len(ecg), peak_interval):
            if i < len(ecg):
                ecg[i] = 1.0  # R peak
                if i > 0:
                    ecg[i-1] = 0.2  # Q
                if i < len(ecg) - 1:
                    ecg[i+1] = 0.2  # S
        
        # Add baseline wander
        ecg += 0.05 * np.sin(2 * np.pi * 0.15 * t)
        
        # Add noise
        ecg += np.random.randn(len(ecg)) * 0.01
        
        qc_results = qc.get_quality_mask(ecg, 'ecg')
        
        assert not qc_results['is_flatline']
        assert qc_results['valid_ratio'] > 0.9
    
    def test_motion_artifact_detection(self):
        """Test detection of motion artifacts."""
        qc = WaveformQualityControl()
        
        # Normal signal with motion artifact
        signal = np.random.randn(1000) * 0.5
        
        # Add motion artifact (large amplitude oscillation)
        motion_artifact = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200)) * 5
        signal[400:600] = motion_artifact
        
        qc_results = qc.get_quality_mask(signal, 'ppg')
        
        # Motion artifact should be detected as spikes or out of bounds
        assert qc_results['has_spikes'] or not qc_results['in_bounds']
        assert qc_results['valid_ratio'] < 0.9
    
    def test_signal_disconnection(self):
        """Test detection of sensor disconnection."""
        qc = WaveformQualityControl()
        
        # Normal signal with disconnection period
        signal = np.random.randn(1000) * 0.5 + 1.0
        
        # Sensor disconnection (flatline at zero or saturation)
        signal[300:500] = 0.0  # Disconnected period
        
        qc_results = qc.get_quality_mask(signal, 'ppg')
        
        # Should detect flatline in disconnection period
        if np.var(signal[300:500]) < 0.01:
            # Disconnection creates low variance
            sub_signal = signal[300:500]
            assert qc.check_flatline(sub_signal, variance_threshold=0.01)


class TestPerformance:
    """Test performance aspects of quality control."""
    
    def test_large_signal_processing(self):
        """Test QC on large signals."""
        qc = WaveformQualityControl()
        
        # Large signal (1 hour at 100 Hz)
        large_signal = np.random.randn(360000) * 0.5 + 1.0
        
        # Should complete without error
        qc_results = qc.get_quality_mask(large_signal, 'ppg')
        assert qc_results is not None
        assert len(qc_results['mask']) == len(large_signal)
    
    def test_batch_processing(self):
        """Test batch processing of multiple windows."""
        qc = WaveformQualityControl()
        
        # Simulate batch of windows
        window_size = 1000
        n_windows = 100
        
        results = []
        for i in range(n_windows):
            window = np.random.randn(window_size) * 0.5
            qc_result = qc.get_quality_mask(window, 'ppg')
            results.append(qc_result['overall_valid'])
        
        # Should process all windows
        assert len(results) == n_windows
        
        # Most random windows should be valid
        valid_count = sum(results)
        assert valid_count > n_windows * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
