"""
Test v1_basic feature extraction (50 features)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features_vitaldb import VitalDBFeatureExtractor


class TestV1BasicFeatures:
    """Test suite for v1_basic feature extraction"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = VitalDBFeatureExtractor(feature_set='v1_basic', sample_rate=125)
        
    def create_synthetic_signal(self, duration_sec=10, fs=125, freq_hz=1.5):
        """Create synthetic PPG/ECG signal"""
        t = np.linspace(0, duration_sec, int(duration_sec * fs))
        # Base signal with heart rate component
        signal = np.sin(2 * np.pi * freq_hz * t)  # ~90 bpm
        # Add some HRV
        signal += 0.1 * np.sin(2 * np.pi * 0.1 * t)
        # Add noise
        signal += 0.05 * np.random.randn(len(t))
        return signal
    
    def test_feature_count_complete(self):
        """Test that exactly 50 features are returned with both signals"""
        # Create synthetic signals
        ppg = self.create_synthetic_signal()
        ecg = self.create_synthetic_signal(freq_hz=1.2)  # Slightly different
        
        window = {'ppg': ppg, 'ecg': ecg}
        features = self.extractor.compute_features_v1_basic(window)
        
        assert len(features) == 50, f"Expected 50 features, got {len(features)}"
        assert features.shape == (50,), f"Expected shape (50,), got {features.shape}"
    
    def test_missing_channel_nan_policy(self):
        """Test NaN policy: missing channel yields NaNs only in its block"""
        ppg = self.create_synthetic_signal()
        
        # Only PPG provided (no ECG)
        window = {'ppg': ppg, 'ecg': None}
        features = self.extractor.compute_features_v1_basic(window)
        
        assert len(features) == 50, "Should still return 50 features"
        
        # Check PPG features are valid (indices depend on order)
        # Statistical: PPG is indices 7-13 (7 features)
        ppg_statistical = features[7:14]
        assert not np.all(np.isnan(ppg_statistical)), "PPG statistical features should be valid"
        
        # ECG features should be NaN (indices 0-6)
        ecg_statistical = features[0:7]
        assert np.all(np.isnan(ecg_statistical)), "ECG statistical features should be NaN when ECG missing"
        
        # Test opposite: ECG only
        window = {'ppg': None, 'ecg': ppg}  # Reuse signal
        features = self.extractor.compute_features_v1_basic(window)
        
        ecg_statistical = features[0:7]
        assert not np.all(np.isnan(ecg_statistical)), "ECG features should be valid when provided"
        
        ppg_statistical = features[7:14]
        assert np.all(np.isnan(ppg_statistical)), "PPG features should be NaN when PPG missing"
    
    def test_psd_powers_non_negative(self):
        """Test that PSD-based features are non-negative"""
        ppg = self.create_synthetic_signal()
        ecg = self.create_synthetic_signal()
        
        window = {'ppg': ppg, 'ecg': ecg}
        features = self.extractor.compute_features_v1_basic(window)
        
        # Frequency features start at index 15 (after 15 statistical)
        # Each channel has: dominant_freq, spectral_energy, lf_power, hf_power, lf_hf_ratio, entropy
        
        # Check spectral energy (indices 16 and 22 for ECG/PPG)
        ecg_spectral_energy = features[16]
        ppg_spectral_energy = features[22]
        
        if not np.isnan(ecg_spectral_energy):
            assert ecg_spectral_energy >= 0, "ECG spectral energy should be non-negative"
        if not np.isnan(ppg_spectral_energy):
            assert ppg_spectral_energy >= 0, "PPG spectral energy should be non-negative"
        
        # Check LF and HF powers
        ecg_lf = features[17]
        ecg_hf = features[18]
        ppg_lf = features[23]
        ppg_hf = features[24]
        
        for power, name in [(ecg_lf, "ECG LF"), (ecg_hf, "ECG HF"), 
                            (ppg_lf, "PPG LF"), (ppg_hf, "PPG HF")]:
            if not np.isnan(power):
                assert power >= 0, f"{name} power should be non-negative"
    
    def test_ratios_bounded(self):
        """Test that ratio features are properly bounded"""
        ppg = self.create_synthetic_signal()
        ecg = self.create_synthetic_signal()
        
        window = {'ppg': ppg, 'ecg': ecg}
        features = self.extractor.compute_features_v1_basic(window)
        
        # LF/HF ratios (indices 19 and 25)
        ecg_lf_hf = features[19]
        ppg_lf_hf = features[25]
        
        for ratio, name in [(ecg_lf_hf, "ECG LF/HF"), (ppg_lf_hf, "PPG LF/HF")]:
            if not np.isnan(ratio):
                assert ratio >= 0, f"{name} ratio should be non-negative"
                # Note: LF/HF can be > 1, so we don't cap at 1
        
        # Perfusion index (part of morphological features)
        # This is at index ~33 (after statistical + frequency + some morphological)
        # Skip detailed index calculation for now, just check general properties
        
        # Cross-correlation (index 14 - last statistical feature)
        cross_corr = features[14]
        if not np.isnan(cross_corr):
            assert -1 <= cross_corr <= 1, "Cross-correlation should be between -1 and 1"
    
    def test_feature_names_match_count(self):
        """Test that feature names list matches feature count"""
        assert len(self.extractor.feature_names) == 50, \
            f"Feature names count {len(self.extractor.feature_names)} != 50"
        
        # Check breakdown
        assert self.extractor.feature_counts['statistical'] == 15
        assert self.extractor.feature_counts['frequency'] == 12
        assert self.extractor.feature_counts['morphological'] == 13
        assert self.extractor.feature_counts['cross_modal'] == 1
        assert self.extractor.feature_counts['temporal_complexity'] == 9
        assert sum(self.extractor.feature_counts.values()) == 50
    
    def test_window_extraction(self):
        """Test window extraction with overlap"""
        # Create 30 second signal
        long_signal = self.create_synthetic_signal(duration_sec=30)
        
        signals = {'ppg': long_signal, 'ecg': long_signal}
        windows = self.extractor.extract_windows(signals, window_sec=10, overlap=0.5)
        
        # With 30s signal, 10s windows, 50% overlap: 
        # Windows at 0-10, 5-15, 10-20, 15-25, 20-30 = 5 windows
        assert len(windows) == 5, f"Expected 5 windows, got {len(windows)}"
        
        # Check each window has correct keys
        for window in windows:
            assert 'ppg' in window
            assert 'ecg' in window
            assert len(window['ppg']) == 10 * 125  # 10 seconds at 125 Hz
    
    def test_nan_propagation(self):
        """Test that NaN inputs don't crash feature extraction"""
        # Signal with some NaNs
        ppg = self.create_synthetic_signal()
        ppg[100:200] = np.nan
        
        window = {'ppg': ppg, 'ecg': None}
        features = self.extractor.compute_features_v1_basic(window)
        
        assert len(features) == 50, "Should handle NaN gracefully"
        # Some features might be NaN but shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
