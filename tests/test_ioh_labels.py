"""
Test IOH label creation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.labels_vitaldb import VitalDBLabelCreator


class TestIOHLabels:
    """Test suite for IOH label creation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.label_creator = VitalDBLabelCreator(sample_rate=125)
        
    def create_synthetic_abp(self, duration_sec=300, fs=125, base_map=80):
        """
        Create synthetic ABP signal with controllable MAP
        
        Args:
            duration_sec: Duration in seconds
            fs: Sample rate
            base_map: Baseline MAP value
        """
        n_samples = int(duration_sec * fs)
        t = np.linspace(0, duration_sec, n_samples)
        
        # Create pulsatile ABP (SBP ~120, DBP ~80 for MAP ~93)
        # MAP = DBP + 1/3(SBP - DBP)
        heart_rate = 1.0  # 60 bpm
        
        # Base pressure
        abp = np.ones(n_samples) * base_map
        
        # Add pulsatile component
        pulse_amplitude = 40  # SBP - DBP
        abp += pulse_amplitude/3 * np.sin(2 * np.pi * heart_rate * t)
        
        # Add some variation
        abp += 5 * np.sin(2 * np.pi * 0.05 * t)  # Slow variation
        
        return abp
    
    def test_ioh_positive_case_60s(self):
        """Test IOH detection with MAP < 65 for exactly 60 seconds"""
        # Create 5-minute signal
        abp = self.create_synthetic_abp(duration_sec=300, base_map=80)
        
        # Insert hypotension event from 120s to 180s (60 seconds)
        hypotension_start = 120 * 125  # samples
        hypotension_end = 180 * 125
        abp[hypotension_start:hypotension_end] = 50  # MAP = 50 < 65
        
        # Test with 10s window at start (0-10s)
        # Check 5-minute horizon (should see event at 120-180s)
        result = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],  # 5 minutes
            map_thresh=65,
            min_duration=60
        )
        
        # Should be positive because hypotension occurs within 5-min horizon
        assert result['labels']['ioh_5min'] == 1, \
            "Should detect IOH when MAP < 65 for 60s within horizon"
    
    def test_ioh_negative_case_59s(self):
        """Test that 59 seconds < threshold does NOT trigger IOH"""
        abp = self.create_synthetic_abp(duration_sec=300, base_map=80)
        
        # Insert hypotension for only 59 seconds
        hypotension_start = 120 * 125
        hypotension_end = 179 * 125  # 59 seconds
        abp[hypotension_start:hypotension_end] = 50
        
        result = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=65,
            min_duration=60
        )
        
        # Should be negative because duration < 60s
        assert result['labels']['ioh_5min'] == 0, \
            "Should NOT detect IOH when MAP < 65 for only 59s"
    
    def test_horizon_alignment(self):
        """Test that horizons align correctly to window end + horizon time"""
        # Create 20-minute signal
        abp = self.create_synthetic_abp(duration_sec=1200, base_map=80)
        
        # Add hypotension events at specific times
        # Event 1: 6-7 minutes (visible in 5-min horizon from 0-10s window)
        event1_start = 6 * 60 * 125
        event1_end = 7 * 60 * 125
        abp[event1_start:event1_end] = 50
        
        # Event 2: 11-12 minutes (visible in 10-min horizon)
        event2_start = 11 * 60 * 125
        event2_end = 12 * 60 * 125
        abp[event2_start:event2_end] = 50
        
        # Event 3: 16-17 minutes (visible in 15-min horizon)
        event3_start = 16 * 60 * 125
        event3_end = 17 * 60 * 125
        abp[event3_start:event3_end] = 50
        
        # Test from 10s window starting at t=0
        result = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5, 10, 15],
            map_thresh=65,
            min_duration=60
        )
        
        # Check each horizon
        # 5-min horizon: looks at 10s to 5min+10s (event at 6-7min should be visible)
        assert result['labels']['ioh_5min'] == 1, \
            "5-min horizon should detect event at 6-7 minutes"
        
        # 10-min horizon: looks at 10s to 10min+10s (event at 11-12min should be visible)
        assert result['labels']['ioh_10min'] == 1, \
            "10-min horizon should detect event at 11-12 minutes"
        
        # 15-min horizon: looks at 10s to 15min+10s (event at 16-17min should be visible)
        assert result['labels']['ioh_15min'] == 1, \
            "15-min horizon should detect event at 16-17 minutes"
    
    def test_no_abp_returns_nan(self):
        """Test that missing ABP signal returns NaN labels"""
        result = self.label_creator.create_ioh_labels(
            abp=None,
            window_start_idx=0,
            window_sec=10,
            horizons=[5, 10, 15],
            map_thresh=65,
            min_duration=60
        )
        
        assert np.isnan(result['labels']['ioh_5min'])
        assert np.isnan(result['labels']['ioh_10min'])
        assert np.isnan(result['labels']['ioh_15min'])
        assert result['metadata']['has_abp'] == False
    
    def test_map_calculation(self):
        """Test MAP calculation from ABP waveform"""
        # Create simple ABP with known SBP/DBP
        abp = np.zeros(1000)
        # Simulate: DBP=60, SBP=120, so MAP=60+1/3(120-60)=80
        abp[::2] = 120  # Peaks (SBP)
        abp[1::2] = 60   # Troughs (DBP)
        
        map_signal = self.label_creator._calculate_map(abp)
        
        # MAP should be around 80
        # Due to windowing, exact value might vary
        assert 75 <= np.mean(map_signal) <= 85, \
            f"MAP should be ~80, got {np.mean(map_signal)}"
    
    def test_multiple_events(self):
        """Test handling of multiple hypotension events"""
        abp = self.create_synthetic_abp(duration_sec=600, base_map=80)
        
        # Add two separate hypotension events
        # Event 1: 60-120s (60s duration)
        abp[60*125:120*125] = 50
        # Event 2: 200-280s (80s duration)
        abp[200*125:280*125] = 45
        
        result = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=65,
            min_duration=60
        )
        
        # Both events are within 5-min horizon, should be positive
        assert result['labels']['ioh_5min'] == 1
        
        # Check metadata
        assert result['metadata']['ioh_events']['5min']['has_ioh'] == True
        assert result['metadata']['ioh_events']['5min']['min_map'] < 65
    
    def test_threshold_sensitivity(self):
        """Test different MAP thresholds"""
        abp = self.create_synthetic_abp(duration_sec=300, base_map=70)
        
        # MAP is 70, add event at 60 for 60s
        abp[60*125:120*125] = 60
        
        # Test with threshold 65 (event at 60 < 65)
        result_65 = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=65,
            min_duration=60
        )
        assert result_65['labels']['ioh_5min'] == 1
        
        # Test with threshold 55 (event at 60 > 55)
        result_55 = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=55,
            min_duration=60
        )
        assert result_55['labels']['ioh_5min'] == 0
    
    def test_bp_regression_targets(self):
        """Test blood pressure regression target creation"""
        # Create ABP with known values
        abp = self.create_synthetic_abp(duration_sec=600, base_map=80)
        
        result = self.label_creator.create_bp_targets(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizon_min=5,
            targets=['SBP', 'DBP', 'MAP']
        )
        
        # Should have all three targets
        assert 'sbp_5min' in result['targets']
        assert 'dbp_5min' in result['targets']
        assert 'map_5min' in result['targets']
        
        # MAP should be around 80
        map_value = result['targets']['map_5min']
        assert 70 <= map_value <= 90, f"MAP should be ~80, got {map_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
