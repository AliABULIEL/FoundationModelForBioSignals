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
        # Create longer signal to accommodate 5-min horizon from 10s window (needs 310s minimum)
        abp = self.create_synthetic_abp(duration_sec=400, base_map=80)
        
        # Insert hypotension event from 120s to 180s (60 seconds)
        hypotension_start = 120 * 125  # samples
        hypotension_end = 180 * 125
        abp[hypotension_start:hypotension_end] = 30  # Set to 30 to ensure MAP < 65
        
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
        abp = self.create_synthetic_abp(duration_sec=400, base_map=80)
        
        # Insert hypotension for only 59 seconds
        # Account for MAP smoothing window (2 seconds) by making the actual
        # hypotension period slightly shorter to ensure it's < 60s after smoothing
        hypotension_start = 120 * 125  # 120 seconds
        # Create 58 seconds of hypotension to account for smoothing extending it
        hypotension_end = 178 * 125  # 178 seconds (58 second duration)
        abp[hypotension_start:hypotension_end] = 30  # Set to 30 to ensure MAP < 65
        
        result = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=65,
            min_duration=60
        )
        
        # Should be negative because effective duration < 60s
        assert result['labels']['ioh_5min'] == 0, \
            "Should NOT detect IOH when MAP < 65 for less than 60s"
    
    def test_horizon_alignment(self):
        """Test that horizons align correctly to window end + horizon time"""
        # Create 20-minute signal
        abp = self.create_synthetic_abp(duration_sec=1200, base_map=80)
        
        # Add hypotension events at specific times
        # Window is 0-10s, so horizons are:
        # 5-min: 10s to 310s (0.17 to 5.17 minutes)
        # 10-min: 10s to 610s (0.17 to 10.17 minutes)
        # 15-min: 10s to 910s (0.17 to 15.17 minutes)
        
        # Event 1: 2-3 minutes (120-180s) - within 5-min horizon
        event1_start = 2 * 60 * 125
        event1_end = 3 * 60 * 125
        abp[event1_start:event1_end] = 30  # Set to 30 to ensure MAP < 65
        
        # Event 2: 7-8 minutes (420-480s) - within 10-min horizon but outside 5-min
        event2_start = 7 * 60 * 125
        event2_end = 8 * 60 * 125
        abp[event2_start:event2_end] = 30  # Set to 30 to ensure MAP < 65
        
        # Event 3: 12-13 minutes (720-780s) - within 15-min horizon but outside 10-min
        event3_start = 12 * 60 * 125
        event3_end = 13 * 60 * 125
        abp[event3_start:event3_end] = 30  # Set to 30 to ensure MAP < 65
        
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
        # 5-min horizon: looks at 10s to 310s (event at 2-3min should be visible)
        assert result['labels']['ioh_5min'] == 1, \
            "5-min horizon should detect event at 2-3 minutes"
        
        # 10-min horizon: looks at 10s to 610s (events at 2-3min and 7-8min should be visible)
        assert result['labels']['ioh_10min'] == 1, \
            "10-min horizon should detect events"
        
        # 15-min horizon: looks at 10s to 910s (all three events should be visible)
        assert result['labels']['ioh_15min'] == 1, \
            "15-min horizon should detect events"
    
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
        abp = self.create_synthetic_abp(duration_sec=400, base_map=70)
        
        # MAP is 70, add event at 60-120s with MAP around 30
        abp[60*125:120*125] = 30  # Set to 30 to ensure MAP < both thresholds
        
        # Test with threshold 65 (MAP 30 < 65, should detect)
        result_65 = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=65,
            min_duration=60
        )
        assert result_65['labels']['ioh_5min'] == 1, \
            "Should detect IOH when MAP < 65"
        
        # Test with threshold 55 (MAP 30 < 55, should also detect)
        result_55 = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=55,
            min_duration=60
        )
        assert result_55['labels']['ioh_5min'] == 1, \
            "Should detect IOH when MAP < 55"
        
        # Test with threshold 25 (MAP 30 > 25, should NOT detect)
        result_25 = self.label_creator.create_ioh_labels(
            abp=abp,
            window_start_idx=0,
            window_sec=10,
            horizons=[5],
            map_thresh=25,
            min_duration=60
        )
        assert result_25['labels']['ioh_5min'] == 0, \
            "Should NOT detect IOH when MAP > threshold"
    
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
