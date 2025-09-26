"""
VitalDB Label Creator for TabPFN v2 Integration
Creates IOH labels and BP regression targets from VitalDB data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class LabelConfig:
    """Configuration for label creation"""
    
    # Intraoperative Hypotension (IOH) thresholds
    IOH_MAP_THRESHOLD = 65  # mmHg
    IOH_MIN_DURATION = 60   # seconds
    IOH_HORIZONS = [5, 10, 15]  # minutes
    
    # Blood Pressure regression targets
    BP_TARGETS = ['SBP', 'DBP', 'MAP']
    BP_DEFAULT_HORIZON = 5  # minutes
    
    # Sampling rate for ABP signal
    ABP_SAMPLE_RATE = 125  # Hz


class VitalDBLabelCreator:
    """
    Create labels for supervised learning from VitalDB signals
    Supports both classification (IOH) and regression (BP) tasks
    """
    
    def __init__(self, sample_rate: int = 125):
        """
        Initialize label creator
        
        Args:
            sample_rate: Sample rate of ABP signal
        """
        self.sample_rate = sample_rate
    
    def create_ioh_labels(self,
                         abp: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         window_start_idx: int = 0,
                         window_sec: float = 10.0,
                         horizons: List[int] = None,
                         map_thresh: float = 65.0,
                         min_duration: float = 60.0) -> Dict[str, Any]:
        """
        Create IOH labels for different prediction horizons
        
        Args:
            abp: Arterial blood pressure signal
            timestamps: Optional timestamps for each sample
            window_start_idx: Start index of current window
            window_sec: Window size in seconds
            horizons: List of prediction horizons in minutes (default: [5, 10, 15])
            map_thresh: MAP threshold for IOH in mmHg (default: 65)
            min_duration: Minimum duration for IOH event in seconds (default: 60)
            
        Returns:
            Dict with keys 'ioh_5min', 'ioh_10min', 'ioh_15min' (binary labels)
            and metadata about the IOH events
        """
        if horizons is None:
            horizons = LabelConfig.IOH_HORIZONS
        
        labels = {}
        metadata = {
            'map_threshold': map_thresh,
            'min_duration': min_duration,
            'window_start_idx': window_start_idx,
            'window_sec': window_sec
        }
        
        if abp is None or len(abp) == 0:
            # Return NaN labels if no ABP signal
            for h in horizons:
                labels[f'ioh_{h}min'] = np.nan
            metadata['has_abp'] = False
            return {'labels': labels, 'metadata': metadata}
        
        # Calculate MAP from ABP waveform
        map_signal = self._calculate_map(abp)
        
        # Check IOH at different horizons
        for horizon_min in horizons:
            horizon_samples = int(horizon_min * 60 * self.sample_rate)
            
            # Get future window
            future_start = window_start_idx + int(window_sec * self.sample_rate)
            future_end = future_start + horizon_samples
            
            if future_end <= len(map_signal):
                future_map = map_signal[future_start:future_end]
                
                # Check for IOH event (MAP < threshold for >= min_duration)
                ioh_label = self._detect_ioh_event(future_map, map_thresh, min_duration)
                labels[f'ioh_{horizon_min}min'] = int(ioh_label)
                
                # Store additional metadata
                if f'ioh_events' not in metadata:
                    metadata['ioh_events'] = {}
                metadata['ioh_events'][f'{horizon_min}min'] = {
                    'has_ioh': bool(ioh_label),
                    'min_map': float(np.min(future_map)) if len(future_map) > 0 else np.nan,
                    'mean_map': float(np.mean(future_map)) if len(future_map) > 0 else np.nan,
                    'duration_below_threshold': self._get_duration_below_threshold(
                        future_map, map_thresh
                    )
                }
            else:
                # Not enough future data
                labels[f'ioh_{horizon_min}min'] = np.nan
        
        metadata['has_abp'] = True
        return {'labels': labels, 'metadata': metadata}
    
    def create_bp_targets(self,
                         abp: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         window_start_idx: int = 0,
                         window_sec: float = 10.0,
                         horizon_min: float = 5.0,
                         targets: List[str] = None) -> Dict[str, Any]:
        """
        Create blood pressure regression targets
        
        Args:
            abp: Arterial blood pressure signal
            timestamps: Optional timestamps
            window_start_idx: Start index of current window
            window_sec: Window size in seconds
            horizon_min: Prediction horizon in minutes (default: 5)
            targets: List of targets ['SBP', 'DBP', 'MAP'] (default: all)
            
        Returns:
            Dict with BP values (SBP, DBP, MAP) at future horizon
        """
        if targets is None:
            targets = LabelConfig.BP_TARGETS
        
        bp_targets = {}
        metadata = {
            'horizon_min': horizon_min,
            'window_start_idx': window_start_idx,
            'window_sec': window_sec
        }
        
        if abp is None or len(abp) == 0:
            # Return NaN targets if no ABP signal
            for target in targets:
                bp_targets[f'{target.lower()}_{horizon_min}min'] = np.nan
            metadata['has_abp'] = False
            return {'targets': bp_targets, 'metadata': metadata}
        
        # Calculate future BP values
        horizon_samples = int(horizon_min * 60 * self.sample_rate)
        future_start = window_start_idx + int(window_sec * self.sample_rate)
        future_end = future_start + horizon_samples
        
        if future_end <= len(abp):
            future_abp = abp[future_start:future_end]
            
            # Extract BP components
            if 'SBP' in targets:
                sbp = self._calculate_sbp(future_abp)
                bp_targets[f'sbp_{horizon_min}min'] = float(np.mean(sbp))
            
            if 'DBP' in targets:
                dbp = self._calculate_dbp(future_abp)
                bp_targets[f'dbp_{horizon_min}min'] = float(np.mean(dbp))
            
            if 'MAP' in targets:
                map_val = self._calculate_map(future_abp)
                bp_targets[f'map_{horizon_min}min'] = float(np.mean(map_val))
            
            metadata['has_abp'] = True
            metadata['future_samples'] = len(future_abp)
        else:
            # Not enough future data
            for target in targets:
                bp_targets[f'{target.lower()}_{horizon_min}min'] = np.nan
            metadata['has_abp'] = False
            metadata['insufficient_future_data'] = True
        
        return {'targets': bp_targets, 'metadata': metadata}
    
    def _calculate_map(self, abp: np.ndarray) -> np.ndarray:
        """
        Calculate Mean Arterial Pressure from ABP waveform
        MAP = DBP + 1/3(SBP - DBP)
        """
        if len(abp) == 0:
            return np.array([])
        
        # Use moving window to calculate MAP
        window_size = int(self.sample_rate * 2)  # 2-second windows
        if len(abp) < window_size:
            # For short signals, use entire signal
            sbp = np.max(abp)
            dbp = np.min(abp)
            map_val = dbp + (sbp - dbp) / 3
            return np.full(len(abp), map_val)
        
        map_signal = np.zeros(len(abp))
        for i in range(len(abp)):
            start = max(0, i - window_size // 2)
            end = min(len(abp), i + window_size // 2)
            segment = abp[start:end]
            
            if len(segment) > 0:
                sbp = np.percentile(segment, 90)  # Use percentiles for robustness
                dbp = np.percentile(segment, 10)
                map_signal[i] = dbp + (sbp - dbp) / 3
        
        return map_signal
    
    def _calculate_sbp(self, abp: np.ndarray) -> np.ndarray:
        """Calculate Systolic Blood Pressure (peaks)"""
        if len(abp) == 0:
            return np.array([])
        
        # Use moving window to find local maxima
        window_size = int(self.sample_rate * 1)  # 1-second windows
        sbp_signal = np.zeros(len(abp) // window_size)
        
        for i in range(len(sbp_signal)):
            start = i * window_size
            end = min(start + window_size, len(abp))
            segment = abp[start:end]
            sbp_signal[i] = np.percentile(segment, 90) if len(segment) > 0 else np.nan
        
        return sbp_signal
    
    def _calculate_dbp(self, abp: np.ndarray) -> np.ndarray:
        """Calculate Diastolic Blood Pressure (troughs)"""
        if len(abp) == 0:
            return np.array([])
        
        # Use moving window to find local minima
        window_size = int(self.sample_rate * 1)  # 1-second windows
        dbp_signal = np.zeros(len(abp) // window_size)
        
        for i in range(len(dbp_signal)):
            start = i * window_size
            end = min(start + window_size, len(abp))
            segment = abp[start:end]
            dbp_signal[i] = np.percentile(segment, 10) if len(segment) > 0 else np.nan
        
        return dbp_signal
    
    def _detect_ioh_event(self, map_signal: np.ndarray, 
                         threshold: float, min_duration_sec: float) -> bool:
        """
        Detect if IOH event occurs in the signal
        IOH defined as MAP < threshold for >= min_duration
        """
        if len(map_signal) == 0:
            return False
        
        # Find continuous segments below threshold
        below_threshold = map_signal < threshold
        min_samples = int(min_duration_sec * self.sample_rate)
        
        # Find runs of True values
        if not np.any(below_threshold):
            return False
        
        # Pad with False to handle edge cases
        padded = np.concatenate(([False], below_threshold, [False]))
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        
        # Check if any segment is long enough
        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_samples:
                return True
        
        return False
    
    def _get_duration_below_threshold(self, map_signal: np.ndarray, 
                                     threshold: float) -> float:
        """Get total duration (in seconds) that MAP is below threshold"""
        if len(map_signal) == 0:
            return 0.0
        
        below_threshold = map_signal < threshold
        total_samples = np.sum(below_threshold)
        duration_sec = total_samples / self.sample_rate
        
        return float(duration_sec)
    
    def create_labels_for_windows(self,
                                 abp: np.ndarray,
                                 windows: List[Dict[str, np.ndarray]],
                                 task: str = 'ioh',
                                 **kwargs) -> List[Dict[str, Any]]:
        """
        Create labels for multiple windows
        
        Args:
            abp: Full ABP signal
            windows: List of window dictionaries
            task: 'ioh' or 'bp'
            **kwargs: Additional arguments for label creation
            
        Returns:
            List of label dictionaries
        """
        labels_list = []
        
        for i, window in enumerate(windows):
            window_start_idx = kwargs.get('window_start_indices', [i * self.sample_rate * 10])[i]
            
            if task == 'ioh':
                labels = self.create_ioh_labels(
                    abp=abp,
                    window_start_idx=window_start_idx,
                    **{k: v for k, v in kwargs.items() if k != 'window_start_indices'}
                )
            elif task == 'bp':
                labels = self.create_bp_targets(
                    abp=abp,
                    window_start_idx=window_start_idx,
                    **{k: v for k, v in kwargs.items() if k != 'window_start_indices'}
                )
            else:
                raise ValueError(f"Unknown task: {task}")
            
            labels_list.append(labels)
        
        return labels_list
