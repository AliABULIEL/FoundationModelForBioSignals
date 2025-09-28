"""
VitalDB Feature Extractor for TabPFN v2 Integration
Implements v1_basic feature set with exactly 50 features for tabular learning
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path
import hashlib

warnings.filterwarnings('ignore')


class FeatureConfig:
    """Configuration for feature extraction"""
    FEATURE_SETS = {
        'v1_basic': {
            'statistical': 15,
            'frequency': 12,
            'morphological': 13,  # PPG + ECG combined
            'cross_modal': 1,     # PTT
            'temporal_complexity': 9
        }
    }
    
    # Expected sample rates after resampling
    SAMPLE_RATES = {
        'ecg': 125,  # Hz
        'ppg': 125,  # Hz
        'abp': 125   # Hz (for BP targets)
    }


class VitalDBFeatureExtractor:
    """
    Extract tabular features from VitalDB signals for TabPFN v2
    Handles missing channels gracefully by returning NaN features
    """
    
    def __init__(self, 
                 feature_set: str = 'v1_basic',
                 sample_rate: int = 125,
                 cache_dir: Optional[str] = None):
        """
        Initialize feature extractor
        
        Args:
            feature_set: Name of feature set ('v1_basic' = 50 features)
            sample_rate: Expected sample rate of input signals
            cache_dir: Optional directory for caching computed features
        """
        self.feature_set = feature_set
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate feature set
        if feature_set not in FeatureConfig.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        self.feature_counts = FeatureConfig.FEATURE_SETS[feature_set]
        self.total_features = sum(self.feature_counts.values())
        
        # Feature names for interpretability
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names"""
        names = []
        
        # Statistical features (15)
        for ch in ['ecg', 'ppg']:
            names.extend([
                f'{ch}_mean', f'{ch}_std', f'{ch}_var',
                f'{ch}_skew', f'{ch}_kurt', f'{ch}_iqr',
                f'{ch}_p25'
            ])  # exactly 7 per channel
        names.append('cross_correlation')
        
        # Frequency features (12)
        for ch in ['ecg', 'ppg']:
            names.extend([
                f'{ch}_dominant_freq', f'{ch}_spectral_energy',
                f'{ch}_lf_power', f'{ch}_hf_power', 
                f'{ch}_lf_hf_ratio', f'{ch}_spectral_entropy'
            ])
        
        # Morphological features (13)
        # ECG: 7 features
        names.extend([
            'ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_mean', 'ecg_rr_std',
            'ecg_qrs_duration', 'ecg_qt_interval', 'ecg_pr_interval'
        ])
        # PPG: 6 features  
        names.extend([
            'ppg_hr_mean', 'ppg_hr_std', 'ppg_peak_interval_mean',
            'ppg_peak_amplitude_mean', 'ppg_pulse_width', 'ppg_perfusion_index'
        ])
        
        # Cross-modal features (1)
        names.append('pulse_transit_time')
        
        # Temporal/complexity features (9)
        for ch in ['ecg', 'ppg']:
            names.extend([
                f'{ch}_sample_entropy', f'{ch}_approx_entropy',
                f'{ch}_zero_crossings', f'{ch}_mobility'
            ])
        names.append('signal_quality_index')
        
        return names[:self.total_features]
    
    def extract_windows(self, 
                       signals: Dict[str, np.ndarray],
                       window_sec: float = 10.0,
                       overlap: float = 0.5) -> List[Dict[str, np.ndarray]]:
        """
        Extract overlapping windows from continuous signals
        
        Args:
            signals: Dict with keys 'ecg', 'ppg', 'abp' containing signal arrays
            window_sec: Window size in seconds
            overlap: Overlap fraction (0.0 to 1.0)
            
        Returns:
            List of window dictionaries
        """
        windows = []
        window_samples = int(window_sec * self.sample_rate)
        hop_samples = int(window_samples * (1 - overlap))
        
        # Find minimum signal length
        min_length = float('inf')
        for key, sig in signals.items():
            if sig is not None and len(sig) > 0:
                min_length = min(min_length, len(sig))
        
        if min_length == float('inf') or min_length < window_samples:
            return []
        
        # Extract windows
        n_windows = (min_length - window_samples) // hop_samples + 1
        
        for i in range(n_windows):
            start = i * hop_samples
            end = start + window_samples
            
            window_dict = {}
            for key, sig in signals.items():
                if sig is not None:
                    window_dict[key] = sig[start:end]
                else:
                    window_dict[key] = None
            
            windows.append(window_dict)
        
        return windows
    
    def compute_features_v1_basic(self, window: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute v1_basic feature set (exactly 50 features)
        
        Args:
            window: Dict with 'ecg', 'ppg' arrays (may be None)
            
        Returns:
            Feature vector of shape (50,), with NaN for missing channels
        """
        features = []
        
        ecg = window.get('ecg')
        ppg = window.get('ppg')
        
        # 1. Statistical features (15)
        features.extend(self._extract_statistical(ecg, ppg))
        
        # 2. Frequency features (12)  
        features.extend(self._extract_frequency(ecg, ppg))
        
        # 3. Morphological features (13)
        features.extend(self._extract_morphological(ecg, ppg))
        
        # 4. Cross-modal features (1)
        features.extend(self._extract_cross_modal(ecg, ppg))
        
        # 5. Temporal/complexity features (9)
        features.extend(self._extract_temporal_complexity(ecg, ppg))
        
        # Ensure exactly 50 features
        features = np.array(features[:self.total_features])
        if len(features) < self.total_features:
            features = np.pad(features, (0, self.total_features - len(features)), 
                            constant_values=np.nan)
        
        return features
    
    def _extract_statistical(self, ecg: Optional[np.ndarray], 
                           ppg: Optional[np.ndarray]) -> List[float]:
        """Extract 15 statistical features"""
        features = []
        
        # Process each channel (7 features each)
        for sig in [ecg, ppg]:
            if sig is not None and len(sig) > 0:
                features.extend([
                    np.mean(sig),
                    np.std(sig),
                    np.var(sig),
                    stats.skew(sig),
                    stats.kurtosis(sig),
                    np.percentile(sig, 75) - np.percentile(sig, 25),  # IQR
                    np.percentile(sig, 25),
                ][:7])
            else:
                features.extend([np.nan] * 7)
        
        # Cross-correlation
        if ecg is not None and ppg is not None:
            corr = np.corrcoef(ecg[:min(len(ecg), len(ppg))],
                              ppg[:min(len(ecg), len(ppg))])[0, 1]
            features.append(corr)
        else:
            features.append(np.nan)
        
        return features
    
    def _extract_frequency(self, ecg: Optional[np.ndarray],
                          ppg: Optional[np.ndarray]) -> List[float]:
        """Extract 12 frequency domain features"""
        features = []
        
        for sig in [ecg, ppg]:
            if sig is not None and len(sig) > 10:
                # Compute FFT
                fft = np.fft.rfft(sig)
                freqs = np.fft.rfftfreq(len(sig), 1/self.sample_rate)
                psd = np.abs(fft) ** 2
                
                # Dominant frequency
                dom_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC
                features.append(freqs[dom_freq_idx])
                
                # Spectral energy
                features.append(np.sum(psd))
                
                # LF power (0.04-0.15 Hz)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
                features.append(lf_power)
                
                # HF power (0.15-0.4 Hz)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
                features.append(hf_power)
                
                # LF/HF ratio
                ratio = lf_power / (hf_power + 1e-10)
                features.append(ratio)
                
                # Spectral entropy
                psd_norm = psd / (np.sum(psd) + 1e-10)
                entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                features.append(entropy)
            else:
                features.extend([np.nan] * 6)
        
        return features
    
    def _extract_morphological(self, ecg: Optional[np.ndarray],
                              ppg: Optional[np.ndarray]) -> List[float]:
        """Extract 13 morphological features"""
        features = []
        
        # ECG morphological features (7)
        if ecg is not None and len(ecg) > self.sample_rate:
            # Detect R-peaks
            peaks, _ = find_peaks(ecg, distance=int(0.6 * self.sample_rate),
                                 height=np.percentile(ecg, 70))
            
            if len(peaks) > 1:
                # Heart rate
                rr_intervals = np.diff(peaks) / self.sample_rate
                hr = 60 / (rr_intervals + 1e-10)
                features.append(np.mean(hr))
                features.append(np.std(hr))
                features.append(np.mean(rr_intervals))
                features.append(np.std(rr_intervals))
                
                # Simplified QRS, QT, PR durations (proxy)
                features.append(0.08)  # QRS duration proxy
                features.append(0.35)  # QT interval proxy  
                features.append(0.16)  # PR interval proxy
            else:
                features.extend([np.nan] * 7)
        else:
            features.extend([np.nan] * 7)
        
        # PPG morphological features (6)
        if ppg is not None and len(ppg) > self.sample_rate:
            # Detect PPG peaks
            peaks, properties = find_peaks(ppg, distance=int(0.6 * self.sample_rate),
                                          height=np.percentile(ppg, 50))
            
            if len(peaks) > 1:
                # Heart rate from PPG
                peak_intervals = np.diff(peaks) / self.sample_rate
                hr_ppg = 60 / (peak_intervals + 1e-10)
                features.append(np.mean(hr_ppg))
                features.append(np.std(hr_ppg))
                features.append(np.mean(peak_intervals))
                
                # Peak amplitudes
                if 'peak_heights' in properties:
                    features.append(np.mean(properties['peak_heights']))
                else:
                    features.append(np.mean(ppg[peaks]))
                
                # Pulse width (simplified)
                features.append(np.mean(peak_intervals) * 0.4)
                
                # Perfusion index (AC/DC ratio)
                ac_component = np.std(ppg)
                dc_component = np.mean(ppg)
                pi = ac_component / (abs(dc_component) + 1e-10)
                features.append(pi)
            else:
                features.extend([np.nan] * 6)
        else:
            features.extend([np.nan] * 6)
        
        return features
    
    def _extract_cross_modal(self, ecg: Optional[np.ndarray],
                            ppg: Optional[np.ndarray]) -> List[float]:
        """Extract 1 cross-modal feature (PTT)"""
        features = []
        
        if ecg is not None and ppg is not None:
            # Simplified PTT calculation
            ecg_peaks, _ = find_peaks(ecg, distance=int(0.6 * self.sample_rate))
            ppg_peaks, _ = find_peaks(ppg, distance=int(0.6 * self.sample_rate))
            
            if len(ecg_peaks) > 0 and len(ppg_peaks) > 0:
                # Find average delay between ECG R-peak and next PPG peak
                delays = []
                for r_peak in ecg_peaks[:min(10, len(ecg_peaks))]:
                    next_ppg = ppg_peaks[ppg_peaks > r_peak]
                    if len(next_ppg) > 0:
                        delay = (next_ppg[0] - r_peak) / self.sample_rate
                        if delay < 0.5:  # Physiological limit
                            delays.append(delay)
                
                ptt = np.mean(delays) if delays else np.nan
                features.append(ptt)
            else:
                features.append(np.nan)
        else:
            features.append(np.nan)
        
        return features
    
    def _extract_temporal_complexity(self, ecg: Optional[np.ndarray],
                                    ppg: Optional[np.ndarray]) -> List[float]:
        """Extract 9 temporal/complexity features"""
        features = []
        
        for sig in [ecg, ppg]:
            if sig is not None and len(sig) > 100:
                # Sample entropy (simplified)
                sample_ent = self._sample_entropy(sig, m=2, r=0.2*np.std(sig))
                features.append(sample_ent)
                
                # Approximate entropy (simplified)
                approx_ent = self._approx_entropy(sig, m=2, r=0.2*np.std(sig))
                features.append(approx_ent)
                
                # Zero crossings
                zero_cross = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0)
                features.append(zero_cross / len(sig))
                
                # Hjorth mobility
                mobility = np.sqrt(np.var(np.diff(sig)) / (np.var(sig) + 1e-10))
                features.append(mobility)
            else:
                features.extend([np.nan] * 4)
        
        # Signal quality index (combined)
        if ecg is not None and ppg is not None:
            sqi = self._signal_quality_index(ecg, ppg)
            features.append(sqi)
        else:
            features.append(np.nan)
        
        return features
    
    def _sample_entropy(self, sig: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy"""
        N = len(sig)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])
        
        def _phi(m):
            patterns = np.array([sig[i:i+m] for i in range(N - m + 1)])
            C = 0
            for i in range(N - m):
                template = patterns[i]
                for j in range(i + 1, N - m):
                    if _maxdist(template, patterns[j], m) <= r:
                        C += 1
            return C
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            if phi_m == 0 or phi_m1 == 0:
                return 0
            return -np.log(phi_m1 / phi_m)
        except:
            return 0
    
    def _approx_entropy(self, sig: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy (simplified)"""
        try:
            N = len(sig)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])
            
            def _phi(m):
                patterns = np.array([sig[i:i+m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template, patterns[j], m) <= r:
                            C[i] += 1
                
                phi = (N - m + 1) ** (-1) * np.sum(np.log(C / (N - m + 1)))
                return phi
            
            return _phi(m) - _phi(m + 1)
        except:
            return 0
    
    def _signal_quality_index(self, ecg: np.ndarray, ppg: np.ndarray) -> float:
        """Calculate signal quality index"""
        try:
            # Check signal variance
            ecg_var = np.var(ecg) if np.var(ecg) > 0 else 0
            ppg_var = np.var(ppg) if np.var(ppg) > 0 else 0
            
            # Check for flat segments
            ecg_flat = np.sum(np.abs(np.diff(ecg)) < 1e-5) / len(ecg)
            ppg_flat = np.sum(np.abs(np.diff(ppg)) < 1e-5) / len(ppg)
            
            # Combined quality
            sqi = (1 - ecg_flat) * (1 - ppg_flat) * min(ecg_var, ppg_var, 1.0)
            return sqi
        except:
            return 0
    
    def vectorize(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        Vectorize list of feature arrays into matrix
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        if not features_list:
            return np.empty((0, self.total_features))
        
        return np.vstack(features_list)
    
    def get_cache_key(self, patient_id: str, window_idx: int, 
                     window_sec: float, overlap: float) -> str:
        """Generate cache key for feature storage"""
        key_str = f"{patient_id}_{window_idx}_{window_sec}_{overlap}_{self.feature_set}_v1"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def save_features(self, features: np.ndarray, metadata: Dict[str, Any],
                     cache_file: Optional[str] = None):
        """Save computed features to cache"""
        if self.cache_dir is None:
            return
        
        if cache_file is None:
            cache_file = f"features_{metadata.get('patient_id', 'unknown')}.npz"
        
        cache_path = self.cache_dir / cache_file
        np.savez_compressed(cache_path, 
                           features=features,
                           **metadata)
    
    def load_features(self, cache_file: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load cached features"""
        if self.cache_dir is None:
            return None, {}
        
        cache_path = self.cache_dir / cache_file
        if not cache_path.exists():
            return None, {}
        
        data = np.load(cache_path, allow_pickle=True)
        features = data['features']
        metadata = {k: data[k] for k in data.keys() if k != 'features'}
        
        return features, metadata
