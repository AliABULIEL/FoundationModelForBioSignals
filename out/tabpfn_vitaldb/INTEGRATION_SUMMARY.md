# TabPFN v2 Integration Summary

## Overview
Complete integration of TabPFN v2 for tabular biosignal learning with VitalDB dataset.

## Files Created/Modified

### New Files
1. **data/features_vitaldb.py** (519 lines)
   - `VitalDBFeatureExtractor` class
   - `extract_windows()`: Sliding window extraction
   - `compute_features_v1_basic()`: 50-feature extraction
   - `vectorize()`: Convert to feature matrix
   - Optional .npz caching with patient/window keys

2. **data/labels_vitaldb.py** (310 lines)
   - `VitalDBLabelCreator` class
   - `create_ioh_labels()`: Binary IOH at 5/10/15 min horizons
   - `create_bp_targets()`: SBP/DBP/MAP regression targets
   - MAP < 65 mmHg for ≥ 60s criterion

3. **models/tabpfn_foundation.py** (383 lines)
   - `TabPFNFoundation` wrapper class
   - `fit_context()`: In-context learning interface
   - `predict_proba()`: Classification predictions
   - `predict()`: Regression predictions
   - Handles NaN features, respects 10K sample / 500 feature limits

4. **configs/tabpfn_vitaldb.yaml** (73 lines)
   - Default configuration for tabular mode
   - IOH @ 5min horizon default
   - Context size: 1000, Query size: 500
   - Class balancing and patient-aware splits

5. **scripts/run_tabpfn_ioh.py** (241 lines)
   - Complete IOH prediction pipeline
   - Patient-aware data splitting
   - Class-balanced context selection
   - No execution/printing (as requested)

6. **scripts/run_tabpfn_bp.py** (125 lines)
   - BP regression skeleton
   - Multi-target support (SBP/DBP/MAP)
   - Placeholder implementation

### Modified Files
1. **data.py** (minimal changes)
   - Added `mode` parameter ('timeseries'|'tabular')
   - Added tabular-specific parameters
   - New `_getitem_tabular()` method
   - Returns (features, label, context) in tabular mode
   - Backward compatible - timeseries mode unchanged

2. **requirements.txt** (2 lines added)
   - `tabpfn>=1.0.0` for TabPFN v2
   - Optional `tabpfn_extensions` commented

## Feature Set: v1_basic (50 features)

### Statistical (15 features)
- ECG: mean, std, var, skew, kurt, IQR, p25 (7)
- PPG: mean, std, var, skew, kurt, IQR, p25 (7)
- Cross-correlation (1)

### Frequency (12 features)
- ECG: dominant_freq, spectral_energy, LF, HF, LF/HF, entropy (6)
- PPG: dominant_freq, spectral_energy, LF, HF, LF/HF, entropy (6)

### Morphological (13 features)
- ECG: HR_mean, HR_std, RR_mean, RR_std, QRS, QT, PR (7)
- PPG: HR_mean, HR_std, peak_interval, amplitude, width, PI (6)

### Cross-modal (1 feature)
- Pulse Transit Time (PTT)

### Temporal/Complexity (9 features)
- ECG: sample_entropy, approx_entropy, zero_crossings, mobility (4)
- PPG: sample_entropy, approx_entropy, zero_crossings, mobility (4)
- Signal Quality Index (1)

## Input/Output Specifications

### Dataset __getitem__ (tabular mode)
```python
# Input: index
# Output: (features, label, context)
features: torch.FloatTensor[50]  # 50 features, NaN allowed
label: torch.FloatTensor[1]      # Binary (IOH) or continuous (BP)
context: {
    'patient_id': int,
    'window_idx': int,
    'split': str
}
```

### TabPFNFoundation Interface
```python
# Context fitting
model.fit_context(X_context, y_context)
# X_context: np.ndarray[n_context, 50]
# y_context: np.ndarray[n_context]

# Prediction
y_pred = model.predict(X_query)
# X_query: np.ndarray[n_query, 50]
# Returns: np.ndarray[n_query]
```

## Configuration Keys

### Core Settings
- `dataset.mode`: 'tabular' (vs 'timeseries')
- `dataset.feature_set`: 'v1_basic'
- `dataset.window_sec`: 10.0
- `dataset.overlap`: 0.5

### Task Settings
- `task.target`: 'ioh' | 'bp'
- `task.horizon_min`: 5
- `task.ioh.map_threshold`: 65
- `task.ioh.min_duration`: 60

### Model Settings
- `model.N_ensemble_configurations`: 4
- `model.max_features`: 500
- `training.context_size`: 1000
- `training.class_balance`: true

## Patient-Level Split Enforcement
- Context and query sets use **different patients**
- No patient appears in both sets (no leakage)
- Implemented via patient ID tracking in context dict
- ~70% patients → context, ~30% → query

## Missing Channel Handling
- Missing PPG/ECG → NaN features (no imputation)
- TabPFN v2 handles NaN natively
- No artificial zero filling

## TabPFN v2 Constraints Respected
- Max ~10,000 samples per run
- Max 500 features (v1_basic = 50 ✓)
- NaN features supported
- No internal model modifications