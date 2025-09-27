# VitalDB Data Specification - Waveforms Implementation

## Modality-Specific Processing Parameters

### PPG Processing
- **Filter**: Chebyshev Type-II, 4th order, [0.5, 10] Hz (CONSENSUS)
  - Implementation: `scipy.signal.cheby2(4, 40, [0.5, 10], btype='band', fs=100, output='sos')`
- **Resampling**: 100 Hz → 25 Hz (CONSENSUS) 
  - Implementation: `scipy.signal.resample` or decimation with anti-aliasing
- **Window**: 10 seconds = 250 samples @ 25 Hz (CONSENSUS)
- **Hop**: 5 seconds = 125 samples @ 25 Hz (CONSENSUS)

### ECG Processing  
- **Filter**: Butterworth, 4th order, [0.5, 40] Hz (CONSENSUS)
  - Implementation: `scipy.signal.butter(4, [0.5, 40], btype='band', fs=500, output='sos')`
  - Zero-phase: `scipy.signal.sosfiltfilt(sos, signal)`
- **Resampling**: 500 Hz → 125 Hz (CHOICE: compatibility with MIMIC)
- **Window**: 10 seconds = 1250 samples @ 125 Hz
- **Hop**: 5 seconds = 625 samples @ 125 Hz

### ABP Processing
- **Filter**: Butterworth, 2nd order, [0.5, 10] Hz (CONSENSUS)
  - Implementation: `scipy.signal.butter(2, [0.5, 10], btype='band', fs=100, output='sos')`
- **Resampling**: Keep at 100 Hz (CONSENSUS)
- **Window**: 10 seconds = 1000 samples @ 100 Hz
- **Hop**: 5 seconds = 500 samples @ 100 Hz

### EEG Processing (if used)
- **Filter**: Wavelet denoising, db16, 6-level (CONSENSUS)
  - Implementation: `pywt.wavedec(signal, 'db16', level=6)`
- **Resampling**: Keep at 128 Hz (CONSENSUS)
- **Window**: 56 seconds = 7168 samples @ 128 Hz (CHOICE: DoA monitoring)
- **Hop**: 1 second = 128 samples @ 128 Hz

## Quality Control Rules

### Flatline Detection
- **Threshold**: variance < 0.01 (CONSENSUS)
- **Alternative**: >50 consecutive identical values (CONSENSUS)
- **Implementation**:
```python
is_flatline = (np.var(window) < 0.01) or (np.diff(window) == 0).sum() > 50
```

### Spike Detection
- **Z-score threshold**: |z| > 3.0 (CONSENSUS)
- **Consecutive requirement**: ≥5 samples (CONSENSUS)
- **Implementation**:
```python
z_scores = np.abs((signal - np.mean(signal)) / np.std(signal))
spike_mask = z_scores > 3.0
is_spike = np.convolve(spike_mask, np.ones(5), 'same') >= 5
```

### Physiologic Bounds
- **Heart Rate**: 30-200 BPM (CONSENSUS)
- **Blood Pressure**:
  - SBP: 60-200 mmHg (CONSENSUS)
  - DBP: 30-150 mmHg (CONSENSUS)
  - MAP: 40-180 mmHg (CONSENSUS)
- **SpO2**: 70-100% (CONSENSUS)

### Signal Quality Index
- **PPG SQI**: skewness > 3.0 required (CONSENSUS)
- **PPG-ABP correlation**: >0.9 for synchronized data (CHOICE)
- **Minimum valid seconds**: 7/10 seconds must be valid (CHOICE: 70% threshold)

## Clinical Data Integration

### Core Clinical Fields
```python
CLINICAL_FIELDS = {
    'demographics': ['age', 'sex', 'height', 'weight', 'bmi'],
    'surgery': ['asa', 'department', 'optype', 'approach', 'duration'],
    'drugs': ['PPF20_CE', 'RFTN20_CE', 'vasopressor_rate'],
    'vitals': ['hr_baseline', 'map_baseline', 'spo2_baseline']
}
```

### Temporal Alignment Rules
- **Forward-fill limits**:
  - Drug infusions: 600 seconds (10 min) (CONSENSUS)
  - Vital signs: 60 seconds (1 min) (CONSENSUS)
  - Demographics: unlimited (static) (CONSENSUS)
  
- **Interpolation**:
  - Linear for continuous with gaps < 300s (5 min) (CHOICE)
  - Nearest for categorical (CONSENSUS)
  
- **Window alignment**:
  - Align clinical data to window END time (CONSENSUS)
  - Use last-observation-carried-forward within window (CONSENSUS)

## Caching Specification

### Cache Key Components
```python
cache_key = hashlib.md5(
    f"{SPEC_VERSION}_{patient_id}_{modality}_{window_sec}_{hop_sec}_"
    f"{filter_params}_{fs}_{qc_flags}_{clinical_fields}".encode()
).hexdigest()
```

### SPEC Version
- **Current**: "v1.0-2025-09" (increment on any spec change)
- **Location**: Store in cache metadata JSON

### Cache Structure
```
cache_dir/
├── vitaldb_waveform_cache_v1.0/
│   ├── metadata.json  # spec version, params
│   ├── {patient_id}/
│   │   ├── waveforms_{cache_key}.npz  # compressed arrays
│   │   ├── clinical_{cache_key}.json  # aligned clinical data
│   │   └── qc_masks_{cache_key}.npz   # quality masks
```

### Invalidation Rules
- Invalidate if SPEC_VERSION changes
- Invalidate if any filter/window/QC parameter changes
- Check cache validity via metadata.json comparison

## Implementation Notes

### NaN Policy
- **Preserve NaN**: Do not impute in waveforms (CONSENSUS)
- **Mark invalid**: Set QC mask to False for NaN regions
- **Document**: Add comment referencing VitalDB paper preserving artifacts

### Deterministic Processing
- **Random seed**: Use patient_id as seed for reproducibility
- **Window slicing**: Use exact sample indices, not time-based
- **Rounding**: Use `np.round` with 'half_even' for consistency

## References
- Filter specifications: Nature Scientific Data 2022, PMC 2022
- QC thresholds: Frontiers Digital Health 2022
- Clinical alignment: PMC 2024
- Cache versioning: Best practice from implementation
