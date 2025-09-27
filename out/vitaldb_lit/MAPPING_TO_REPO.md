# Mapping SPEC to Current Repository

## Waveform Processing

| SPEC Rule | Current Implementation | Status | Location |
|-----------|------------------------|--------|----------|
| **PPG: Chebyshev Type-II 4th order [0.5,10]Hz** | Butterworth 4th order with configurable bands | **DIVERGES** | data.py:53-63 |
| **ECG: Butterworth 4th order [0.5,40]Hz** | Same filter type, different params | **DIVERGES** | data.py:53-63 |
| **ABP: Butterworth 2nd order [0.5,10]Hz** | Not explicitly handled | **MISSING** | N/A |
| **Window: 10s, Hop: 5s** | Configurable, defaults may differ | **PARTIAL** | data.py:162-163 |
| **Zero-phase filtering** | Not using sosfiltfilt | **MISSING** | data.py:59 |
| **Resampling targets (PPG→25Hz, ECG→125Hz)** | Uses config target_fs | **PARTIAL** | data.py:67-74 |

## Quality Control

| SPEC Rule | Current Implementation | Status | Location |
|-----------|------------------------|--------|----------|
| **Flatline: variance < 0.01** | Not implemented | **MISSING** | N/A |
| **Spike: z-score > 3 for 5+ samples** | Not implemented | **MISSING** | N/A |
| **Physiologic bounds checking** | Not implemented | **MISSING** | N/A |
| **PPG SQI (skewness > 3)** | Not implemented | **MISSING** | N/A |
| **Minimum 70% valid window** | Not implemented | **MISSING** | N/A |
| **QC mask generation** | Not implemented | **MISSING** | N/A |

## Clinical Data Integration

| SPEC Rule | Current Implementation | Status | Location |
|-----------|------------------------|--------|----------|
| **Demographics extraction** | Basic implementation exists | **PARTIAL** | data.py:250-333 |
| **ASA status** | Not extracted | **MISSING** | N/A |
| **Drug infusion tracking** | Not implemented | **MISSING** | N/A |
| **Vasopressor rates** | Not implemented | **MISSING** | N/A |
| **Forward-fill with limits** | Not implemented | **MISSING** | N/A |
| **Window-aligned clinical data** | Not implemented | **MISSING** | N/A |
| **Clinical context in output** | Not structured | **MISSING** | N/A |

## Caching

| SPEC Rule | Current Implementation | Status | Location |
|-----------|------------------------|--------|----------|
| **Versioned cache keys** | Basic caching without versioning | **DIVERGES** | data.py:461-479 |
| **SPEC version in metadata** | Not implemented | **MISSING** | N/A |
| **Compressed .npz format** | Uses .npy | **DIVERGES** | data.py:474 |
| **Cache invalidation on param change** | Not implemented | **MISSING** | N/A |
| **Metadata JSON** | Not implemented | **MISSING** | N/A |

## Label Creation

| SPEC Rule | Current Implementation | Status | Location |
|-----------|------------------------|--------|----------|
| **IOH: MAP<65 for ≥60s** | Implemented separately | **MATCHES** | data/labels_vitaldb.py |
| **BP regression targets** | Implemented separately | **MATCHES** | data/labels_vitaldb.py |
| **Multi-horizon prediction** | Implemented | **MATCHES** | data/labels_vitaldb.py |

## TabPFN Integration

| SPEC Rule | Current Implementation | Status | Location |
|-----------|------------------------|--------|----------|
| **Tabular mode support** | Implemented | **MATCHES** | data.py:158-179 |
| **Feature extraction** | Implemented separately | **MATCHES** | data/features_vitaldb.py |
| **Window indexing** | Basic implementation | **PARTIAL** | data.py:438-444 |

## Overall Assessment

### ✅ Matches Spec (20%)
- Label creation logic
- Basic tabular mode support
- Feature extraction framework

### ⚠️ Partial/Diverges (30%)
- Filter implementations (wrong type/params)
- Caching (no versioning)
- Demographics (incomplete fields)
- Window/hop configuration

### ❌ Missing (50%)
- **All quality control checks**
- Zero-phase filtering
- Clinical data integration
- Drug infusion tracking
- Cache versioning/invalidation
- Structured clinical context
- ABP-specific processing

## Priority Fixes

1. **CRITICAL**: Implement quality control (flatline, spike, bounds)
2. **HIGH**: Add clinical data extraction and alignment
3. **HIGH**: Fix filter implementations to match spec
4. **MEDIUM**: Add cache versioning and invalidation
5. **MEDIUM**: Structure clinical context properly
6. **LOW**: Optimize resampling targets per modality
