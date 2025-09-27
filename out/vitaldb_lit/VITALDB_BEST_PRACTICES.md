# VitalDB Data Handling Best Practices

## A. Waveform Processing Specifications

### PPG (Photoplethysmography)
- **Bandpass Filter**: 4th order Chebyshev Type-II, 0.5-10 Hz (Nature Scientific Data 2022)
  - Alternative: 2nd order Butterworth, 0.5-10 Hz for cardiac output (Frontiers Digital Health 2022)
- **Sampling Rate**: Downsample from 100 Hz → 25 Hz for feature extraction
- **Window/Stride**: 10s windows, 5s hop (50% overlap) (PulseDB 2022)
- **Quality Control**:
  - Flatline: variance < 0.01 or >50 consecutive identical samples
  - Spikes: z-score > ±3 for ≥5 consecutive samples
  - SQI: skewness > 3.0 required for acceptance
- **Pitfalls**: Motion artifacts, sensor detachment during surgery

### ECG (Electrocardiogram)
- **Bandpass Filter**: 4th order Butterworth, 0.5-40 Hz at 500 Hz (PMC 2022)
  - Zero-phase filtering via scipy.signal.sosfiltfilt()
- **Sampling Rate**: 500 Hz → 125 Hz for compatibility with MIMIC-III
- **Window/Stride**: 10s windows, 5s hop
- **Quality Control**:
  - Heart rate bounds: 30-200 BPM
  - R-peak detection rate > 0.8
  - Signal amplitude checks
- **Pitfalls**: Electrocautery interference, lead disconnection

### ABP (Arterial Blood Pressure)
- **Bandpass Filter**: 2nd order Butterworth, 0.5-10 Hz (PMC 2022)
- **Sampling Rate**: 100 Hz (native), maintain or downsample to 25 Hz
- **Window/Stride**: 10s windows, non-overlapping for IOH prediction
- **Quality Control**:
  - Physiologic bounds: SBP 60-200, DBP 30-150, MAP 40-180 mmHg
  - Pulse pressure: 20-120 mmHg
  - Damping detection via frequency analysis
- **Pitfalls**: Transducer drift, catheter damping, zeroing errors

### EEG (Electroencephalogram)
- **Processing**: Discrete Wavelet Transform, Daubechies 'db16', 6-level decomposition (PMC 2024)
- **Sampling Rate**: 128 Hz (native VitalDB BIS)
- **Frequency Bands**: Delta (0-4 Hz), Theta (4-8 Hz), Alpha (8-16 Hz), Beta (16-32 Hz), Gamma (32-64 Hz)
- **Window/Stride**: 56s windows, 1s stride for real-time monitoring
- **Quality Control**:
  - Impedance checks < 10 kΩ
  - Artifact rejection via entropy thresholding
- **Pitfalls**: Muscle artifacts, eye movement contamination

## B. Clinical/EMR Data

### Most-Used Clinical Fields

#### Demographics (Core)
- Age, Sex, Height, Weight, BMI (VitalDB API)
- ASA physical status (1-6 scale) - critical predictor
- Surgery type, department, approach (open/laparoscopic/robotic)

#### Drug Infusions (Orchestra Pump Data)
- **Propofol**: Orchestra/PPF20_CE track (mcg/mL effect-site concentration)
  - Conversion: volume × 20 mg/mL ÷ weight = mg/kg/hr
- **Remifentanil**: Orchestra/RFTN20_CE track  
- **Vasopressors**: 
  - Norepinephrine (20 mcg/mL), Epinephrine (20 mcg/mL)
  - Phenylephrine (100 mcg/mL)
  - Convert to mcg/kg/min using patient weight

#### Vital Signs & Labs
- Baseline vitals: HR, BP, SpO2, Temperature
- Basic labs if available: Hemoglobin, Creatinine
- Fluid balance: crystalloids, colloids, blood products

### Temporal Alignment Strategies

#### Forward-Fill (CONSENSUS)
- Drug infusions: max gap 10 minutes
- Vital signs: max gap 1 minute  
- Categorical variables: unlimited gap

#### Time Binning
- 1-minute means for trend analysis
- 5-minute windows for prediction tasks
- Align to window end time

#### Linear Interpolation
- Continuous variables with gaps < 5 minutes
- Not for categorical or drug infusion data

## C. Label Specifications

### IOH (Intraoperative Hypotension)
- **Definition**: MAP < 65 mmHg for ≥60 seconds (PMC 2025 consensus)
- **Severity Levels**: 65, 60, 55 mmHg thresholds
- **Prediction Horizons**: 3, 5, 10, 15 minutes
- **Time-weighted Average**: (depth × duration) ÷ total time

### Blood Pressure Targets
- **Regression**: Continuous MAP/SBP/DBP values
- **Windows**: Mean over 30-60 second future windows
- **Multi-horizon**: 1-5 min (short), 5-15 min (medium), 15-30 min (long)

### Depth of Anesthesia
- **BIS Target**: 40-60 for adequate anesthesia
- **Prediction**: 30-180 seconds ahead
- **Missing Data**: Forward-fill up to 5 min or linear interpolation < 30s
- **Exclusion**: Cases with >20% missing BIS

## D. Splits & Evaluation

### Patient-Level Splitting (CONSENSUS)
- **Standard**: 80% train, 10% val, 10% test
- **Stratification**: By demographics, surgery type, case complexity
- **Multiple Surgeries**: Keep all cases from same patient together

### Evaluation Metrics
#### Classification (IOH)
- AUROC (primary, >0.80 good)
- AUPRC (for imbalanced data)
- Sensitivity/Specificity at operating points
- Calibration: Brier score, ECE

#### Regression (BP)
- MAE (clinical interpretability)
- RMSE (error magnitude)
- R² (variance explained)
- Bland-Altman plots

### External Validation
- MIMIC-III for cross-database testing
- Expect 10-15% performance drop
- Report both internal and external metrics

## Citations
- VitalDB Database: Nature Scientific Data 2022 (https://www.nature.com/articles/s41597-022-01411-5)
- PulseDB Processing: Frontiers Digital Health 2022 (https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854)
- IOH Prediction: PMC 2022 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9362925/)
- EEG Processing: PMC 2024 (https://pmc.ncbi.nlm.nih.gov/articles/PMC11582228/)
- Anesthesia Depth: BMC Med Inform 2025 (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-025-02986-w)
