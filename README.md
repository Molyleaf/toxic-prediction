# GenoToxMass Standard Operating Procedure (SOP)

## Overview

This document describes the standard workflow for predicting chemical genotoxicity using the **GenoToxMass** platform, covering sample preparation, LC-MS/MS data acquisition, feature extraction, model invocation, result interpretation, and quality control.

It is intended for:

- Regulatory agencies
- Contract research organizations (CROs)
- Analytical laboratories

The procedure is designed to support standardized implementation, independent validation, and routine operational use.

## 1. Scope and Purpose

This SOP defines the complete workflow for predicting chemical genotoxicity using the GenoToxMass platform, from sample preparation to result interpretation.

## 2. Instrumentation and Materials

### 2.1 Chromatographic System

| Item | Specification |
|---|---|
| Recommended column | C18 column (2.1-3.5 um particle size, 2.1 x 50 mm) |
| Column temperature | 30 +/- 5 C |
| Mobile phase A | Water with 0.1% formic acid (LC-MS grade) |
| Mobile phase B | Acetonitrile with 0.1% formic acid (LC-MS grade) |
| Flow rate | 0.3-1.0 mL/min |
| Injection volume | 1-5 uL |

#### Gradient Program

| Time (min) | %B |
|---:|---:|
| 0.0 | 5 |
| 1.0 | 5 |
| 8.0 | 95 |
| 10.0 | 95 |
| 10.1 | 5 |
| 12.0 | 5 |

### 2.2 Mass Spectrometry System

| Item | Specification |
|---|---|
| Ionization source | Electrospray Ionization (ESI) |
| Scanning method | Select the mode exhibiting higher signal strength |
| Capillary voltage | 5.0 kV |
| Ion source temperature | 325 C |
| Desolvation gas flow | 12 L/min (N2) |
| Cone gas flow | 12 L/min (N2) |
| Collision energy | Ensure the molecular ion peak is retained at moderate intensity (10%-80%) |
| Mass range | m/z 50-1000 |
| Resolution | >= 20,000 (FWHM) at m/z 200 |

### 2.3 Quality Control Standards

| Item | Specification |
|---|---|
| System suitability test mixture | Colchicine (CAS 64-86-8), Reserpine (CAS 50-55-5), each at 1 ug/mL |
| Acceptance criteria | Retention time variation <= 0.2 min; mass accuracy <= 5 ppm; intensity variation <= 20% across three consecutive injections |

## 3. Sample Preparation

### 3.1 Environmental Water Samples

1. Filter sample through a 0.22 um nylon membrane filter.
2. Add internal standard to a final concentration of 100 ng/mL.
3. Concentrate 100 mL sample using solid-phase extraction (SPE).
4. Condition SPE cartridge with 5 mL methanol, then 5 mL water.
5. Load sample at 5 mL/min.
6. Wash with 5 mL 5% methanol in water.
7. Elute with 5 mL methanol.
8. Evaporate to dryness under N2 at 40 C.
9. Reconstitute in 200 uL methanol:water (1:1, v/v).

### 3.2 Pharmaceutical Impurities

1. Dissolve drug substance in an appropriate solvent at 10 mg/mL.
2. Dilute to 10 ug/mL with methanol:water (1:1, v/v).
3. Filter through a 0.22 um PTFE syringe filter.

### 3.3 Food Matrices

1. Homogenize sample (5 g) with 10 mL acetonitrile.
2. Centrifuge at 10,000 x g for 10 min at 4 C.
3. Collect supernatant and perform dispersive SPE cleanup.
4. Evaporate 5 mL supernatant to dryness.
5. Reconstitute in 500 uL methanol:water (1:1, v/v).

## 4. Data Acquisition

### 4.1 Acquisition Mode

- Use data-dependent acquisition (DDA) or data-independent acquisition (DIA).
- For completely unknown samples, acquire both positive and negative ionization modes, then use the spectra from the mode with higher overall intensity for subsequent analysis.
- Acquire MS/MS spectra.
- Set dynamic exclusion to 5 s after two occurrences.

### 4.2 File Format Requirements

- Export raw data as `.raw`.
- Alternatively, use the GenoToxMass template (`.xlsx`) with the following required columns:

| Column | Description | Example |
|---|---|---|
| `mz_list` | Comma-separated list of m/z values | `121.05, 149.02, 177.05` |
| `intensity_list` | Comma-separated list of corresponding intensities | `100, 45, 12` |

## 5. Data Preprocessing and Feature Extraction

> Software performs this section automatically.

### 5.1 Spectral Filtering

1. Identify the maximum intensity (`MaxM`) in the raw spectrum.
2. Set filtering threshold `T = MaxM x P`, where `P = 0.01` (1%).
3. Retain only peaks with intensity >= `T`.

### 5.2 Feature Calculation (14 Parameters)

| Feature | Abbreviation | Calculation / Definition |
|---|---|---|
| Peak Number | PN | Count of peaks after filtering |
| Base Peak | BP | m/z value of the most intense peak in filtered spectrum |
| Base Peak Proximity | BPP | m/z difference between the BP and its nearest peak |
| Maximum Mass | MaxM | Maximum m/z value in filtered spectrum |
| Maximum Mass Proximity | MaxMP | m/z difference between the MaxM and its nearest peak |
| Minimum Mass | MinM | Minimum m/z value in filtered spectrum |
| Mass Mean | MM | Mean of all m/z values in filtered spectrum |
| Mass Standard Deviation | MSD | Standard deviation of all m/z values |
| Intensity Mean | IM | Mean of all intensity values |
| Intensity Standard Deviation | ISD | Standard deviation of all intensity values |
| Intensity Density | ID | BPI / PN |
| Retention Time | RT | As acquired (minutes) |
| Collision Energy | CE | As acquired (eV) |
| Precursor Type | PT | `0` for `[M-H]-`, `1` for `[M+H]+`, `[M+Na]+`, `[M+K]+` |

#### Optional Generic Input Profile

If the user does not provide the three test-related parameters (`RT`, `CE`, and `PT`), a generic input profile may be used:

- `RT = 8 min`
- `CE = 30 eV`
- `PT = positive ion mode`

**Important notes:**

- If a specific ionization mode is selected, spectra must be acquired under that same mode.
- This fallback is **not recommended unless absolutely necessary**.
- Predictive performance under this mode is expected to degrade by **>= 10%** relative to external validation results.

## 6. Model Invocation

### 6.1 Online Platform Access

1. Navigate to `https://mspredict.com/genotoxicity`.
2. Download the input template from the website.
3. Prepare data following the template format.
4. Upload the completed `.xlsx` file.
5. Click **Predict** and wait for processing (typically less than 5 seconds).
6. Review the output:
  - **Prediction:** `Toxic` or `Non-Toxic`
  - **Probability:** value between `0` and `1`

## 7. Result Interpretation and Decision Framework

### 7.1 Default Classification Threshold

- Optimal threshold `tau*` is determined by user need.
- Classification rule:
  - If probability >= `tau*` -> classify as **Toxic**
  - If probability < `tau*` -> classify as **Non-Toxic**

### 7.2 Threshold Adjustment for Specific Applications

| Application Priority | Recommended Threshold | Expected Performance Trade-off |
|---|---:|---|
| Maximize Accuracy | 0.42 | Balanced sensitivity/specificity |
| Minimize False Negatives (Safety-First) | 0.30 | Higher recall, lower precision |
| Minimize False Positives (Cost-Sensitive) | 0.65 | Higher specificity, lower recall |
| Regulatory Confirmation | 0.55 | Optimized for both |

### 7.3 Confidence Scoring

| Confidence Level | Probability Range |
|---|---|
| High | `>= 0.8` or `<= 0.2` |
| Medium | `0.6 <= p < 0.8` or `0.2 < p <= 0.4` |
| Low | `0.4 < p < 0.6` |

## 8. Quality Control and Validation

### 8.1 System Suitability Test

1. Inject system suitability test mixture (colchicine, reserpine).
2. Verify retention times are within +/- 0.2 min of established values.
3. Verify mass accuracy <= 5 ppm for all three compounds.
4. Run GenoToxMass prediction on all three.
5. If any prediction deviates, investigate instrument performance.

### 8.2 Batch Quality Control

- Include one positive control (for example, colchicine) and one negative control (for example, reserpine) in each analytical batch.
- Acceptance criteria: correct classification for both controls.
- If controls fail, reject the entire batch and re-analyze.

### 8.3 Inter-Laboratory Transfer

For laboratories implementing this method for the first time:

1. Analyze the reference standard set (`n = 20 compounds`).
2. Compare predictions with reference values.
3. Acceptable performance: `>= 85%` concordance.
4. Document all results for regulatory audit.

## 9. Documentation and Reporting

All analyses should be documented with:

1. Instrument calibration records
2. System suitability test results
3. Sample preparation logs
4. Raw data files
5. Feature extraction worksheets
6. Model prediction outputs
7. Final interpretation and decision rationale

## 10. Troubleshooting Guide

| Problem | Possible Cause | Solution |
|---|---|---|
| Low peak count after filtering | High noise level | Increase injection volume; improve sample cleanup |
| Unexpected predictions | Poor spectral quality | Verify collision energy; check precursor ion selection |
| Platform unavailable | Server maintenance | Contact support |

## Source

Converted from the uploaded SOP PDF for GenoToxMass.
