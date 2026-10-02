# Calibration-Only Study Report (Experimental 42-Class ST-GCN)

## Executive Summary

- **Objective**: Evaluate temperature scaling calibration on the experimental 42-class ST-GCN model without retraining, altering weights, or lowering the 0.70 confidence threshold.
- **Data Splitting Protocol**:
  - **Fitting**: Fitted strictly on the **validation split** (126 samples, `data/processed/include/controlled/validation.csv`).
  - **Final Evaluation**: Evaluated strictly on the **held-out test split** (126 samples, `data/processed/include/controlled/test.csv`).
- **Selection Criterion**: Temperature $T$ minimizing Negative Log-Likelihood (NLL) on validation split.
- **Selected Temperature**: **$T = 0.68$** (Baseline $T = 1.00$).
- **Prediction Invariance**: Verified **100% identical class predictions** before vs after calibration (126/126 test samples match).

---

## 1. Validation Split Temperature Grid Search Results

| Candidate Temperature | Val NLL | Val ECE | Val Top-1 Accuracy | Val Top-3 Accuracy | Selection Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$T = 1.00$ (Baseline)** | **1.2806** | **0.1924** | **66.67%** | **88.10%** | Default Production |
| **$T = 0.68$ (Optimal NLL)** | **1.1880** | **0.1060** | **66.67%** | **88.10%** | **Selected (Min NLL)** |
| **$T = 0.52$ (Optimal ECE)** | **1.2326** | **0.0880** | **66.67%** | **88.10%** | Min ECE Candidate |

---

## 2. Held-Out Test Evaluation: Baseline $T=1.00$ vs Selected $T=0.68$

| Calibration Metric | Baseline $T = 1.00$ | Selected $T = 0.68$ | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **50.79%** (64 / 126) | **50.79%** (64 / 126) | **0.00%** (Invariant) |
| **Top-3 Accuracy** | **79.37%** (100 / 126) | **79.37%** (100 / 126) | **0.00%** (Invariant) |
| **Test NLL** | **1.5351** | **1.4869** | **-0.0482** |
| **Test ECE** | **0.1520** | **0.1236** | **-0.0284** |
| **Mean Confidence (All)** | **40.98%** | **53.89%** | **+12.90%** |
| **Mean Confidence (Correct)** | **47.80%** | **62.03%** | **+14.23%** |
| **Mean Confidence (Incorrect)** | **33.95%** | **45.48%** | **+11.53%** |
| **Accepted ($\ge 0.70$ Threshold)** | **11** (8.73%) | **31** (24.60%) | **+20** |

---

## 3. Held-Out Test Confidence Bin Distributions

### Baseline $T = 1.00$
- **Correct & Accepted ($\ge 0.70$)**: **11**
- **Incorrect & High Confidence ($\ge 0.70$) (False Alarms)**: **0**
- **Correct-but-Rejected ($< 0.70$)**: **53**
- **Incorrect & Rejected ($< 0.70$)**: **62**
- **Bins (Correct Samples)**: $\ge 0.70$: 11 | $[0.50, 0.70)$: 11 | $[0.30, 0.50)$: 31 | $<0.30$: 11
- **Bins (Incorrect Samples)**: $\ge 0.70$: 0 | $[0.50, 0.70)$: 13 | $[0.30, 0.50)$: 18 | $<0.30$: 31

### Selected Temperature $T = 0.68$
- **Correct & Accepted ($\ge 0.70$)**: **20**
- **Incorrect & High Confidence ($\ge 0.70$) (False Alarms)**: **11**
- **Correct-but-Rejected ($< 0.70$)**: **44**
- **Incorrect & Rejected ($< 0.70$)**: **51**
- **Bins (Correct Samples)**: $\ge 0.70$: 20 | $[0.50, 0.70)$: 25 | $[0.30, 0.50)$: 15 | $<0.30$: 4
- **Bins (Incorrect Samples)**: $\ge 0.70$: 11 | $[0.50, 0.70)$: 9 | $[0.30, 0.50)$: 24 | $<0.30$: 18

---

## 4. Output Reports

- **Calibration Study JSON**: [calibration_study.json](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/calibration_study.json)
- **Calibration Study Markdown**: [calibration_study.md](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/calibration_study.md)
