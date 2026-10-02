# Read-Only Error and Confidence Analysis (No-Boundary-Trimming Evaluation)

## Executive Summary

- **Evaluation Dataset**: 126 held-out test videos (`data/processed/include/controlled/test.csv`)
- **Pipeline Mode**: `backend="experimental_42"`, `use_boundary_trimming=False` (Full sequence classification)
- **Top-1 Accuracy**: **51.59%** (65 / 126 correct)
- **Top-3 Accuracy**: **76.19%** (96 / 126 correct)
- **Acceptance Rate ($\ge 0.70$ Threshold)**: **7.14%** (9 / 126 accepted)
- **Accepted Accuracy**: **100.0%** (9 / 9 accepted predictions are correct)
- **Offline Benchmark Comparison**: Top-1 = 50.79%, Top-3 = 79.37%. Discrepancy: **+0.79% Top-1**, **-3.17% Top-3**.

---

## 1. Quantitative Confidence & Binning Breakdown

### Exact Quantifications
| Metric / Category | Correct Predictions (Top-1 Match) | Incorrect Predictions (Top-1 Miss) | Total |
| :--- | :---: | :---: | :---: |
| **Total Count** | **65** (51.59%) | **61** (48.41%) | **126** |
| **Accepted ($\ge 0.70$ Conf)** | **9** (13.85% of correct) | **0** (0.00% of incorrect) | **9** (7.14%) |
| **Rejected ($< 0.70$ Conf)** | **56** (86.15% of correct) | **61** (100.0% of incorrect) | **117** (92.86%) |
| **Mean Confidence Score** | **44.97%** | **33.30%** | **39.32%** |

### Confidence Distribution Bins
| Confidence Bin | Correct Predictions | % of Correct | Incorrect Predictions | % of Incorrect | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **High ($\ge 0.70$)** | **9** | **13.85%** | **0** | **0.00%** | **9** |
| **Mod-High ($[0.50, 0.70)$)** | **17** | **26.15%** | **5** | **8.20%** | **22** |
| **Mod-Low ($[0.30, 0.50)$)** | **28** | **43.08%** | **31** | **50.82%** | **59** |
| **Low ($< 0.30$)** | **11** | **16.92%** | **25** | **40.98%** | **36** |

---

## 2. Systemic Confidence State Assessment

- **Classification**: **`SYSTEMATICALLY_UNDER_CONFIDENT`**
- **Empirical Evidence**:
  1. **86.15% of correct predictions (56 / 65)** fail the production 0.70 confidence threshold and are rejected as low confidence.
  2. The uncalibrated softmax probability output across 42 classes yields an average correct confidence of only **44.97%**.
  3. **Zero false alarms**: Exactly 0 / 61 incorrect predictions achieved confidence $\ge 0.70$.
  4. Confidence separation margin is positive (+11.67%), but raw magnitudes remain compressed below 0.70 due to uncalibrated temperature $T=1.0000$.

---

## 3. Top-1 vs Top-3 Recoverability Analysis

- **Top-1 Correct**: **65 / 126** (51.59%)
- **Top-3 Correct**: **96 / 126** (76.19%)
- **Recoverable Misses (Top-1 Miss, Top-3 Match)**: **31 samples** (**50.82% of all 61 Top-1 errors are recovered in Top-3!**)
- **Unrecoverable Misses (Not in Top-3)**: **30 samples** (23.81% of test set)

---

## 4. Class Performance (Strongest & Weakest by Recall)

### 10 Strongest Classes (Highest Top-1 Recall)
| Class ID | Label | Support | True Positives | Recall | Precision | F1 Score | Top-3 Recall | Mean Conf |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `12. Truck` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 65.7% |
| 11 | `46. you (plural)` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 88.8% |
| 12 | `47. they` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 43.3% |
| 15 | `5. Cow` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 74.9% |
| 24 | `79. short` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 36.7% |
| 25 | `80. tall` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 45.6% |
| 27 | `82. narrow` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 38.2% |
| 29 | `84. small little` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 41.3% |
| 30 | `85. slow` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 71.7% |
| 39 | `97. dry` | 3 | 3 | 100.0% | 100.0% | 1.0000 | 100.0% | 23.2% |

### 10 Weakest Classes (Lowest Top-1 Recall)
| Class ID | Label | Support | True Positives | Recall | Precision | F1 Score | Top-3 Recall | Mean Conf |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 41 | `99. healthy` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 100.0% | 59.0% |
| 38 | `96. wet` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 0.0% | 18.5% |
| 37 | `95. bad` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 33.3% | 19.4% |
| 36 | `94. good` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 0.0% | 19.3% |
| 35 | `93. young` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 100.0% | 56.7% |
| 31 | `86. fast` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 0.0% | 25.9% |
| 23 | `77. Boy` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 0.0% | 22.6% |
| 16 | `50. Alright` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 100.0% | 46.4% |
| 13 | `48. Hello` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 66.7% | 17.7% |
| 9 | `43. she` | 3 | 0 | 0.0% | 0.0% | 0.0000 | 100.0% | 34.3% |

---

## 5. Dominant Confusion Pairs & Specific Audit

### Most Frequent Confusion Pairs Across Test Set
- True: `17. Transportation` $\rightarrow$ Pred: `99. healthy` (**3 occurrences**)
- True: `41. you` $\rightarrow$ Pred: `40. I` (**3 occurrences**)
- True: `43. she` $\rightarrow$ Pred: `42. he` (**3 occurrences**)
- True: `50. Alright` $\rightarrow$ Pred: `15. Boat` (**3 occurrences**)
- True: `93. young` $\rightarrow$ Pred: `40. I` (**3 occurrences**)
- True: `99. healthy` $\rightarrow$ Pred: `49. How are you` (**3 occurrences**)
- True: `3. happy` $\rightarrow$ Pred: `99. healthy` (**2 occurrences**)
- True: `48. Hello` $\rightarrow$ Pred: `41. you` (**2 occurrences**)
- True: `52. Good afternoon` $\rightarrow$ Pred: `89. warm` (**2 occurrences**)
- True: `86. fast` $\rightarrow$ Pred: `40. I` (**2 occurrences**)

### Specific Requested Confusion Pairs Audit
| True Label | Predicted Label | Occurrences | Diagnosis / Notes |
| :--- | :--- | :---: | :--- |
| `40. I` | `41. you` | **0** | Pronominal directional index similarity |
| `41. you` | `40. I` | **3** | Reverse pronominal confusion |
| `42. he` | `43. she` | **0** | 3rd person singular pronoun confusion |
| `43. she` | `42. he` | **3** | Reverse 3rd person pronoun confusion |
| `15. Boat` | `50. Alright` | **1** | Hand shape / movement trajectory overlap |
| `52. Good afternoon` | `89. warm` | **2** | Open palm sweeps |
| `77. Boy` | `42. he` / `43. she` / `9. Train` | **3** | Person noun / pronoun confusions |
| `94. good` / `95. bad` | Inter-confusions | **0** | Antonym facial / hand gesture similarity |

---

## 6. Generated Output Artifacts

- **Detailed Analysis JSON**: [no_trim_error_analysis.json](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/no_trim_error_analysis.json)
- **42x42 Confusion Matrix CSV**: [no_trim_confusion_matrix.csv](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/no_trim_confusion_matrix.csv)
- **Markdown Analysis Report**: [no_trim_error_analysis.md](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/no_trim_error_analysis.md)
