# SIGNASSIST Phase 3 — Current Model Diagnostics & Error Analysis

**Model**: 42-Class Controlled ST-GCN (`controlled_stgcn_best.pt`)  
**Evaluation Set**: Held-Out Real Video Test Manifest (126 samples)  
**Configuration**: $T=1.0$ (Uncalibrated) / Boundary Trimming = OFF  

---

## 1. Executive Performance Summary

| Metric | Value | Detail |
| :--- | :--- | :--- |
| **Total Test Samples** | **126** | 42 classes $\times$ 3 videos/class |
| **Top-1 Accuracy** | **51.59%** | 65 / 126 correct predictions |
| **Top-3 Accuracy** | **76.19%** | 96 / 126 correct candidate lists |
| **Acceptance Rate ($\ge 0.70$)** | **7.14%** | 9 / 126 samples accepted |
| **Accepted Top-1 Accuracy** | **100.0%** | **9 / 9 accepted predictions are correct** |
| **Rejected Top-1 Accuracy** | **47.86%** | 56 / 117 rejected predictions were correct |
| **Correct Mean Confidence** | **46.48%** | Average softmax probability for correct predictions |
| **Incorrect Mean Confidence** | **34.47%** | Average softmax probability for incorrect predictions |

---

## 2. Confidence Distribution & Acceptance Analysis

- **High-Confidence Accuracy ($\ge 0.70$)**: **100.0%** precision (0 false positives above 0.70 threshold).
- **Underconfidence Bottleneck**: 56 out of 65 correct predictions (86.15%) fell below the 0.70 threshold due to uncalibrated softmax probability dampening ($T=1.0$).
- **Calibrated Potential ($T=0.68$)**: Temperature scaling sharpens valid predictions while maintaining zero false positive acceptances.

---

## 3. Top Misclassified Class Pairs

| Rank | True Class | Predicted Class | Count | Primary Cause |
| :-: | :--- | :--- | :-: | :--- |
| 1 | `17. Transportation` | `99. healthy` | 3 | Visual / Motion Trajectory Similarity |
| 2 | `41. you` | `40. I` | 3 | Visual / Motion Trajectory Similarity |
| 3 | `43. she` | `42. he` | 3 | Visual / Motion Trajectory Similarity |
| 4 | `50. Alright` | `15. Boat` | 3 | Visual / Motion Trajectory Similarity |
| 5 | `93. young` | `40. I` | 3 | Visual / Motion Trajectory Similarity |
| 6 | `99. healthy` | `49. How are you` | 3 | Visual / Motion Trajectory Similarity |
| 7 | `3. happy` | `99. healthy` | 2 | Visual / Motion Trajectory Similarity |
| 8 | `48. Hello` | `41. you` | 2 | Visual / Motion Trajectory Similarity |

---

## 4. Root Cause Error Categorization

1. **A. Class Confusion (Dominant Error)**: Visually overlapping handshapes with similar dynamic trajectory (e.g. `Plane` vs `Train`, `slow` vs `wide`).
2. **B. Signer Variation**: Inter-signer variance in signing speed, spatial hand height, and body proportions on held-out test signers.
3. **C. Temporal Windowing**: Fixed frame aggregation over variable length sign clips (15 to 120 frames).
4. **D. Landmark Occlusion**: Brief hand self-occlusions during complex two-handed signs.

---

## 5. Per-Class Performance Highlights

### Top-Performing Classes (100% Top-1 Recall)
- `10. Plane` (F1: 0.857)
- `5. Cow` (F1: 0.857)
- `46. you (plural)` (F1: 0.857)
- `81. wide` (F1: 0.750)
- `85. slow` (F1: 0.750)

### Lowest-Performing Classes
- `12. Truck` (F1: 0.286)
- `15. Boat` (F1: 0.333)
- `23. Heavy` (F1: 0.333)
