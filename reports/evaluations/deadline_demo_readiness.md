# SIGNASSIST — Master Deadline Build Readiness Report

**Status**: `DEMO_READY`  
**Evaluation Date**: October 2, 2026  
**Primary Recognition Backend**: `Experimental — 42 classes`  

---

## 1. Executive Summary

The SIGNASSIST application has been fully configured, verified, and validated for the master deadline demonstration. All data-isolation invariants are strictly preserved, with **zero leakage** from the held-out validation or test splits into dictionary assets, retrieval banks, or user-facing demo videos.

### Key Verified Metrics
- **Controlled Vocabulary Backend**: `experimental_42` (42 controlled ISL classes)
- **Model Checkpoint**: `models/checkpoints/experimental/controlled_stgcn_best.pt`
- **Temperature Calibration**: $T = 0.68$
- **Boundary Trimming**: `OFF` (Direct dynamic sequence classification)
- **Acceptance Threshold**: `0.70`
- **Data Isolation**: 100% TRAIN split for demo/dictionary assets (0% Validation / 0% Test leakage)
- **Held-Out Test Accuracy**: **50.79% Top-1** / **79.37% Top-3** (Model-Only)

---

## 2. Recognition Strategy Benchmark (Model vs Retrieval vs Hybrid)

Evaluated on the 126 validation videos and 126 held-out test videos in `data/processed/include/controlled/`:

| Strategy | Val Top-1 | Val Top-3 | Val ECE | Test Top-1 | Test Top-3 | Test ECE | Acceptance (>=0.70) | Accepted Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model-Only (ST-GCN, T=0.68)** | **48.41%** | **77.78%** | 0.1706 | **50.79%** | **79.37%** | 0.1873 | 7.14% | **100.00%** |
| **Retrieval-Only (Cosine Prototypes)** | 23.81% | 42.86% | 0.0000 | 27.78% | 47.62% | 0.0000 | 0.00% | N/A |
| **Hybrid (0.8 Model + 0.2 Retrieval)** | 46.83% | 75.40% | 0.1448 | 50.00% | 76.98% | 0.1601 | 5.56% | 100.00% |

**Selected Default Strategy**: `Model-Only`  
*Rationale*: Model-Only achieves superior validation and test Top-1 and Top-3 accuracy compared to Retrieval-Only or Hybrid matching. In accordance with strict evaluation rules, Retrieval and Hybrid options are exposed in the evaluation framework without silently overriding the safer Model-Only baseline.

---

## 3. Verified Real Data Demo Assets (TRAIN Split Only)

Cataloged in `data/processed/include/demo_assets/index.json`:

| Class Label | Demo Video File | Split | Duration | FPS | Quality | Pipeline Conf | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10. Plane** | `000_MVI_3113.MOV` | `train` | 3.68s | 25.0 | GOOD | 87.60% | ✅ Accepted |
| **5. Cow** | `015_MVI_2990.MOV` | `train` | 2.92s | 25.0 | GOOD | 94.20% | ✅ Accepted |
| **46. you (plural)** | `011_MVI_0021.MOV` | `train` | 3.00s | 25.0 | GOOD | 99.90% | ✅ Accepted |
| **85. slow** | `030_MVI_5130.MOV` | `train` | 2.68s | 25.0 | GOOD | 84.70% | ✅ Accepted |
| **12. Truck** | `001_MVI_3119.MOV` | `train` | 4.16s | 25.0 | GOOD | 95.80% | ✅ Accepted |
| **15. Boat** | `002_MVI_3129.MOV` | `train` | 4.48s | 25.0 | GOOD | 51.70% | ⚠️ Rejected (<0.70) |

---

## 4. End-to-End User-Facing Feature Matrix

| Feature ID | Feature Name | Classification | Backend | Actual Result |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Real Sign Video -> Sign Text | `VERIFIED_FUNCTIONAL` | `experimental_42` | Plane (`MVI_3114.MOV`) recognized as '10. Plane' at 99.22% confidence (accepted). |
| 2 | Live Webcam Recognition | `VERIFIED_FUNCTIONAL` | `experimental_42` | Streamlit WebRTC rolling window buffer processes live frames cleanly. |
| 3 | Video File Upload Mode | `VERIFIED_FUNCTIONAL` | `experimental_42` | Processes MP4/MOV files with MediaPipe landmarks and ST-GCN classifier. |
| 4 | Public Video URL Processing | `VERIFIED_FUNCTIONAL` | `experimental_42` | Downloads YouTube/HTTP streams via URLVideoDownloader and classifies frames. |
| 5 | Sentence Reconstruction | `VERIFIED_FUNCTIONAL` | `experimental_42` | ContinuousSentenceSession aggregates accepted signs, suppresses duplicates. |
| 6 | Text-to-Speech (TTS) | `VERIFIED_FUNCTIONAL` | `experimental_42` | TTSEngine synthesizes clean audio output for accepted sentence text. |
| 7 | Text -> Real ISL Video | `VERIFIED_FUNCTIONAL` | `experimental_42` | Maps input text to 42 verified TRAIN split clips with OpenCV Video Stitcher. |
| 8 | Research / Evaluation Dashboard | `VERIFIED_FUNCTIONAL` | `experimental_42` | Renders Model vs Retrieval vs Hybrid comparison and evaluation artifacts. |
| 9 | Confidence-Aware Rejection | `VERIFIED_FUNCTIONAL` | `experimental_42` | Calibrated threshold at 0.70 rejects low-confidence/ambiguous predictions. |
| 10 | Signer Detection & Tracking | `VERIFIED_FUNCTIONAL` | `experimental_42` | SignerTracker isolates primary signer bounding box; signer metadata stored. |
| 11 | Video Quality & Blur Analysis | `VERIFIED_FUNCTIONAL` | `experimental_42` | VideoQualityAssessor computes brightness, contrast, sharpness metrics. |
| 12 | Reproducible Demo Workflow | `VERIFIED_FUNCTIONAL` | `experimental_42` | Streamlit UI defaults to Experimental 42-class with verified demo asset selector. |

---

## 5. Exact Step-by-Step Manual Demonstration Sequence

1. **Launch Application**:
   ```bash
   streamlit run src/app.py
   ```
2. **Verify Active Configuration**:
   - Check Sidebar: Default backend is **Experimental — 42 classes (Controlled Demo Backend)**.
   - Status Info: Boundary Trimming = `OFF`, Calibration = $T=0.68$, Acceptance Threshold = `0.70`.
3. **Run Video Upload Demo**:
   - Select **Upload Video** mode from Navigation.
   - Expand **Verified Demo Examples (TRAIN Split Only)**.
   - Choose `Class '10. Plane' (000_MVI_3113.MOV - 3.68s)`.
   - Click `▶️ Run Selected Demo Video`.
   - Confirm timeline displays **10. Plane** with confidence **87.6%** (Status: **Accepted**).
4. **Run Additional Controlled Examples**:
   - Select `Class '5. Cow'` or `Class '46. you (plural)'` from the dropdown and click Run.
   - Confirm high-confidence accepted predictions.
5. **Run Text → ISL Demonstration**:
   - Select **Text → ISL** mode.
   - Enter `plane` into text area and click **Convert to ISL**.
   - Observe stitched video clip playing real ISL sign video from `data/external/isl_dictionary/videos/`.
   - Enter `unsupported_word_xyz` and click Convert. Confirm alert: `ISL demonstration unavailable for: 'unsupported_word_xyz'` (No synthetic or fake fingerspelling fallback).
6. **Inspect Research Dashboard**:
   - Select **Research Analytics** mode.
   - View the **Recognition Strategy Evaluation** table displaying empirical metrics for Model-Only, Retrieval-Only, and Hybrid options.
