# SIGNASSIST — Production Model Promotion Criteria

## 1. Overview & Policy Statement
No experimental model checkpoint may replace `models/checkpoints/dynamic_classifier/dynamic_classifier_best.pt` or be connected to the primary production UI pipeline based solely on training loss convergence, unit tests passing, or code execution completing.

A candidate model is eligible for production promotion ONLY after satisfying ALL TEN empirical criteria defined in this specification.

---

## 2. Mandatory Production Promotion Criteria

### Criterion 1: Rigorous Held-Out Test Evaluation
- **Requirement**: Candidate models MUST be evaluated on the strict held-out test split (`data/processed/include/controlled/test.csv`).
- **Rule**: The test dataset must NEVER be used during training, hyperparameter tuning, or early stopping checkpoint selection.

### Criterion 2: Elimination of Class-Collapse Behavior
- **Requirement**: The model's prediction distribution across the test set MUST cover at least 60% of all vocabulary classes.
- **Rule**: Any model predicting fewer than 50% of candidate classes or showing over-prediction of dominant single classes (e.g. >25% of all predictions assigned to one gloss) is disqualified immediately.

### Criterion 3: Meaningful Top-1 Accuracy Threshold
- **Requirement**: Candidate model Top-1 accuracy on the held-out test set must achieve a minimum threshold:
  - For Controlled Vocabulary (42 classes): $\text{Top-1} \ge 35.0\%$
  - For Full Vocabulary (262 classes): $\text{Top-1} \ge 20.0\%$

### Criterion 4: Meaningful Macro F1-Score
- **Requirement**: Macro F1-score across all active vocabulary classes must exceed:
  - Controlled Vocabulary (42 classes): $\text{Macro-F1} \ge 0.3000$
- **Rule**: Evaluates balanced class performance and prevents high overall accuracy caused by majority class dominance.

### Criterion 5: Per-Class Performance Inspection
- **Requirement**: Per-class precision, recall, and F1 metrics must be generated via `scripts/evaluate_controlled_model.py`. Zero-F1 classes must not exceed 25% of the total target vocabulary.

### Criterion 6: Real External Video File Validation
- **Requirement**: Model inference MUST be executed on at least 5 real, unseen physical MP4 video samples outside the preprocessed INCLUDE dataset.
- **Rule**: Ground-truth signs in test videos must be correctly identified with high confidence ($\ge 0.70$).

### Criterion 7: Strict Confidence & Unknown Rejection Verification
- **Requirement**: The model must maintain clean probability calibration.
- **Rule**: Out-of-vocabulary gestures, static poses, or ambiguous movements MUST yield peak softmax probabilities $< 0.70$ and be cleanly rejected by the `RecognitionPipeline`.

### Criterion 8: Real-Time Live Webcam Session Smoke Test
- **Requirement**: End-to-end live webcam session test executing MediaPipe Holistic landmark extraction + ST-GCN inference in real-time.
- **Rule**: Zero runtime crashes, memory leaks, or unhandled tensor dimension errors during a 60-second webcam session.

### Criterion 9: Video Upload Pipeline Smoke Test
- **Requirement**: Successful processing of 1080p and 720p user video uploads through Streamlit UI.
- **Rule**: Accurate generation of skeleton visualization overlay and timeline predictions.

### Criterion 10: Video URL Fetching & Recognition Smoke Test
- **Requirement**: Successful fetching and processing of public video URLs via `yt-dlp` stream handler.
- **Rule**: Clean download, temporal frame extraction, and pipeline evaluation without UI freezing.

---

## 3. Promotion Verification Protocol

When a candidate model satisfies all 10 criteria:

1. Generate promotion validation report:
   ```bash
   python scripts/evaluate_controlled_model.py \
     --checkpoint models/checkpoints/experimental/controlled_stgcn_best.pt \
     --split-csv data/processed/include/controlled/test.csv \
     --out-report reports/evaluations/promotion_candidate_test.json
   ```
2. Inspect held-out metrics, confusion matrix, and prediction distributions.
3. Obtain explicit verification sign-off before overwriting or pointing the active recognition pipeline to the new checkpoint.
