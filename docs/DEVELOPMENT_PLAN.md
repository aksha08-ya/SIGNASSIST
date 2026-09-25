# SIGNASSIST — Project Development Plan

## 1. Development Roadmap Overview & Dependency Graph

SIGNASSIST is implemented across 19 modular phases. Dependencies between phases are strictly defined to ensure a systematic, testable, and reproducible development workflow.

### Phased Roadmap Dependency Diagram

`mermaid
flowchart TD
    P1[Phase 1: Project Infrastructure & Setup] --> P2[Phase 2: Perception & Landmark Extraction]
    P2 --> P3[Phase 3: Graph Construction & Normalization]
    P3 --> P4[Phase 4: Static Sign GNN Model]
    P3 --> P5[Phase 5: Dynamic ST-GCN + Transformer]
    P3 --> P6[Phase 6: Learned Static/Dynamic Router]
    P4 --> P7[Phase 7: Multimodal Fusion Engine]
    P5 --> P7
    P6 --> P7
    P7 --> P8[Phase 8: Confidence Estimation & Rejection]
    P8 --> P9[Phase 9: Sign Boundary Detection]
    P9 --> P10[Phase 10: Sentence Reconstruction & NLP]
    P10 --> P11[Phase 11: Text-to-Speech Engine]
    P2 --> P12[Phase 12: Signer Tracking & Multi-Person Intelligence]
    P12 --> P13[Phase 13: Video File Upload Subsystem]
    P12 --> P14[Phase 14: Video URL Ingestion & Fallback]
    P10 --> P15[Phase 15: Text-to-ISL Demonstration Engine]
    P8 --> P16[Phase 16: Research, LOSO & Ablation Framework]
    P11 --> P17[Phase 17: Streamlit Interactive UI Application]
    P13 --> P17
    P14 --> P17
    P15 --> P17
    P16 --> P17
    P17 --> P18[Phase 18: Integration & E2E Testing Suite]
    P18 --> P19[Phase 19: Deployment Packaging & Audit]
`

---

## 2. Detailed Phase Specifications

### Phase 1: Environment & Project Infrastructure Setup
* **Objective:** Establish virtual environment, project directory structure, core logging, exception taxonomy, and configuration management strategy.
* **Deliverables:**
  * Configured virtual environment with Python 3.12 dependencies (requirements.txt).
  * configs/default.yaml, configs/model.yaml, configs/ui.yaml.
  * Logging system in src/utils/logger.py and exception taxonomy in src/utils/exceptions.py.
* **Dependencies:** None.
* **Validation Criteria:** pytest passes clean initialization tests; config files parse without errors.
* **Definition of Done:** Project structure fully initialized with clean configuration loading and logging verified.

---

### Phase 2: Core Data Ingestion & Landmark Extraction Pipeline
* **Objective:** Implement MediaPipe Holistic integration for real-time 3D landmark extraction from webcam and video frames.
* **Deliverables:**
  * src/data/landmark_extractor.py (HolisticLandmarkExtractor class).
  * Feature outputs: 21 left hand, 21 right hand, 33 body pose, 40 key face landmarks.
* **Dependencies:** Phase 1.
* **Validation Criteria:** Extracts valid landmark coordinates at > 30 FPS on sample video files.
* **Definition of Done:** Landmark extraction unit tested on webcam feed and synthetic image frames without coordinate corruption.

---

### Phase 3: Skeleton Graph Construction & Feature Normalization
* **Objective:** Build spatial graph adjacency matrices and implement torso-centric coordinate normalization + velocity features.
* **Deliverables:**
  * src/data/graph_builder.py (Adjacency matrix builder A in R^(115x115)).
  * src/data/normalizer.py (Translation, scale normalization, kinematic velocity tensor generator).
* **Dependencies:** Phase 2.
* **Validation Criteria:** Normalized landmark tensors invariant to camera distance scaling and position translation.
* **Definition of Done:** Graph adjacency matrices verified; tensor shape normalized to [B, 5, T, 115].

---

### Phase 4: Static Sign Recognition Pipeline (Spatial GNN)
* **Objective:** Build and train 3-layer Spatial Graph Convolutional Network (GCN) for stationary alphabet/numeral signs.
* **Deliverables:**
  * src/models/static_gcn.py (StaticGraphGCN PyTorch module).
  * Training script scripts/train_static.py and saved weights models/static_gcn.pt.
* **Dependencies:** Phase 3.
* **Validation Criteria:** Static GCN achieves > 92% accuracy on static ISL benchmark split.
* **Definition of Done:** Model checkpoint saved; unit test verifies forward pass output shape [B, 256].

---

### Phase 5: Dynamic Sign Recognition Pipeline (ST-GCN + Transformer)
* **Objective:** Implement spatial-temporal graph convolution backbone combined with Transformer temporal encoder for movement-based signs (**strictly excluding LSTM/GRU as primary**).
* **Deliverables:**
  * src/models/stgcn.py (STGCNBackbone module).
  * src/models/transformer_encoder.py (TransformerTemporalEncoder module).
  * Training script scripts/train_dynamic.py and weights models/dynamic_stgcn_transformer.pt.
* **Dependencies:** Phase 3.
* **Validation Criteria:** ST-GCN + Transformer achieves > 88% top-1 accuracy on dynamic sign sequence dataset.
* **Definition of Done:** Dynamic pathway unit tested; forward pass produces spatial-temporal embeddings [B, 256].

---

### Phase 6: Learned Static/Dynamic Routing Mechanism
* **Objective:** Develop learned MLP classifier to route frame sequences to static or dynamic pathway automatically based on kinematic motion features.
* **Deliverables:**
  * src/features/kinematic_features.py (Displacement, speed, motion variance extractor).
  * src/models/router.py (StaticDynamicRouter MLP classifier).
* **Dependencies:** Phase 3.
* **Validation Criteria:** Router achieves > 95% accuracy in separating stationary keyframes from dynamic sequences.
* **Definition of Done:** Router trained and benchmarked against manual threshold baseline.

---

### Phase 7: Multimodal Feature Fusion Engine
* **Objective:** Combine spatial GCN embeddings, dynamic ST-GCN representations, and non-manual facial cues into a unified classification head.
* **Deliverables:**
  * src/models/fusion.py (MultimodalFusionClassifier module).
* **Dependencies:** Phase 4, Phase 5, Phase 6.
* **Validation Criteria:** Multimodal fusion outperforms single-modality baselines by at least 4% in accuracy.
* **Definition of Done:** Unified model forward pass tested; output logits shape [B, N_classes] verified.

---

### Phase 8: Confidence Estimation & Unknown Rejection Engine
* **Objective:** Implement softmax entropy/margin sampling to flag low-confidence predictions as UNKNOWN and present top-3 predictions.
* **Deliverables:**
  * src/pipeline/confidence.py (ConfidenceEstimator class).
* **Dependencies:** Phase 7.
* **Validation Criteria:** System rejects un-signed rest poses and random noise with < 5% false positive emissions.
* **Definition of Done:** Rejection logic verified with configurable confidence thresholds (theta_conf=0.65, theta_margin=0.15).

---

### Phase 9: Continuous Sign Boundary Detection & Segmentation
* **Objective:** Implement motion energy minima detection and rest-pose classification to segment continuous sign streams.
* **Deliverables:**
  * src/pipeline/boundary_detector.py (SignBoundaryDetector class).
  * src/pipeline/temporal_smoother.py (Majority voting & debounce hysteresis filter).
* **Dependencies:** Phase 8.
* **Validation Criteria:** Detects sign transitions in continuous videos with F1-score > 0.85 on boundary timestamps.
* **Definition of Done:** Stream segmentation unit tested on multi-sign video clips.

---

### Phase 10: Sentence Reconstruction & Context-Aware NLP Layer
* **Objective:** Implement context-aware ISL-to-English grammar translation rules, word sequence assembly, duplicate suppression, and punctuation.
* **Deliverables:**
  * src/nlp/sentence_reconstructor.py (SentenceReconstructor class).
  * src/nlp/grammar_rules.py (Rule-based SVO transformation engine).
* **Dependencies:** Phase 9.
* **Validation Criteria:** Reconstructs grammatical English sentences from raw sign sequences without inventing unsupported facts.
* **Definition of Done:** NLP layer unit tested across 30 standard ISL sentence templates.

---

### Phase 11: Text-to-Speech (TTS) Engine Integration
* **Objective:** Implement voice synthesis using offline pyttsx3 with online gTTS fallback for recognized signs and sentences.
* **Deliverables:**
  * src/tts/tts_engine.py (TTSEngine module).
* **Dependencies:** Phase 10.
* **Validation Criteria:** Synthesizes audio cleanly without blocking UI thread.
* **Definition of Done:** Non-blocking TTS playback verified on Windows and Linux audio drivers.

---

### Phase 12: Signer Detection, Tracking & Multi-Person Intelligence
* **Objective:** Build bounding-box signer detector, IoU + Kalman Filter tracker, and multi-person selection mechanism.
* **Deliverables:**
  * src/video/signer_tracker.py (SignerTracker class).
  * Target selection hook for manual signer isolation.
* **Dependencies:** Phase 2.
* **Validation Criteria:** Maintains signer track ID across 100% of frames during occlusions or person swaps.
* **Definition of Done:** Multi-person bounding box overlay tested on sample multi-person video files.

---

### Phase 13: Video File Upload & Chunking Subsystem
* **Objective:** Support local MP4, MOV, AVI upload, video quality checks, frame chunking, timestamped timeline generation, and exportable PDF/JSON transcripts.
* **Deliverables:**
  * src/video/file_reader.py (VideoFileReader module).
  * src/utils/report_exporter.py (PDF/JSON transcript generator).
* **Dependencies:** Phase 12, Phase 10.
* **Validation Criteria:** Processes 1-minute video file in < 15 seconds on standard CPU.
* **Definition of Done:** Upload pipeline produces valid timeline and export files.

---

### Phase 14: Video URL Ingestion & Graceful Fallback Pipeline
* **Objective:** Integrate yt-dlp for YouTube, Instagram, Facebook, and Drive links with platform accessibility validation and local file fallback.
* **Deliverables:**
  * src/video/url_downloader.py (URLVideoDownloader module).
* **Dependencies:** Phase 13.
* **Validation Criteria:** Successfully retrieves public YouTube clips; gracefully fails with clear UI prompt when URL is restricted/private.
* **Definition of Done:** Fallback logic unit tested against mock private and invalid URL parameters.

---

### Phase 15: Text-to-ISL Reverse Demonstration Engine
* **Objective:** Implement reverse communication mode converting text phrases into sequential ISL dictionary video clips.
* **Deliverables:**
  * src/nlp/text_to_isl.py (TextToISLConverter module).
  * Curated ISL video dictionary index under data/external/isl_dictionary/.
* **Dependencies:** Phase 10.
* **Validation Criteria:** Stitches ISL video clips smoothly for any input text phrase present in dictionary.
* **Definition of Done:** Video player preview tested with word lookup and alphabet fingerspelling fallback.

---

### Phase 16: Academic Research, Evaluation & Ablation Framework
* **Objective:** Implement confusion matrix generation, F1 metrics, Leave-One-Signer-Out (LOSO) cross-validation, and ablation study framework.
* **Deliverables:**
  * src/utils/metrics.py (MetricsEvaluator class).
  * scripts/evaluate_loso.py (LOSO evaluation suite).
  * scripts/run_ablation.py (Ablation study executor).
* **Dependencies:** Phase 8, Phase 7.
* **Validation Criteria:** Generates publication-ready confusion matrices and ablation table breakdowns automatically.
* **Definition of Done:** LOSO cross-validation script executed; results saved to
reports/loso_results.json.

---

### Phase 17: Interactive Streamlit UI Application (6 Modes)
* **Objective:** Build single-entry Streamlit web interface (src/app.py) managing 6 application modes: Live Webcam, Continuous Sentence, Upload Video, Video URL, Text -> ISL, and Research Analytics.
* **Deliverables:**
  * src/app.py (Main Streamlit dashboard).
  * Dedicated view components under src/ui/.
* **Dependencies:** Phases 1 to 16.
* **Validation Criteria:** All 6 UI tabs render correctly and respond interactively without crashing or freezing.
* **Definition of Done:** UI validated end-to-end across webcam stream, upload file, URL link, text conversion, and analytics dashboard.

---

### Phase 18: System Integration, Unit & E2E Testing Suite
* **Objective:** Perform end-to-end integration testing across all modules using pytest.
* **Deliverables:**
  * Comprehensive test suite under tests/unit/ and tests/integration/.
* **Dependencies:** Phase 17.
* **Validation Criteria:** pytest passes 100% of test cases; zero memory leaks or unhandled exception crashes.
* **Definition of Done:** All unit and integration test assertions pass cleanly.

---

### Phase 19: Deployment Packaging, Model Checkpoints & Verification Audit
* **Objective:** Finalize model checkpoints, export CPU ONNX models, write user documentation, and verify reproducible setup scripts.
* **Deliverables:**
  * Model artifacts saved in models/.
  * ONNX exported models in models/onnx/.
  * Updated README.md and user guide.
* **Dependencies:** Phase 18.
* **Validation Criteria:** Clean system setup executable on a fresh environment via pip install -r requirements.txt.
* **Definition of Done:** Final project audit passed; zero uncommitted secrets or non-reproducible scripts.
