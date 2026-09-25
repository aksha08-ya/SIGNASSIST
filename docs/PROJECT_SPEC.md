# SIGNASSIST — AI-Powered Indian Sign Language Recognition

## 1. Project Objective

SIGNASSIST is an AI-powered Indian Sign Language (ISL) recognition and translation system designed to convert signed communication into meaningful text and optional speech using a standard webcam or uploaded video.

The system must support static signs, dynamic signs, continuous signing, sentence reconstruction, video analysis, confidence-aware recognition, and accessible text-to-speech output.

The system should prioritize signer-independent recognition, robustness to real-world environments, and practical deployment without specialized gloves, sensors, or hardware.

---

## 2. Core Functional Requirements

### 2.1 Real-Time Webcam Recognition

The system shall:
- Capture live video from a standard webcam.
- Detect and track the active signer.
- Extract relevant hand, body, and non-manual facial information.
- Recognize static ISL alphabet/sign gestures.
- Recognize dynamic ISL signs involving movement.
- Automatically determine whether the observed sign is static or dynamic.
- Convert recognized signs into text.
- Display prediction confidence.
- Provide top alternative predictions where appropriate.
- Reject or flag low-confidence predictions.
- Smooth predictions temporally to reduce flickering and duplicate outputs.
- Provide optional text-to-speech output.

### 2.2 Static Sign Recognition

The system shall recognize static signs using spatial hand/body representations.

The recognition architecture should support graph/skeleton-based learning rather than relying only on conventional image classification.

### 2.3 Dynamic Sign Recognition

The system shall recognize movement-based signs using temporal information.

The primary temporal architecture should use:
- Spatial-temporal graph representations.
- ST-GCN or equivalent spatial-temporal graph modelling.
- Transformer-based temporal modelling.

LSTM/GRU should not be the primary temporal architecture.

### 2.4 Automatic Static/Dynamic Routing

A learned routing mechanism shall determine whether the incoming sign sequence should be processed through the static or dynamic recognition pathway.

The routing system should avoid relying solely on a manually selected motion threshold.

---

## 3. Multimodal Recognition

The system should use multiple available modalities where useful:

- Hand landmarks
- Arm/body landmarks
- Facial/non-manual cues
- Temporal motion information

The architecture should support multimodal feature fusion before final prediction.

---

## 4. Confidence-Aware Recognition

The recognition pipeline shall provide:

- Prediction confidence
- Top-3 candidate predictions
- Low-confidence detection
- Unknown/rejection state
- Temporal confidence smoothing
- Repeat/confirmation mechanism for uncertain predictions

The system must not force an arbitrary class prediction when confidence is insufficient.

---

## 5. Continuous Sign Recognition

The system shall support sequences containing multiple signs.

It should:

- Detect sign boundaries.
- Segment continuous signing into individual sign units.
- Recognize the resulting sign sequence.
- Suppress accidental duplicate predictions.
- Maintain prediction history.
- Construct a readable sentence from recognized signs.

This should be treated as continuous sign recognition followed by sentence reconstruction rather than claiming unrestricted natural-language ISL grammar translation.

---

## 6. Sentence Reconstruction

The system should provide a context-aware sentence construction layer.

It should support:

- Word/sign sequence assembly.
- Context-aware phrase construction.
- Basic grammatical cleanup where appropriate.
- Duplicate removal.
- Punctuation reconstruction where confidence permits.
- Manual editing of the generated sentence.
- Copy/export of the final sentence.
- Sentence-level text-to-speech.

The NLP layer must not invent unsupported information.

---

## 7. Video Upload Mode

The application shall support local video files including:

- MP4
- MOV
- AVI/WebM where supported

For uploaded videos, the system should provide:

- Video validation.
- Video quality assessment.
- Frame extraction.
- Long-video processing/chunking.
- Signer detection.
- Signer tracking.
- Multi-person detection.
- Manual signer selection when multiple people are present.
- Sign segmentation.
- Recognition.
- Timestamped sign timeline.
- Confidence per recognized sign.
- Complete transcript.
- Text-to-speech.
- Exportable recognition report.

---

## 8. Video URL Mode

The application should support processing accessible video URLs where technically and legally permitted.

Target sources include:

- YouTube
- Instagram
- Facebook
- Google Drive

The URL pipeline shall:

1. Validate the URL.
2. Determine whether the resource is accessible.
3. Attempt permitted retrieval.
4. Download/process the video when technically possible.
5. Fall back to local upload when the platform prevents programmatic access, authentication is required, or the resource is private.
6. Never claim unrestricted access to arbitrary private or protected content.

---

## 9. Video Intelligence

The video pipeline should support:

- Signer detection.
- Signer tracking.
- Multi-person handling.
- Manual signer selection.
- Camera-distance awareness.
- Occlusion detection/warnings.
- Video quality assessment.
- Background and lighting robustness checks.

---

## 10. Text and Speech Output

The application shall provide:

- Recognized sign text.
- Continuous sentence output.
- Editable transcript.
- Copy-to-clipboard functionality where supported.
- Transcript export.
- Optional text-to-speech.
- Sentence-level speech output.

---

## 11. Reverse Communication Mode

The application should include a Text-to-ISL demonstration mode.

Given a text phrase, the system should:

1. Tokenize the input.
2. Identify supported words/signs.
3. Retrieve corresponding demonstration videos from a curated local ISL dictionary.
4. Display the demonstrations in sequence.

This feature is a dictionary/demonstration system and is not intended to be a full generative ISL avatar.

---

## 12. Robustness and Generalization

The project shall evaluate performance across:

- Different signers.
- Previously unseen signers.
- Different lighting conditions.
- Different backgrounds.
- Different camera distances.
- Different clothing/background combinations where relevant.
- Partial hand/body occlusion where feasible.

Evaluation should include signer-independent testing, preferably using leave-one-signer-out or equivalent evaluation methodology.

---

## 13. Research and Evaluation Module

The project should provide research-oriented analytics including:

### Classification metrics
- Accuracy
- Precision
- Recall
- F1-score
- Per-class performance
- Confusion matrix

### Generalization analysis
- Seen vs unseen signer performance.
- Signer-wise performance.
- Environment-wise performance.
- Lighting-wise performance.
- Background-wise performance.

### Ablation studies
Where feasible, compare:
- Hand-only features.
- Hand + body features.
- Hand + body + facial/non-manual features.
- Static-only pathway.
- Dynamic-only pathway.
- Multimodal fusion.
- Router enabled vs disabled.

### System performance
Measure:
- FPS
- End-to-end latency
- Inference time
- Processing time per video
- Memory usage where feasible

### Explainability
Where feasible, provide:
- Attention visualization.
- Feature contribution analysis.
- Recognition confidence visualization.

---

## 14. User Interface

The application should provide a clean Streamlit interface with separate modes:

1. Live Webcam
2. Continuous Sentence
3. Upload Video
4. Video URL
5. Text → ISL
6. Research / Analytics

The interface should display relevant system status such as:

- Current recognition mode.
- Current prediction.
- Confidence.
- Top alternatives.
- Processing status.
- FPS.
- Latency.
- Sign history.

---

## 15. Proposed AI Architecture

High-level pipeline:

WEBCAM / VIDEO / URL
    ↓
Video Input Layer
    ↓
Video Quality Assessment
    ↓
Person Detection / Signer Tracking
    ↓
Hand + Body + Facial Feature Extraction
    ↓
Feature Engineering / Normalization
    ↓
Learned Static-Dynamic Router
    ├── Static → Graph-based Spatial Model
    └── Dynamic → ST-GCN → Transformer
    ↓
Multimodal Feature Fusion
    ↓
Confidence Estimation
    ↓
Temporal Smoothing
    ↓
Sign Boundary Detection
    ↓
Recognized Sign Stream
    ↓
Context-Aware NLP
    ↓
Sentence Reconstruction
    ↓
Text + Optional TTS
    ↓
Transcript / Timeline / Export

---

## 16. Technology Principles

The implementation should prioritize:

- Python 3.12
- PyTorch
- OpenCV
- Graph-based deep learning
- Spatial-temporal modelling
- Transformer architectures
- Streamlit
- Modular software architecture
- Reproducible experiments
- Version-controlled development

The perception layer should be selected based on practical landmark quality and compatibility rather than novelty alone.

MediaPipe Holistic may be used where it provides reliable landmark extraction, but the recognition architecture should remain differentiated through graph-based spatial modelling, spatial-temporal modelling, Transformer temporal modelling, multimodal fusion, and learned routing.

---

## 17. Data Requirements

The project should maintain clear separation between:

- Raw data
- Processed data
- Extracted features
- Training/validation/test splits
- Model checkpoints
- Evaluation results

Dataset splitting must prevent signer leakage between training and test sets when evaluating signer-independent performance.

---

## 18. Software Engineering Requirements

The project should use:

- Git
- Virtual environment
- Modular Python packages
- Configuration files
- Reproducible training scripts
- Unit tests
- Logging
- Experiment tracking where practical
- Clear documentation
- Environment variable management
- Model/version tracking

Large datasets, trained model weights, credentials, and private configuration files must not be committed to Git.

---

## 19. Scope Boundary

The project should provide a strong practical prototype without claiming capabilities that have not been experimentally validated.

The following are outside the initial core scope:

- Full unrestricted natural-language ISL grammar translation.
- Fully generative photorealistic ISL avatars.
- Specialized wearable/glove hardware.
- Medical or accessibility certification.
- Guaranteed recognition under arbitrary environmental conditions.

These may be documented as future extensions.

---

## 20. Primary Success Criteria

The project should demonstrate:

1. Real-time webcam sign recognition.
2. Static and dynamic sign recognition.
3. Automatic static/dynamic routing.
4. Multimodal feature processing.
5. Confidence-aware predictions.
6. Continuous sign sequence handling.
7. Sentence reconstruction.
8. Optional TTS.
9. Video upload processing.
10. Accessible video URL processing where permitted.
11. Signer tracking and multi-person handling.
12. Signer-independent evaluation.
13. Robustness analysis.
14. Research-grade evaluation metrics.
15. A usable Streamlit interface.
16. Reproducible experiments and documentation.

