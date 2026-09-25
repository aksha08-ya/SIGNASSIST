# SIGNASSIST — System Architecture Specification

## 1. System Overview & High-Level Architecture

SIGNASSIST is an AI-powered Indian Sign Language (ISL) recognition and translation system designed to process static gestures, dynamic signs, continuous signing sequences, uploaded video files, and accessible video URLs. The system prioritizes **signer-independent generalization**, **spatial-temporal graph representations**, **learned static/dynamic routing**, **multimodal landmark fusion**, and **reproducible research analytics**.

### High-Level System Architecture Diagram

`mermaid
flowchart TD
    subgraph Input_Layer [Input Layer]
        WC[Webcam Stream - OpenCV Frame Ingestion]
        VU[Video File Ingestion - MP4, MOV, AVI]
        URL[Video URL Ingestion - yt-dlp / Fallback]
    end

    subgraph Perception_Layer [Perception & Video Intelligence]
        VQ[Video Quality & Occlusion Inspector]
        ST[Signer Detector & Tracker - IoU / Kalman Filter]
        MP[Multi-Person Target Selector]
        LM[Multimodal Landmark Extractor - MediaPipe Holistic]
        FE[Coordinate Normalizer & Skeleton Graph Builder]
    end

    subgraph Recognition_Engine [Dual-Pathway Recognition Engine]
        LSR[Learned Static-Dynamic Router - MLP Classifier]
        S_GNN[Static Pathway: Spatial Graph Convolutional Model]
        D_STGCN[Dynamic Pathway: Spatial-Temporal GCN ST-GCN]
        D_TRANS[Dynamic Temporal Modelling: Transformer Encoder]
        MMF[Multimodal Feature Fusion Engine]
    end

    subgraph Post_Processing [Decision & Post-Processing]
        CE[Confidence Estimator & Rejection Unit]
        BD[Sign Boundary Detector - Landmark Energy Minima]
        TS[Temporal Smoother & Duplicate Suppressor]
    end

    subgraph NLP_Translation [Translation & NLP Layer]
        SEQ[Sign Sequence Buffer]
        NLP[Context-Aware Sentence Reconstruction Engine]
    end

    subgraph Output_Presentation [Output & Reverse Mode]
        UI[Streamlit Interactive User Interface - 6 Modes]
        TTS[Text-to-Speech Engine - pyttsx3 / gTTS]
        T2ISL[Text-to-ISL Demonstration Engine]
    end

    subgraph Research_Analytics [Research & Evaluation Suite]
        MET[Classification Metrics & Confusion Matrix]
        LOSO[Signer-Independent LOSO Evaluation]
        ABL[Ablation Analysis Suite]
        PROF[FPS & Latency Profiler]
    end

    WC --> VQ
    VU --> VQ
    URL --> VQ
    VQ --> ST
    ST --> MP
    MP --> LM
    LM --> FE
    FE --> LSR
    LSR -- Static --> S_GNN
    LSR -- Dynamic --> D_STGCN
    D_STGCN --> D_TRANS
    S_GNN --> MMF
    D_TRANS --> MMF
    MMF --> CE
    CE --> BD
    BD --> TS
    TS --> SEQ
    SEQ --> NLP
    NLP --> UI
    NLP --> TTS
    UI --> T2ISL

    FE -. Metrics .-> MET
    MMF -. Performance .-> PROF
    NLP -. Evaluation .-> LOSO
    LSR -. Ablation .-> ABL
`

---

## 2. Component & Module Responsibilities

The system is organized into decoupled Python packages under src/:

| Module Path | Primary Responsibility | Key Classes / Components |
| :--- | :--- | :--- |
| src/video/ | Video capture, frame extraction, URL download, quality assessment, signer tracking. | WebcamStream, VideoFileReader, URLVideoDownloader, SignerTracker, QualityInspector |
| src/data/ | Landmark extraction, skeleton graph construction, coordinate normalization, dataset splitters. | HolisticLandmarkExtractor, GraphBuilder, CoordinateNormalizer, SignerLOSODataset |
| src/features/ | Spatial-temporal feature engineering, motion energy computation, velocity extraction. | KinematicFeatureExtractor, MotionEnergyCalculator, FacialFeatureAggregator |
| src/models/ | Neural network architectures (Static GNN, Dynamic ST-GCN, Transformer, Learned Router, Fusion Classifier). | StaticGraphGCN, STGCNBackbone, TransformerTemporalEncoder, StaticDynamicRouter, MultimodalFusionClassifier |
| src/pipeline/ | End-to-end inference coordination, confidence estimation, temporal smoothing, boundary detection. | InferencePipeline, ConfidenceEstimator, TemporalSmoother, SignBoundaryDetector |
| src/nlp/ | Sign sequence assembly, context-aware sentence construction, grammar cleanup, transcript management. | SentenceReconstructor, GrammarRuleEngine, TranscriptBuffer |
| src/tts/ | Text-to-speech synthesis (offline pyttsx3 fallback to online gTTS). | TTSEngine, AudioPlayer |
| src/utils/ | System configuration, logging, export generators (PDF/JSON), metrics profilers. | ConfigManager, Logger, ReportExporter, PerformanceProfiler |

---

## 3. Data Flow Specification

### 3.1 Frame Processing Data Flow
1. **Frame Ingestion:** Raw frame  \in \mathbb{R}^{H 	imes W 	imes 3}$ captured at 30 FPS from Webcam, Upload, or Downloaded URL stream.
2. **Quality Check:** Blur detection (Laplacian variance), illumination analysis, and camera-distance score computed.
3. **Signer Detection & Tracking:** Person bounding boxes extracted; target signer tracked using Kalman Filter + IoU bounding-box tracking.
4. **Landmark Extraction:** Selected signer ROI passed to MediaPipe Holistic, outputting raw landmark tensors for Left Hand (21x3), Right Hand (21x3), Body Pose (33x4), and Key Face Landmarks (40x3).
5. **Normalization & Graph Construction:** Landmarks translated relative to shoulder-center / wrist, scaled to unit body height, yielding node tensor  \in \mathbb{R}^{V 	imes C}$ (=115$ nodes, =5$ features: , y, z, v_x, v_y$).
6. **Sliding Window Buffer:** $ consecutive normalized frames stacked into sequence tensor $\mathbf{X} \in \mathbb{R}^{B 	imes C 	imes T 	imes V}$.
7. **Routing & Inference:** StaticDynamicRouter evaluates kinematic energy {\text{motion}}$. If static ({\text{static}} > 0.5$), single-frame spatial GCN runs; if dynamic, ST-GCN + Transformer pipeline runs.
8. **Fusion & Confidence:** Multimodal Fusion Classifier combines static, dynamic, and non-manual features into prediction logits $\mathbf{z} \in \mathbb{R}^{N_{\text{classes}}}$.
9. **Decision Logic:** Softmax probabilities evaluated by ConfidenceEstimator. Low confidence triggers UNKNOWN state.
10. **Smoothing & Boundary Detection:** Predictions smoothed via sliding window majority voting; sign boundaries detected at local motion energy minima.
11. **Sentence Assembly & Output:** Recognized sign appended to sequence buffer -> SentenceReconstructor builds sentence -> Streamlit UI updates -> Optional TTS audio generated.

---

## 4. Perception & Landmark Feature Pipeline

### 4.1 Topology & Node Definition
The system represents the human signer as a spatial-temporal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with  = 115$ total keypoint nodes:

* **Left Hand ($\mathcal{V}_{\text{LH}}$):** 21 3D landmarks (wrist, thumb, index, middle, ring, pinky joints).
* **Right Hand ($\mathcal{V}_{\text{RH}}$):** 21 3D landmarks.
* **Body Pose ($\mathcal{V}_{\text{Pose}}$):** 33 landmarks (shoulders, elbows, wrists, hips, trunk, head anchor).
* **Facial Non-Manual Cues ($\mathcal{V}_{\text{Face}}$):** 40 selected key facial contour points (eyebrows, eyes, outer lip contour, jawline) critical for ISL grammatical markers.

### 4.2 Spatial Graph Construction
The adjacency matrix $\mathbf{A} \in \mathbb{R}^{V 	imes V}$ is partitioned into three structural edge types:
1. **Physical Anatomical Connections ($\mathbf{A}_{\text{physical}}$):** Skeletal bones connecting adjacent joints (e.g., wrist to palm, elbow to shoulder).
2. **Inter-Modal Anchors ($\mathbf{A}_{\text{anchor}}$):** Structural links connecting wrist landmarks to pose wrist keypoints, and nose/chin landmarks to facial contour points.
3. **Self-Loops ($\mathbf{A}_{\text{self}}$):** Identity matrix $\mathbf{I}_V$ supporting self-feature retention.

\mathbf{A}_{\text{normalized}} = \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}, \quad \text{where } \mathbf{\tilde{A}} = \mathbf{A}_{\text{physical}} + \mathbf{A}_{\text{anchor}} + \mathbf{I}_V

### 4.3 Feature Normalization & Invariance Transformations
To ensure signer-independent and distance-invariant recognition:
* **Translation Invariance:** Coordinates centered relative to mid-shoulder midpoint $\mathbf{p}_{\text{mid\_shoulder}} = \frac{1}{2}(\mathbf{p}_{\text{left\_shoulder}} + \mathbf{p}_{\text{right\_shoulder}})$.
* **Scale Invariance:** All joint distances normalized by torso height {\text{torso}} = \|\mathbf{p}_{\text{mid\_shoulder}} - \mathbf{p}_{\text{mid\_hip}}\|_2$.
* **Hand-Local Origin:** Hand keypoints additionally offset relative to respective wrist keypoint $\mathbf{p}_{\text{wrist}}$ to maintain micro-gesture accuracy.
* **Kinematic Derivatives:** First-order temporal velocity calculated across adjacent frames:

\mathbf{v}_i^{(t)} = \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(t-1)}

---

## 5. Dual-Pathway Recognition Architecture

### 5.1 Learned Static-Dynamic Router
Rather than relying on manual, brittle motion thresholding, SIGNASSIST employs a learned MLP classifier to route sequence windows.

* **Feature Vector f_router in R^12:**
  1. Mean spatial hand displacement velocity v_hand.
  2. Max hand velocity v_max.
  3. Spectral motion variance sigma^2_motion across frame window T.
  4. Bounding box expansion/contraction rate.
  5. Hand-to-body relative distance delta.
* **Network Architecture:**
  * Linear(12, 64) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(64, 32) -> ReLU -> Linear(32, 2) -> Softmax
* **Output:** Router distribution [p_static, p_dynamic]. If p_static > 0.5, route to Static Pathway; else Dynamic Pathway.

`mermaid
graph LR
    InputTensor[Normalized Landmark Sequence B x 5 x T x 115] --> FeatureExtractor[Kinematic Feature Extractor - Velocities and Variance]
    FeatureExtractor --> RouterMLP[Learned Router MLP 64-32-2]
    RouterMLP --> SoftmaxChoice{Is p_static > 0.5?}
    SoftmaxChoice -- Yes --> StaticGNN[Spatial Graph Model - Static Path]
    SoftmaxChoice -- No --> DynamicSTGCN[ST-GCN + Transformer - Dynamic Path]
`

### 5.2 Static-Sign Recognition Pathway
Recognizes stationary alphabet signs (e.g., ISL 'A', 'B', '1', '2') from keyframe spatial topologies.

* **Architecture:** 3-layer Spatial Graph Convolutional Network (GCN).
* **Input Tensor Shape:** [B, C, V] where C=5, V=115 (extracted from keyframe t).
* **Layer Design:**
  * H^(1) = ReLU(BatchNorm1d(A_norm X W^(1))) (Channels: 5 -> 64)
  * H^(2) = ReLU(BatchNorm1d(A_norm H^(1) W^(2))) (Channels: 64 -> 128)
  * H^(3) = ReLU(BatchNorm1d(A_norm H^(2) W^(3))) (Channels: 128 -> 256)
* **Global Pooling:** Global Spatial Mean Pooling -> z_static in R^(B x 256).

### 5.3 Dynamic-Sign Recognition Pathway (ST-GCN + Transformer)
Designed for movement-based gestures requiring joint spatio-temporal reasoning. **LSTM and GRU are explicitly excluded as primary temporal architectures.**

* **Spatial-Temporal Graph Convolution (ST-GCN):**
  * **Input Tensor Shape:** [B, C, T, V] = [B, 5, 30, 115]
  * **ST-GCN Block:** Spatial graph convolution across V joints combined with 1D temporal convolution across T frames (kernel size K_t=9).
  * **Layer Stack:** 4 cascaded ST-GCN blocks with channels 5 -> 64 -> 128 -> 256.
  * **Output Shape:** Spatial feature maps pooled across joints V -> F_stgcn in R^(B x T x 256).

* **Transformer Temporal Encoder:**
  * **Input:** ST-GCN temporal sequence F_stgcn in R^(B x T x 256).
  * **Positional Encoding:** Sinusoidal Positional Encoding added to preserve frame order.
  * **Encoder Architecture:** 2-layer Transformer Encoder.
    * d_model = 256
    *
head = 8
    * dim_feedforward = 512
    * dropout = 0.1
  * **Temporal Aggregation:** Multi-head attention pooling across time dimension T -> z_dynamic in R^(B x 256).

### 5.4 Multimodal Feature Fusion Engine
Combines spatial hand embeddings, dynamic ST-GCN representations, and non-manual facial features.

z_fused = LayerNorm(W_s * z_static + W_d * z_dynamic + W_f * z_face)

* **Classification Head:** Linear Classifier Linear(256, N_classes) producing raw logits z_logits in R^(B x N_classes).

---

## 6. Decision, Rejection & Continuous Recognition

### 6.1 Confidence Estimation & Rejection Mechanism
To prevent false positive classifications on random movements or un-signed gestures:

* **Softmax Probability:** p_k = exp(z_k) / sum_j exp(z_j)
* **Prediction Margin:** delta_p = p_(1) - p_(2) (difference between top-1 and top-2 candidates).
* **Entropy Score:** H(p) = - sum_k p_k * log(p_k).
* **Rejection Logic:**

Predicted Class C = argmax p_k if p_(1) >= 0.65 AND delta_p >= 0.15 else UNKNOWN.

Default Thresholds: theta_conf = 0.65, theta_margin = 0.15. The system presents the top-3 candidate predictions with confidence percentages whenever C != UNKNOWN.

### 6.2 Temporal Smoothing & Duplicate Suppression
* **Sliding Window Majority Vote:** A queue of size W_smooth = 5 frames buffers predictions. The dominant class is returned if it appears in at least 3 frames.
* **Debounce Hysteresis:** A recognized sign must remain stable for N_debounce = 3 frames before triggering a new emission.
* **Duplicate Suppressor:** Identical consecutive signs emitted within T_cooldown = 1.2 seconds are suppressed unless separated by an explicit rest pose or boundary.

### 6.3 Sign Boundary Detection
Detects transitions between signs in continuous streams without manual segmentation:

1. **Motion Energy Minima:** Computes total kinetic energy E(t) = sum_v ||v_v^(t)||^2.
2. **Zero-Crossing of Derivative:** Sign boundaries detected where dE/dt == 0 and d^2E/dt^2 > 0 (local energy minima representing brief pauses between signs).
3. **Rest-Pose Classifier:** Detects neutral hand position (hands lowered near waist level).

`mermaid
graph TD
    Energy[Kinematic Motion Energy E_t] --> Deriv[Compute Energy Derivative dE/dt]
    Deriv --> CheckMinima{Is dE/dt == 0 and Local Minima?}
    CheckMinima -- Yes --> BoundaryDetected[Sign Boundary Triggered: Emit Segment]
    CheckMinima -- No --> ContinueWindow[Accumulate Frame into Sliding Window Buffer]
`

---

## 7. NLP & Sentence Reconstruction Architecture

SIGNASSIST distinguishes between **continuous sign token recognition** and **context-aware sentence reconstruction**. It does not claim unrestricted natural-language ISL translation; instead, it uses a rule-based + phrase-matching NLP engine.

`mermaid
flowchart LR
    TokenStream[Raw Sign Sequence Buffer e.g. ME GO SCHOOL TOMORROW] --> DupFilter[Duplicate & Rest Suppressor]
    DupFilter --> PhraseMatcher[ISL Phrase & Grammar Cleanup Rule Engine]
    PhraseMatcher --> ContextBuilder[Context & Punctuation Reconstruction]
    ContextBuilder --> OutputSentence[Reconstructed Sentence: I will go to school tomorrow]
`

### 7.1 NLP Processing Steps
1. **Sequence Buffer Assembly:** Collects incoming recognized sign tokens [s_1, s_2, ..., s_m].
2. **Duplicate & Noise Removal:** Strips accidental double hits and removes UNKNOWN tokens.
3. **ISL Grammar to English Mapping:** Applies deterministic transformation rules:
   * Subject-Object-Verb (ISL) to Subject-Verb-Object (English) mapping rules.
   * Pronoun normalization (e.g., ME -> I, GO + TOMORROW -> will go).
   * Tense & auxiliary verb insertion based on time markers (YESTERDAY, TOMORROW, NOW).
4. **Punctuation & Capitalization:** Adds trailing question marks for interrogative signs (WHAT, WHERE, WHY) and capitalizes sentence starts.
5. **Editing Hooks:** Exposes raw token buffer and generated sentence to Streamlit UI, allowing manual user edits before export or TTS playback.

---

## 8. Video Intelligence & Ingestion Pipelines

### 8.1 Video File Upload Pipeline
* **Supported Formats:** MP4, MOV, AVI, WebM.
* **Validation Subsystem:** Checks container integrity, codec compatibility, frame rate, and resolution (720p / 1080p recommended).
* **Chunking Manager:** For videos > 30 seconds, frames are split into 1-second overlapping temporal chunks for parallel landmark processing.
* **Output Artifacts:** Interactive sign timeline, timestamped confidence table, downloadable transcript (.txt), and exportable summary report (.json / .pdf).

### 8.2 Accessible Video URL Pipeline & Graceful Fallback
* **Engine:** Integrated yt-dlp wrapper supporting YouTube, Instagram, Facebook, and Google Drive video links.
* **Access Validation Protocol:**
  1. Validates URL syntax and platform domain.
  2. Queries video metadata (duration, format, accessibility permissions) without full download.
  3. Checks authentication restrictions, DRM, and geo-blocking.
* **Graceful Fallback Logic:** If the platform blocks programmatic access or requires login, the system halts URL retrieval safely, logs an URLExtractionError, and presents a clear user notification requesting a direct file upload.

`mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant Pipe as URL Ingestion Pipeline
    participant YTDLP as yt-dlp Core
    participant Ext as External URL YouTube/IG/FB

    UI->>Pipe: Submit Video URL
    Pipe->>Pipe: Validate URL Syntax
    Pipe->>YTDLP: Query Metadata & Access Status
    YTDLP->>Ext: Check Permissions & Stream Info
    alt URL Accessible & Public
        Ext-->>YTDLP: Stream Data OK
        YTDLP-->>Pipe: Download Temporary MP4
        Pipe-->>UI: Processing Video Stream...
    else Private / Auth Required / Geo-Blocked
        Ext-->>YTDLP: Access Denied 403 / Login Req
        YTDLP-->>Pipe: Extraction Failed
        Pipe-->>UI: Display Fallback Notice: URL restricted. Please upload file locally.
    end
`

### 8.3 Signer Detection, Tracking & Multi-Person Handling
* **Detection:** OpenCV DNN / MediaPipe Pose detector identifies all human bounding boxes B_k = [x1, y1, x2, y2, score].
* **Tracking:** IoU matching + Kalman Filter maintains stable track IDs (T_id) across frames.
* **Multi-Person Selection:**
  * **Auto Mode:** System tracks the primary signer based on bounding box size and central camera alignment.
  * **Manual Mode:** When multiple people are detected, the UI overlays numbered bounding boxes, allowing the user to select target Track ID = k.

---

## 9. Reverse Communication & Speech Subsystems

### 9.1 Text-to-ISL Demonstration Architecture
Converts input English text into sequential ISL sign video demonstrations.

1. **Text Normalizer & Tokenizer:** Cleans input string, removes punctuation, converts to uppercase tokens.
2. **Dictionary Mapper:** Queries curated local ISL video dictionary (data/external/isl_dictionary/).
3. **Video Segment Stitcher:** Retrieves pre-recorded MP4 clips for each recognized word. For missing words, falls back to fingerspelling static alphabet clips.
4. **Playback Synthesizer:** Chains video clips into a continuous OpenCV / Streamlit video player sequence.

### 9.2 Text-to-Speech (TTS) Architecture
Provides voice output for recognized signs and constructed sentences.

* **Primary Offline Engine:** pyttsx3 (SAPI5 on Windows / espeak on Linux) for zero-latency offline synthesis.
* **Online Fallback Engine:** gTTS (Google Text-to-Speech) for higher voice naturalness when internet connectivity is available.
* **Control API:** Supports play, pause, stop, pitch, rate adjustment, and sentence-level audio export (.mp3 / .wav).

---

## 10. User Interface Architecture (Streamlit)

The Streamlit UI is structured as a single entry point (src/app.py) managing 6 dedicated application views:

`mermaid
graph TD
    App[src/app.py - Streamlit Entry Point] --> Mode1[1. Live Webcam Mode]
    App --> Mode2[2. Continuous Sentence Mode]
    App --> Mode3[3. Upload Video Mode]
    App --> Mode4[4. Video URL Mode]
    App --> Mode5[5. Text -> ISL Demo Mode]
    App --> Mode6[6. Research & Analytics Mode]
`

### UI View Details:
1. **Live Webcam Mode:** Real-time video preview, active landmark overlay toggle, live prediction box, top-3 candidates bar chart, FPS/latency overlay, TTS trigger button.
2. **Continuous Sentence Mode:** Real-time sign sequence stream, sentence assembly box with manual edit capability, transcript history, copy-to-clipboard, audio playback.
3. **Upload Video Mode:** Drag-and-drop file uploader (MP4/MOV/AVI), video player, signer selection dropdown (if multi-person), processing progress bar, timestamped sign timeline table, transcript PDF export.
4. **Video URL Mode:** URL text input field, platform detector badge, download progress bar, graceful fallback notification banner, processed video player with timeline.
5. **Text -> ISL Mode:** English text prompt input, word token breakdown display, ISL clip sequence player, playback speed control.
6. **Research / Analytics Mode:** Confusion matrix renderer, per-class F1-score breakdown, Seen vs. Unseen Signer generalization plot, Ablation study toggle grid (Modality/Router impact), FPS/latency breakdown chart, Attention heatmap visualizer.

---

## 11. Research, Evaluation & Ablation Architecture

Designed to support academic research standards and rigorous empirical validation.

### 11.1 Evaluation Metrics Engine
Computes comprehensive classification and performance metrics:
* **Accuracy, Precision, Recall, F1-Score:** Overall macro-averaged and per-class breakdowns.
* **Confusion Matrix:** Normalized matrix saved to
reports/figures/confusion_matrix.png.
* **Latency Profiling:** Per-frame breakdown: Ingestion (t_ingest), Landmark Extraction (t_lm), Router (t_router), Neural Model Inference (t_inf), Post-processing (t_post).

### 11.2 Leave-One-Signer-Out (LOSO) Cross-Validation
Prevents dataset leakage by ensuring test splits contain signers never seen during training:

Train Split = Union of Signer_j (j != k), Test Split = Signer_k.

Evaluates and logs **Seen Signer Accuracy** vs. **Unseen Signer Accuracy** to quantify generalization.

### 11.3 Ablation Study Framework
Systematically evaluates component contributions by toggling sub-modules in configs/ablation.yaml:

| Experiment ID | Hand | Body | Face | Static Path | Dynamic Path | Router Enabled |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP_HAND_ONLY | Yes | No | No | Yes | Yes | Yes |
| EXP_HAND_BODY | Yes | Yes | No | Yes | Yes | Yes |
| EXP_FULL_MULTIMODAL | Yes | Yes | Yes | Yes | Yes | Yes |
| EXP_NO_ROUTER | Yes | Yes | Yes | Yes | Yes | No |
| EXP_DYNAMIC_ONLY | Yes | Yes | Yes | No | Yes | No |

---

## 12. Non-Functional Specifications & Constraints

### 12.1 Hardware Abstraction & CPU Fallback
* **Device Selection:** PyTorch device manager automatically targets CUDA GPU if available, smoothly defaulting to CPU (	orch.device('cuda' if torch.cuda.is_available() else 'cpu')).
* **ONNX Runtime Export:** Models exportable to ONNX format for accelerated CPU inference on standard laptops without GPU hardware.
* **Latency Target:** < 40 ms per frame (> 25 FPS) on standard Intel i7 / Ryzen 7 CPUs.

### 12.2 Configuration Strategy
All system parameters are strictly externalized in YAML files under configs/:
* configs/default.yaml: Camera index, video quality thresholds, logging level.
* configs/model.yaml: GCN channels, ST-GCN layer depths, Transformer head counts, learning rates, loss functions.
* configs/ui.yaml: Streamlit theme colors, layout options, default mode selection.

### 12.3 Error-Handling Strategy & Logging
* **Custom Exception Taxonomy:**
  * SignAssistException (Base exception class)
    * VideoProcessingError (Corrupt video, frame read failure)
    * URLExtractionError (Inaccessible link, 403 Forbidden, yt-dlp failure)
    * LandmarkExtractionError (No person detected in ROI)
    * ModelInferenceError (Tensor shape mismatch, missing checkpoint)
* **Structured Logging:** Python logging module writes structured output to
reports/logs/signassist.log with automatic log rotation (10 MB per file, max 5 backups).

### 12.4 Software Testing Strategy
* **Unit Testing (tests/unit/):** Tests landmark normalizer, graph adjacency builder, router MLP, confidence thresholding, duplicate suppressor.
* **Integration Testing (tests/integration/):** Evaluates end-to-end pipeline using synthetic/mock landmark sequence tensors without requiring live webcam hardware.
* **UI Verification:** Verification of Streamlit app views using Playwright CLI / Streamlit testing framework.

---

## 13. Technology Stack Summary & Rationale

| Layer | Primary Technology | Selection Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Modern syntax, robust typing, deep learning ecosystem support. |
| **Deep Learning** | PyTorch 2.x | Native graph neural network flexibility, TorchScript / ONNX export. |
| **Perception** | MediaPipe Holistic & OpenCV | High-speed, CPU-friendly 3D landmark extraction for hands, pose, face. |
| **Web UI** | Streamlit | Rapid, Python-native interactive UI deployment for multi-page research demos. |
| **URL Ingestion** | yt-dlp | Industry standard for multi-platform public video retrieval. |
| **Speech** | pyttsx3 & gTTS | Lightweight offline synthesis with fallback to high-quality online TTS. |
| **Testing** | pytest | Standard Python unit and integration test framework. |
