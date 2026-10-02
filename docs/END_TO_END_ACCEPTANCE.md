# SIGNASSIST — End-to-End Acceptance Specification

## 1. Overview & Core Philosophy
This document establishes strict acceptance criteria for all user-facing features in the SIGNASSIST platform. 
In alignment with project rules:
- **No Mocking / Placeholder Behavior**: Every feature must process real inputs through real pipelines and output real, verified data.
- **Strict Confidence Enforcement**: The `0.70` confidence threshold is non-negotiable. Predictions below 0.70 must be treated as `Unknown / Low Confidence` and rejected from sentence formation.
- **Verification Requirement**: A feature is accepted ONLY when end-to-end processing completes with clean execution logs and correct output under real test inputs.

---

## 2. Feature Acceptance Specifications

### Feature 1: Live Webcam Sign Recognition
* **Input**: Real-time webcam video stream (30 FPS, RGB frames via MediaPipe Holistic).
* **Processing**:
  1. Real-time keypoint extraction (Pose, Face, Hands: 115 landmarks, 5 channels `[x, y, z, visibility, presence]`).
  2. Frame buffer normalization and temporal slicing (T <= 60).
  3. Inference via `RecognitionPipeline` using `DynamicSignClassifier`.
* **Expected Output**:
  * If confidence >= 0.70: Predicted gloss name, confidence percentage, top-3 candidates, and keypoint overlay.
  * If confidence < 0.70: "Unknown / Unrecognized Sign" notification, candidate probabilities muted, no gloss added to sentence.
* **Acceptance Standard**: Zero UI crashes during 60-second webcam session; accurate detection of supported signs; strict rejection when sign confidence is < 0.70.

### Feature 2: Continuous Signing & Sentence Translation
* **Input**: Continuous sequence of signed gestures over time.
* **Processing**:
  1. Sliding temporal window keypoint buffer (T=60, step=15 frames).
  2. Sequential frame classification & temporal filtering (repetition suppression).
  3. Real-time sentence reconstruction using grammar rules and transition smoothing.
* **Expected Output**:
  * Array of detected high-confidence glosses (>= 0.70).
  * Synthesized English sentence (e.g., `["YOU", "GO", "WHERE"]` -> `"Where are you going?"`).
* **Acceptance Standard**: No duplicate gloss flooding from static pose; valid English output; accurate sentence building without manual user correction for valid gesture streams.

### Feature 3: Local Video File Upload Recognition
* **Input**: User-uploaded MP4 / AVI / MOV video file containing ISL signing.
* **Processing**:
  1. OpenCV video decoding and frame rate normalization (target 30 FPS).
  2. Full-sequence MediaPipe Holistic landmark extraction.
  3. Pipeline execution with `DynamicSignClassifier`.
* **Expected Output**:
  * Playable processed video preview with skeleton overlay.
  * Timeline breakdown of recognized signs with timestamps.
  * Top-1 and Top-3 predictions with confidence scores.
* **Acceptance Standard**: Successful processing of 1080p and 720p MP4 files; correct skeleton visualization; accurate sign identification or explicit rejection if confidence < 0.70.

### Feature 4: External Video URL Recognition
* **Input**: Valid HTTP/HTTPS public video URL (e.g., Instagram reel, YouTube short).
* **Processing**:
  1. Video downloading to temporary cache via `yt-dlp` / HTTP stream handler.
  2. Video frame extraction and resolution normalization.
  3. Holistic landmark extraction and ST-GCN pipeline execution.
* **Expected Output**:
  * Download progress and frame count indicator.
  * Recognized glosses with confidence breakdown.
  * Clear error handling for unsupported formats or broken links.
* **Acceptance Standard**: Handled end-to-end without crashing; invalid URLs cleanly rejected; downloaded videos successfully parsed into skeleton tensors.

### Feature 5: Text-to-ISL Translation & Video Synthesis
* **Input**: English text input string (e.g., `"Good morning"`).
* **Processing**:
  1. `TextToISLConverter` parses input text into ordered ISL gloss tokens.
  2. `ISLDictionaryIndex` resolves tokens to pre-rendered ISL video files (`data/external/isl_dictionary/videos/*.mp4`).
  3. `ISLVideoStitcher` concatenates video clips with smooth crossfade transitions into a single MP4 video asset.
* **Expected Output**:
  * Rendered high-definition MP4 video playing the corresponding ISL sign sequence in order.
  * Detailed breakdown showing mapped gloss tokens and dictionary lookup status.
* **Acceptance Standard**: Seamless playback of concatenated ISL videos; graceful fallback/warning for out-of-vocabulary words.

### Feature 6: Text-to-Speech (TTS) Audio Generation
* **Input**: Synthesized English sentence string from recognition pipeline.
* **Processing**:
  1. TTS engine (`pyttsx3` / `gTTS`) synthesizes audio buffer.
  2. Audio file rendered to standard HTML5 audio block in Streamlit UI.
* **Expected Output**:
  * Clear, audible speech playback corresponding exactly to the recognized text.
* **Acceptance Standard**: Immediate playback on button click; correct audio waveform generation with 0 latency errors.

### Feature 7: Low-Confidence / Unknown Sign Handling
* **Input**: Skeletons or frames representing gestures outside the trained model distribution or with ambiguous motion.
* **Processing**:
  1. Pipeline evaluates `max(softmax(logits))`.
  2. Evaluates threshold logic: `is_accepted = (top_1_prob >= 0.70)`.
* **Expected Output**:
  * UI explicitly shows "Sign Not Recognized / Confidence Too Low".
  * Raw top predictions displayed only under expandable debug mode.
  * Sentence accumulator ignores unaccepted predictions.
* **Acceptance Standard**: Zero low-confidence (< 0.70) predictions accepted as valid glosses.

### Feature 8: Session Export & History
* **Input**: User trigger to export session log.
* **Processing**:
  1. Session history data formatted as JSON / CSV / Text summary.
* **Expected Output**:
  * Downloadable structured file containing timestamps, recognized glosses, confidence scores, and synthesized sentences.
* **Acceptance Standard**: File generated cleanly and downloadable via browser interface.
