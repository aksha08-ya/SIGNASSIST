# SIGNASSIST — Next-Generation Multimodal Video Interpretation & AI Grounding

**Document Version**: 1.0.0  
**Date**: October 2, 2026  
**System Architecture**: Hybrid Local ST-GCN + Cloud Gemini Video Intelligence  

---

## 1. Executive Overview

SIGNASSIST integrates a shared **Video Intelligence Service** ([`src/ai/video_intelligence.py`](file:///home/akshaya/SIGNASSIST/src/ai/video_intelligence.py)) that combines local 42-class controlled Spatio-Temporal Graph Convolutional Network (ST-GCN) recognition with cloud-assisted multimodal AI video analysis (Google GenAI SDK `gemini-3.8-flash`) and Google Search Grounding.

This architecture explicitly distinguishes between:
1. **Local ST-GCN Recognition**: High-speed, calibrated ($T=0.68$), controlled-vocabulary ($N=42$) gesture recognition.
2. **AI-Assisted Multimodal Interpretation**: Structural video analysis, phrase boundary detection, and visual description using `gemini-3.8-flash`.
3. **Research Grounding & Evidence**: Google Search grounding references to canonical Indian Sign Language (ISL) resources.
4. **Final Interpretation**: Marked explicitly as `[AI-assisted / unverified]`.

---

## 2. Evidence & Processing Flow

```
                     ┌─────────────────────────────────────────┐
                     │           Input Video Source            │
                     │ (File Upload / Video URL / Live Session)│
                     └────────────────────┬────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 │                                                 │
                 ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│     Local SIGNASSIST Model      │               │     Video Intelligence Service    │
│  (Controlled ST-GCN 42-Class)   │               │ (Google GenAI gemini-3.8-flash) │
└────────────────┬────────────────┘               └────────────────┬────────────────┘
                 │                                                 │
                 │ Calibrated Scores (T=0.68)                      │ Gemini Files API Upload
                 │ Threshold Policy (>=0.70)                       │ Search Grounding Metadata
                 ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│  Section A: Local ST-GCN Output │               │   Section B & C: AI Analysis    │
│ (Controlled Vocabulary Warning) │               │   & Search Grounding Sources    │
└────────────────┬────────────────┘               └────────────────┬────────────────┘
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Section D: Combined Meaning   │
                         │    [AI-assisted / unverified]   │
                         └─────────────────────────────────┘
```

---

## 3. Core Architecture Components

### A. Video Intelligence Service (`src/ai/video_intelligence.py`)
- **SDK**: Official Google GenAI Python SDK (`google-genai`).
- **Authentication**: `GEMINI_API_KEY` environment variable.
- **Graceful Fallback**: If `GEMINI_API_KEY` is missing, the service returns `status="AI_ANALYSIS_UNAVAILABLE"` without application failure. Local 42-class recognition continues to function normally.
- **SHA-256 Response Caching**: Responses are cached locally under `data/processed/ai_video_cache/<sha256_hash>_<model>_<version>.json` to eliminate redundant API billing and latency for identical video files.

### B. Strict ISL Prompting & Schema Validation
The service uses a strict, non-hallucinating research prompt requiring the model to:
1. First verify if Indian Sign Language (ISL) is visually present.
2. Provide per-segment start/end timestamps, visual gesture descriptions, and likely ISL glosses.
3. Explicitly report spoken audio transcripts and visible subtitles separately from signing evidence.
4. Mark ambiguous or unknown signs as `UNKNOWN/UNCERTAIN`.
5. Return valid structured JSON.

### C. Research Grounding & Web Evidence
- Uses `types.Tool(google_search=types.GoogleSearch())` to retrieve canonical ISL dictionaries and academic research references.
- Surface Web Grounding sources with source titles, URLs, relevance notes, and support levels ("Contextual evidence only").

---

## 4. UI Presentation Rules

In accordance with strict SIGNASSIST presentation policies:
- Local 42-class model predictions are displayed under Section A with an explicit warning:
  > *"WARNING: This is a controlled-vocabulary prediction and does not establish that the video means this phrase in continuous ISL."*
- AI video analysis results are displayed under Section B and C.
- The final combined sentence interpretation is tagged with the provenance `[AI-assisted / unverified]` and accompanied by a mandatory human-in-the-loop disclaimer.

---

## 5. Security & Privacy Controls

- **API Keys**: Secrets are loaded strictly from `os.getenv("GEMINI_API_KEY")` and never logged or serialized to UI artifacts.
- **Data Isolation**: Video files submitted to Gemini Files API are explicitly deleted upon completion of content generation.
- **Data Leakage Guarantee**: 0% train/validation/test set leakage is maintained across local dictionary assets.
