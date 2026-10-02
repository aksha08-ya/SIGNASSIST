# SIGNASSIST — Text-to-ISL Video Asset Requirements Specification

## 1. Overview
This document specifies the exact technical requirements, container specifications, index schema, and runtime behaviors for Indian Sign Language (ISL) dictionary video assets within SIGNASSIST.

---

## 2. Technical Video Asset Specifications

All pre-rendered ISL sign demonstration video clips stored in `data/external/isl_dictionary/videos/` must comply strictly with the following technical parameters:

| Property | Requirement | Notes / Validation Rule |
|---|---|---|
| **Container Format** | `.mp4` (MPEG-4 Part 14) | Must be readable by OpenCV `cv2.VideoCapture` and HTML5 `<video>` tags |
| **Video Codec** | H.264 / AVC (`avc1` or `h264`) | Baseline or Main profile recommended |
| **Audio Codec** | AAC (Optional / Muted) | Muted audio preferred for sign clips |
| **Frame Rate (FPS)** | `30.0` FPS | Strict tolerance: `30.0` ± 1.0 FPS |
| **Resolution** | `640 x 480` (4:3) or `1280 x 720` (16:9) | Standard fixed resolution across all dictionary clips |
| **Pixel Format** | `yuv420p` | Standard color subsampling for broad browser compatibility |
| **File Naming Convention** | `lowercase_snake_case.mp4` | Examples: `hello.mp4`, `thank_you.mp4`, `good_morning.mp4` |

---

## 3. Dictionary Index Schema (`index.json`)

The dictionary catalog file MUST be located at `data/external/isl_dictionary/index.json`.

### Schema Definition
`index.json` contains a top-level JSON array of entry objects.

```json
[
  {
    "word": "hello",
    "video_path": "videos/hello.mp4",
    "category": "greetings",
    "attributes": {
      "fps": 30,
      "resolution": [640, 480]
    }
  },
  {
    "word": "thank you",
    "video_path": "videos/thank_you.mp4",
    "category": "polite",
    "attributes": {
      "fps": 30,
      "resolution": [640, 480]
    }
  }
]
```

### Field Definitions
- **`word`** *(string, required)*: The canonical lowercased English term or phrase matched against input tokens. Must contain only lowercase alphanumeric characters, spaces, or hyphens.
- **`video_path`** *(string, required)*: Relative path from `data/external/isl_dictionary/` to the physical MP4 video file.
- **`category`** *(string, optional)*: Semantic group (e.g., `greetings`, `numbers`, `questions`, `common`).
- **`attributes`** *(object, optional)*: Metadata containing `fps` (int) and `resolution` (`[width, height]`).

---

## 4. Tokenization & Parsing Rules

1. **Text Normalization**:
   - Input sentences are converted to lowercase.
   - Punctuation marks (`. , ! ? ; : ' " ( ) [ ] { } -`) are stripped from token boundaries before lookup.
   - Multiple whitespace characters are collapsed into single spaces.
2. **Phrase Matching**:
   - Multi-word entries in `index.json` (e.g., `"thank you"`, `"good morning"`) take precedence over single-word token splits.
3. **Punctuation Policy**:
   - Punctuation tokens are ignored during clip lookup and do not produce demonstration video clips.

---

## 5. Missing Token & Fallback Behaviors

In alignment with SIGNASSIST zero-fabrication rules:

> [!CAUTION]
> **No Fake Data or Simulated Videos**
> - **Zero Synthetic Clips**: Never generate blank, animated, or placeholder videos for missing words.
> - **Zero Default Videos**: Never substitute unrelated video clips for unindexed terms.

### Runtime Fallback Rules:
1. **Fully Missing Tokens**:
   - If a word is not present in `index.json` and no valid physical MP4 file exists, the system flags the word as `UNKNOWN`.
   - The UI MUST explicitly render a warning:
     `"ISL demonstration unavailable for: <word>"`
2. **Partial / Mixed Sentence Translation**:
   - For sentences containing both available and missing terms (e.g., `"Hello world"` where `"hello"` exists but `"world"` is missing):
     - The stitcher concatenates clips ONLY for available dictionary terms.
     - The UI displays the stitched video of available signs alongside explicit notification of missing tokens.
     - The system MUST NOT claim full translation completeness.

---

## 6. Required Essential ISL Vocabulary List

The dictionary should aim to cover the following core vocabulary entries:

- **Greetings & Polite Expressions**: `hello`, `good morning`, `good afternoon`, `good night`, `thank you`, `please`, `sorry`, `welcome`, `goodbye`.
- **Questions**: `what`, `where`, `when`, `who`, `why`, `how`.
- **Numbers**: `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`.
- **Common Phrases**: `yes`, `no`, `help`, `name`, `learn`, `sign`, `deaf`, `india`, `friend`.
