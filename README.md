# SIGNASSIST 🤟
### Real-Time Indian Sign Language (ISL) Recognition & Translation System

SIGNASSIST is a modular, high-accuracy computer vision and deep learning system designed for real-time translation of Indian Sign Language (ISL) gestures into text and natural spoken audio.

---

## 🌟 Architecture Overview

SIGNASSIST utilizes a dual-stream recognition pipeline combining static (alphabet/digit) and dynamic (continuous gesture) models with automatic learned routing:

- **Landmark Extractor**: 543 3D landmarks extracted frame-by-frame via MediaPipe Holistic (Face, Pose, Hands).
- **Learned Router**: Motion energy & spatial variation analyzer routing frames to Static or Dynamic pathways.
- **Static Classifier**: Graph Neural Network (GNN) for isolated static signs (A-Z, 0-9).
- **Dynamic Classifier**: Spatial-Temporal Graph Convolutional Network (ST-GCN) + Temporal Transformer for continuous sequence modelling.
- **Post-Processor**: Temporal smoothing, duplicate suppression, confidence scoring, and sentence reconstruction with Text-to-Speech (TTS).

For detailed system design, refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md).

---

## 📁 Repository Structure

```
SIGNASSIST/
├── configs/                 # YAML configuration files
│   ├── default.yaml         # Core system, paths, video, & logging settings
│   ├── model.yaml           # Model hyperparameters & routing thresholds
│   └── ui.yaml              # Streamlit interface & display defaults
├── data/                    # Dataset storage (raw, processed, landmarks)
├── docs/                    # Architectural & specifications documentation
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT_PLAN.md
│   └── PROJECT_SPEC.md
├── models/                  # Checkpoints & serialized model files
├── notebooks/               # Research & exploration notebooks
├── reports/                 # Evaluation reports & log outputs
│   └── logs/
├── scripts/                 # Utility & execution scripts
├── src/                     # Core application source package
│   ├── data/                # Data loading & preprocessing pipelines
│   ├── features/            # MediaPipe extraction & landmark processing
│   ├── models/              # PyTorch model definitions (ST-GCN, Transformer, MLP)
│   ├── nlp/                 # Grammatical reconstruction & smoothing
│   ├── pipeline/            # End-to-end inference orchestrator
│   ├── tts/                 # Text-To-Speech integration
│   ├── utils/               # Configuration, logging, and custom exceptions
│   └── video/               # Frame capture, stream handling, and video utils
├── tests/                   # Unit and integration tests
├── .env.example             # Environment variable template
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🚀 Phase 1 Infrastructure & Setup

### Environment Setup (WSL / Ubuntu)

1. Clone or navigate to the repository in WSL:
   ```bash
   cd /home/akshaya/SIGNASSIST
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional):
   ```bash
   cp .env.example .env
   ```

---

## 🧪 Running Tests & Smoke Check

Run all unit tests using `pytest`:

```bash
pytest
```

Run infrastructure smoke check:

```bash
python -c "from src.utils.config import ConfigManager; from src.utils.logger import setup_logger; logger = setup_logger(); cfg = ConfigManager().load_all(); logger.info(f'Loaded {cfg.system.name} v{cfg.system.version} successfully!')"
```

---

## 📋 Development Roadmap

- **Phase 1: Environment & Infrastructure Setup** *(Completed)*
- **Phase 2: Landmark Processing & Feature Extraction**
- **Phase 3: Dataset Preparation & Preprocessing Pipeline**
- **Phase 4: Static-Sign Classifier Implementation & Training**
- **Phase 5: Dynamic-Sign Recognition Engine (ST-GCN + Transformer)**
- **Phase 6: Learned Routing & Continuous Translation Pipeline**
- **Phase 7: NLP Post-Processing & TTS Integration**
- **Phase 8: User Interface & Video/Stream Processing**
- **Phase 9: Comprehensive Integration, Verification, & Quality Audit**

See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for full phase details.
