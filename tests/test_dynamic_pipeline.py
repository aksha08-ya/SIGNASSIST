"""Unit tests for SIGNASSIST Dynamic-Only Inference Mode & RecognitionPipeline."""

import json
from pathlib import Path
import pytest
import torch

from src.models.confidence import ConfidenceEstimator, ConfidenceOutput
from src.models.dynamic_classifier import DynamicSignClassifier
from src.pipeline.recognition import RecognitionPipeline
from src.ui.services import get_recognition_readiness


def test_dynamic_only_checkpoint_discovery(tmp_path: Path):
    """Test that RecognitionPipeline(mode='dynamic_only') correctly discovers dynamic checkpoint."""
    # 1. Create a dummy state dict matching DynamicSignClassifier
    model = DynamicSignClassifier(num_classes=262)
    ckpt_dir = tmp_path / "checkpoints" / "dynamic_classifier"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "dynamic_classifier_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # 2. Create valid dummy class mapping JSON
    mapping_path = tmp_path / "class_mapping.json"
    mapping_data = {
        "dataset": "INCLUDE",
        "num_classes": 262,
        "id_to_class": {str(i): f"Class_{i}" for i in range(262)},
    }
    mapping_path.write_text(json.dumps(mapping_data), encoding="utf-8")

    # 3. Instantiate pipeline in dynamic_only mode
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        models_dir=tmp_path,
        class_mapping_path=mapping_path,
    )

    assert pipeline.is_available() is True
    assert "Dynamic-only checkpoint loaded successfully" in pipeline._readiness_reason


def test_strict_checkpoint_loading(tmp_path: Path):
    """Test that checkpoint loading uses strict=True and fails on mismatched state dict."""
    # Create invalid state dict missing required weights
    invalid_ckpt = tmp_path / "invalid_dynamic.pt"
    torch.save({"model_state_dict": {"stgcn_backbone.adj_norm": torch.ones(115, 115)}}, invalid_ckpt)

    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"id_to_class": {str(i): f"Class_{i}" for i in range(262)}}),
        encoding="utf-8",
    )

    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        dynamic_checkpoint_path=invalid_ckpt,
        class_mapping_path=mapping_path,
    )

    assert pipeline.is_available() is False
    assert "Failed to load dynamic classifier checkpoint" in pipeline._readiness_reason


def test_dynamic_only_inference_synthetic_tensor():
    """Test dynamic-only forward execution with synthetic tensor shaped [1, 5, 60, 115]."""
    model = DynamicSignClassifier(num_classes=262).eval()
    x = torch.randn(1, 5, 60, 115)
    mask = torch.ones(1, 60, 115, dtype=torch.bool)

    with torch.no_grad():
        out = model(x, mask=mask)

    assert out.logits.shape == (1, 262)
    assert out.embedding.shape == (1, 256)
    assert out.stgcn_features.shape == (1, 256, 60, 115)


def test_262_class_output_shape():
    """Test ConfidenceEstimator output dimensions on 262-class logits."""
    estimator = ConfidenceEstimator(confidence_threshold=0.7, top_k=3)
    logits = torch.randn(2, 262)

    output = estimator(logits)

    assert output.predicted_class.shape == (2,)
    assert output.confidence.shape == (2,)
    assert output.top_k_indices.shape == (2, 3)
    assert output.top_k_probabilities.shape == (2, 3)
    assert output.probabilities.shape == (2, 262)
    assert output.is_accepted.shape == (2,)


def test_real_class_mapping_resolution():
    """Test resolving class IDs 0 to 261 using real class_mapping.json if present."""
    mapping_file = Path("data/processed/include/class_mapping.json")
    if not mapping_file.exists():
        pytest.skip("class_mapping.json not found on disk")

    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_class = data["id_to_class"]
    assert len(id_to_class) == 262
    assert id_to_class["0"] == "1. Dog"
    assert id_to_class["261"] == "Ex. Monsoon"

    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        class_mapping_path=mapping_file,
    )
    assert pipeline.get_dynamic_label(0) == "1. Dog"
    assert pipeline.get_dynamic_label(261) == "Ex. Monsoon"

    with pytest.raises(KeyError):
        pipeline.get_dynamic_label(9999)


def test_missing_checkpoint_failure(tmp_path: Path):
    """Test that missing checkpoint sets is_available() to False."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"id_to_class": {str(i): f"Class_{i}" for i in range(262)}}),
        encoding="utf-8",
    )

    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        models_dir=empty_dir,
        class_mapping_path=mapping_path,
    )

    assert pipeline.is_available() is False
    assert "Missing required dynamic classifier checkpoint" in pipeline._readiness_reason


def test_missing_or_malformed_class_mapping_failure(tmp_path: Path):
    """Test that missing or malformed class mapping fails explicitly with useful error."""
    model = DynamicSignClassifier(num_classes=262)
    ckpt_path = tmp_path / "dynamic_classifier.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # Missing class mapping
    with pytest.raises(FileNotFoundError):
        RecognitionPipeline(
            mode="dynamic_only",
            dynamic_checkpoint_path=ckpt_path,
            class_mapping_path=tmp_path / "non_existent.json",
        )

    # Malformed class mapping
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps({"invalid_key": 123}), encoding="utf-8")

    with pytest.raises(ValueError):
        RecognitionPipeline(
            mode="dynamic_only",
            dynamic_checkpoint_path=ckpt_path,
            class_mapping_path=malformed_path,
        )


def test_multimodal_mode_behavior_unchanged(tmp_path: Path):
    """Test that multimodal mode still requires all 4 checkpoint files."""
    empty_dir = tmp_path / "models"
    empty_dir.mkdir()

    pipeline = RecognitionPipeline(mode="multimodal", models_dir=empty_dir)

    assert pipeline.mode == "multimodal"
    assert pipeline.is_available() is False
    assert "Missing required checkpoint files: static_gcn, dynamic_classifier, router, fusion" in pipeline._readiness_reason

    readiness = get_recognition_readiness(models_dir=empty_dir, mode="multimodal")
    assert readiness["recognition_pipeline_available"] is False
