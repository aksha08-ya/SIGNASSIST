"""Unit tests for SIGNASSIST Phase 17 Block 6 Research Analytics UI Integration."""

import json
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from src.app import render_mode_research_analytics
from src.ui.services import parse_research_artifact
from src.ui.state import init_session_state
from src.utils.ablation import AblationConfig, AblationResult, AblationStudyResult
from src.utils.loso_evaluator import LOSOEvaluationResult, LOSOFoldResult
from src.utils.metrics import EvaluationResult


class DummyUploadedJSONFile:
    """Mock Streamlit UploadedFile object containing JSON data for headless testing."""

    def __init__(self, name: str, content_dict: Dict[str, Any]):
        self.name = name
        self.content_bytes = json.dumps(content_dict).encode("utf-8")
        self.size = len(self.content_bytes)

    def getvalue(self) -> bytes:
        return self.content_bytes

    def read(self) -> bytes:
        return self.content_bytes


class DummyRawBytesFile:
    """Mock Streamlit UploadedFile object containing arbitrary bytes."""

    def __init__(self, name: str, raw_bytes: bytes):
        self.name = name
        self.content_bytes = raw_bytes
        self.size = len(raw_bytes)

    def getvalue(self) -> bytes:
        return self.content_bytes

    def read(self) -> bytes:
        return self.content_bytes


# 1. Test Research Analytics header and empty state render
def test_research_analytics_header_and_empty_state() -> None:
    """Tests that header and empty state are rendered when no file is uploaded and demo is off."""
    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=None), \
         patch("streamlit.checkbox", return_value=False), \
         patch("src.app.render_header") as mock_header, \
         patch("src.app.render_empty_state") as mock_empty, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        mock_header.assert_called_once()
        mock_empty.assert_called_once()
        assert "Research Evaluation Framework" in mock_empty.call_args[1].get("title", "")


# 2. Test valid EvaluationResult JSON loads successfully
def test_parse_research_artifact_evaluation_result() -> None:
    """Tests parse_research_artifact with standard EvaluationResult payload."""
    data = {
        "accuracy": 0.85,
        "macro_precision": 0.82,
        "macro_recall": 0.80,
        "macro_f1": 0.81,
        "micro_precision": 0.85,
        "micro_recall": 0.85,
        "micro_f1": 0.85,
        "weighted_precision": 0.84,
        "weighted_recall": 0.85,
        "weighted_f1": 0.84,
        "total_samples": 100,
        "num_classes": 2,
        "class_names": ["c0", "c1"],
        "per_class_metrics": {
            "c0": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
            "c1": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "support": 50},
        },
        "confusion_matrix": [[40, 10], [5, 45]],
    }
    parsed = parse_research_artifact(data)
    assert parsed["artifact_type"] == "evaluation"
    assert parsed["payload"]["accuracy"] == 0.85


# 3. Test core metrics, per-class metrics, and confusion matrix are displayed
def test_research_analytics_renders_evaluation_metrics() -> None:
    """Tests that core metrics, per-class metrics, and confusion matrix dataframes render."""
    eval_dict = {
        "accuracy": 0.85,
        "macro_precision": 0.82,
        "macro_recall": 0.80,
        "macro_f1": 0.81,
        "micro_precision": 0.85,
        "micro_recall": 0.85,
        "micro_f1": 0.85,
        "weighted_precision": 0.84,
        "weighted_recall": 0.85,
        "weighted_f1": 0.84,
        "total_samples": 100,
        "num_classes": 2,
        "class_names": ["CLASS_A", "CLASS_B"],
        "per_class_metrics": {
            "CLASS_A": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
            "CLASS_B": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "support": 50},
        },
        "confusion_matrix": [[40, 10], [5, 45]],
    }
    dummy_file = DummyUploadedJSONFile("eval_results.json", eval_dict)

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.checkbox", return_value=False), \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        assert state["research_result_type"] == "evaluation"
        assert state["research_result"]["accuracy"] == 0.85

        # Verify st.metric calls
        metric_labels = [c[0][0] for c in mock_metric.call_args_list]
        assert "Accuracy" in metric_labels
        assert "Macro F1" in metric_labels

        # Verify per-class and confusion matrix dataframes rendered
        assert mock_df.call_count >= 2


# 4. Test invalid/malformed JSON produces an error without crashing
def test_research_analytics_malformed_json_error() -> None:
    """Tests that malformed JSON payload renders an error state without crashing."""
    dummy_file = DummyRawBytesFile("corrupt.json", b"{ malformed json string }")

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.checkbox", return_value=False), \
         patch("src.app.render_error_state") as mock_error, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        assert state["research_result"] is None
        assert "Malformed JSON" in state["research_last_error"]
        mock_error.assert_called_once()


# 5. Test unsupported JSON produces error without fabricating metrics
def test_research_analytics_unsupported_json_structure() -> None:
    """Tests that unsupported JSON dict structure renders error state without fabricating metrics."""
    dummy_file = DummyUploadedJSONFile("unsupported.json", {"unrelated_key": 123})

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.checkbox", return_value=False), \
         patch("src.app.render_error_state") as mock_error, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        assert state["research_result"] is None
        assert "Unsupported research evaluation artifact" in state["research_last_error"]
        mock_error.assert_called_once()


# 6. Test valid LOSO JSON displays fold and aggregate information
def test_research_analytics_loso_json_display() -> None:
    """Tests loading and rendering of a LOSOEvaluationResult artifact."""
    loso_dict = {
        "evaluation_type": "leave_one_signer_out",
        "num_signers": 2,
        "num_samples": 40,
        "num_classes": 2,
        "class_names": ["C0", "C1"],
        "folds": [
            {
                "held_out_signer": "Signer_1",
                "train_sample_count": 20,
                "test_sample_count": 20,
                "train_sample_ids": ["s1"],
                "test_sample_ids": ["s2"],
                "true_labels": [0, 1],
                "predicted_labels": [0, 1],
                "metrics": {
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "per_class_metrics": {},
                    "confusion_matrix": [[10, 0], [0, 10]],
                },
            }
        ],
        "aggregate_metrics": {
            "accuracy": 0.95,
            "macro_precision": 0.95,
            "macro_recall": 0.95,
            "macro_f1": 0.95,
            "micro_precision": 0.95,
            "micro_recall": 0.95,
            "micro_f1": 0.95,
            "weighted_precision": 0.95,
            "weighted_recall": 0.95,
            "weighted_f1": 0.95,
            "total_samples": 40,
            "num_classes": 2,
            "class_names": ["C0", "C1"],
            "per_class_metrics": {
                "C0": {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 20},
                "C1": {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 20},
            },
            "confusion_matrix": [[19, 1], [1, 19]],
        },
    }

    dummy_file = DummyUploadedJSONFile("loso_results.json", loso_dict)

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.checkbox", return_value=False), \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        assert state["research_result_type"] == "loso"
        assert state["research_result"]["num_signers"] == 2
        mock_df.assert_called()


# 7. Test valid Ablation JSON displays experiment results
def test_research_analytics_ablation_json_display() -> None:
    """Tests loading and rendering of an AblationStudyResult artifact."""
    ablation_dict = {
        "evaluation_type": "standard",
        "num_experiments": 2,
        "experiments": [
            {
                "name": "full_model",
                "description": "Full model with all features",
                "parameters": {"use_face": True},
                "sample_count": 50,
                "signer_count": 3,
                "metrics": {
                    "accuracy": 0.90,
                    "macro_f1": 0.89,
                },
            },
            {
                "name": "no_face",
                "description": "Excludes facial keypoints",
                "parameters": {"use_face": False},
                "sample_count": 50,
                "signer_count": 3,
                "metrics": {
                    "accuracy": 0.75,
                    "macro_f1": 0.74,
                },
            },
        ],
    }

    dummy_file = DummyUploadedJSONFile("ablation_results.json", ablation_dict)

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.checkbox", return_value=False), \
         patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        assert state["research_result_type"] == "ablation"
        mock_df.assert_called_once()
        table_rows = mock_df.call_args[0][0]
        assert len(table_rows) == 2
        assert table_rows[0]["Experiment Name"] == "full_model"
        assert table_rows[1]["Experiment Name"] == "no_face"


# 8. Test Demo mode is explicitly labeled DEMONSTRATION/TEST ONLY and not stored as real research result
def test_research_analytics_demo_mode_prominent_warning() -> None:
    """Tests that activating demo mode renders the required DEMONSTRATION / TEST ONLY warning."""
    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=None), \
         patch("streamlit.checkbox", return_value=True), \
         patch("streamlit.error") as mock_error, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        assert state["research_demo_active"] is True
        mock_error.assert_called()
        error_msg = mock_error.call_args[0][0]
        assert "DEMONSTRATION / TEST ONLY" in error_msg
        assert "SYNTHETIC DATA" in error_msg
        assert "NOT RESEARCH RESULTS" in error_msg


# 9. Test cached loaded results survive ordinary reruns without reload
def test_research_analytics_cached_result_survives_rerun() -> None:
    """Tests that pre-loaded research_result in session_state persists across reruns."""
    eval_payload = {
        "accuracy": 0.99,
        "macro_precision": 0.99,
        "macro_recall": 0.99,
        "macro_f1": 0.99,
        "micro_precision": 0.99,
        "micro_recall": 0.99,
        "micro_f1": 0.99,
        "weighted_precision": 0.99,
        "weighted_recall": 0.99,
        "weighted_f1": 0.99,
        "total_samples": 10,
        "num_classes": 1,
        "class_names": ["c0"],
        "per_class_metrics": {"c0": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 10}},
        "confusion_matrix": [[10]],
    }

    state = init_session_state({})
    state["research_result"] = eval_payload
    state["research_result_type"] = "evaluation"
    state["research_result_filename"] = "cached_eval.json"

    with patch("streamlit.file_uploader", return_value=None), \
         patch("streamlit.checkbox", return_value=False), \
         patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.session_state", state):

        render_mode_research_analytics({})

        mock_df.assert_called()
        assert state["research_result"]["accuracy"] == 0.99
