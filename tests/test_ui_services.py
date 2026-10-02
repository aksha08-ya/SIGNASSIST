"""Unit tests for SIGNASSIST Phase 17 Block 2 UI Service Layer."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex
from src.nlp.text_to_isl import DemonstrationSource, ISLDemonstrationSequence
from src.pipeline.recognition import RecognitionPipeline
from src.ui.services import (
    TextToISLResult,
    _create_default_video_processor,
    convert_text_to_isl,
    evaluate_classification,
    get_recognition_readiness,
    process_video_file,
    process_video_url,
)
from src.utils.metrics import EvaluationResult, MetricsEvaluator
from src.video.processor import VideoProcessingResult, VideoProcessor


# 1. Video Processing Service Test
def test_process_video_file_service_delegation() -> None:
    """Tests that process_video_file delegates to VideoProcessor and returns its result unchanged."""
    mock_processor = MagicMock()
    mock_result = MagicMock(spec=VideoProcessingResult)
    mock_result.recognition_available = False
    mock_processor.process_file.return_value = mock_result

    res = process_video_file(
        file_path="test_video.mp4",
        processor=mock_processor,
        manual_signer_id=1,
        enable_tts=True,
        chunk_size=30,
    )

    mock_processor.process_file.assert_called_once_with(
        file_path="test_video.mp4",
        manual_signer_id=1,
        enable_tts=True,
        chunk_size=30,
    )
    assert res is mock_result
    assert res.recognition_available is False


def test_process_video_file_default_processor_creation(tmp_path: Path) -> None:
    """Tests that process_video_file creates default VideoProcessor with RecognitionPipeline when processor is None."""
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    with patch("src.ui.services._create_default_video_processor") as mock_factory:
        mock_processor = MagicMock(spec=VideoProcessor)
        mock_result = MagicMock(spec=VideoProcessingResult)
        mock_processor.process_file.return_value = mock_result
        mock_factory.return_value = mock_processor

        res = process_video_file(file_path=dummy_video)

        mock_factory.assert_called_once()
        mock_processor.process_file.assert_called_once()
        assert res is mock_result


def test_default_video_processor_factory_pipeline_integration(tmp_path: Path) -> None:
    """Tests _create_default_video_processor instantiates RecognitionPipeline adapter without fabricating recognition."""
    processor = _create_default_video_processor(models_dir=tmp_path)

    assert isinstance(processor, VideoProcessor)
    assert isinstance(processor.recognition_pipeline, RecognitionPipeline)
    # Since no trained checkpoints exist in empty models_dir, pipeline is unavailable
    assert processor.recognition_pipeline.is_available() is False


# 2. Video URL Processing Service Test
def test_process_video_url_service_delegation() -> None:
    """Tests that process_video_url delegates to URLVideoDownloader and returns its result unchanged."""
    mock_downloader = MagicMock()
    mock_processor = MagicMock()
    mock_result = MagicMock(spec=VideoProcessingResult)
    mock_downloader.process_url.return_value = mock_result

    res = process_video_url(
        url="https://youtube.com/watch?v=demo",
        downloader=mock_downloader,
        processor=mock_processor,
        output_dir="/tmp/out",
        manual_signer_id=2,
        enable_tts=False,
        chunk_size=60,
    )

    mock_downloader.process_url.assert_called_once_with(
        url="https://youtube.com/watch?v=demo",
        processor=mock_processor,
        output_dir="/tmp/out",
        manual_signer_id=2,
        enable_tts=False,
        chunk_size=60,
    )
    assert res is mock_result


def test_process_video_url_default_processor_creation() -> None:
    """Tests that process_video_url creates default VideoProcessor with RecognitionPipeline when processor is None and passes it to downloader."""
    mock_downloader = MagicMock()
    mock_result = MagicMock(spec=VideoProcessingResult)
    mock_downloader.process_url.return_value = mock_result

    with patch("src.ui.services._create_default_video_processor") as mock_factory:
        mock_processor = MagicMock(spec=VideoProcessor)
        mock_factory.return_value = mock_processor

        res = process_video_url(
            url="https://youtube.com/watch?v=demo",
            downloader=mock_downloader,
            processor=None,
        )

        mock_factory.assert_called_once()
        mock_downloader.process_url.assert_called_once_with(
            url="https://youtube.com/watch?v=demo",
            processor=mock_processor,
            output_dir=None,
            manual_signer_id=None,
            enable_tts=False,
            chunk_size=60,
        )
        assert res is mock_result


# 3. Text → ISL Service Tests
def test_convert_text_to_isl_dictionary_backed() -> None:
    """Tests convert_text_to_isl with a dictionary-backed sequence."""
    entries = [
        ISLDictionaryEntry(gloss="HELLO", video_path="videos/hello.mp4"),
        ISLDictionaryEntry(gloss="WORLD", video_path="videos/world.mp4"),
    ]
    dict_index = ISLDictionaryIndex(entries=entries)

    res = convert_text_to_isl("HELLO WORLD", dictionary=dict_index)

    assert isinstance(res, TextToISLResult)
    assert isinstance(res.sequence, ISLDemonstrationSequence)
    assert res.dictionary_count == 2
    assert res.unknown_count == 0
    assert res.output_video_path is None


def test_convert_text_to_isl_empty_input() -> None:
    """Tests convert_text_to_isl with empty input string."""
    dict_index = ISLDictionaryIndex(entries=[])
    res = convert_text_to_isl("", dictionary=dict_index)

    assert isinstance(res, TextToISLResult)
    assert len(res.sequence.demonstrations) == 0
    assert res.dictionary_count == 0
    assert res.output_video_path is None


def test_convert_text_to_isl_zero_valid_clips_no_output_path(tmp_path: Path) -> None:
    """Tests that stitch_video=True with 0 valid clips does NOT report an output video path."""
    dict_index = ISLDictionaryIndex(entries=[])
    out_video = tmp_path / "stitched_output.mp4"

    res = convert_text_to_isl(
        text="UNKNOWNWORD",
        dictionary=dict_index,
        stitch_video=True,
        output_video_path=out_video,
    )

    assert isinstance(res, TextToISLResult)
    assert res.unknown_count == 1
    assert res.stitch_result is not None
    assert res.stitch_result.total_clips == 0
    # Crucial check: must NOT claim an output video file exists
    assert res.output_video_path is None


# 4. Research Analytics Service Test
def test_evaluate_classification_delegation() -> None:
    """Tests that evaluate_classification delegates to MetricsEvaluator."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]

    res = evaluate_classification(y_true, y_pred, num_classes=2, class_names=["C0", "C1"])

    direct_res = MetricsEvaluator().evaluate(y_true, y_pred, num_classes=2, class_names=["C0", "C1"])

    assert isinstance(res, EvaluationResult)
    assert res.to_dict() == direct_res.to_dict()


# 5. Recognition Readiness Tests
def test_recognition_readiness_no_checkpoint(tmp_path: Path) -> None:
    """Tests get_recognition_readiness when no model checkpoint exists."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    info = get_recognition_readiness(models_dir)

    assert info["architecture_available"] is True
    assert info["checkpoint_available"] is False
    assert info["recognition_pipeline_available"] is False


def test_recognition_readiness_checkpoint_without_adapter(tmp_path: Path) -> None:
    """Tests get_recognition_readiness when a checkpoint exists but no pipeline adapter is active."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = models_dir / "checkpoint.pt"
    ckpt_file.write_bytes(b"dummy_data")

    info = get_recognition_readiness(models_dir)

    assert info["architecture_available"] is True
    assert info["checkpoint_available"] is True
    # Checkpoint alone MUST NOT mean recognition pipeline is available
    assert info["recognition_pipeline_available"] is False


def test_recognition_readiness_explicit_pipeline(tmp_path: Path) -> None:
    """Tests get_recognition_readiness when explicit_pipeline_available=True."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    info = get_recognition_readiness(models_dir, explicit_pipeline_available=True)

    assert info["architecture_available"] is True
    assert info["recognition_pipeline_available"] is True


def test_recognition_readiness_pipeline_instance_available_true(tmp_path: Path) -> None:
    """Tests get_recognition_readiness with a fake pipeline whose is_available() returns True."""
    class FakeAvailablePipeline:
        def is_available(self) -> bool:
            return True

    info = get_recognition_readiness(tmp_path, pipeline_instance=FakeAvailablePipeline())
    assert info["recognition_pipeline_available"] is True


def test_recognition_readiness_pipeline_instance_available_false(tmp_path: Path) -> None:
    """Tests get_recognition_readiness with a fake pipeline whose is_available() returns False."""
    class FakeUnavailablePipeline:
        def is_available(self) -> bool:
            return False

    info = get_recognition_readiness(tmp_path, pipeline_instance=FakeUnavailablePipeline())
    assert info["recognition_pipeline_available"] is False
