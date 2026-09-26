"""Unit tests for Phase 13 ReportExporter (JSON & PDF report generation)."""

import json
from pathlib import Path
import pytest

from src.nlp.types import SentenceTranscript
from src.utils.exceptions import DataError
from src.utils.report_exporter import ReportExporter
from src.video.file_reader import VideoMetadata
from src.video.processor import TimestampedSignEntry, VideoProcessingResult
from src.video.quality import VideoQualityReport


@pytest.fixture
def sample_video_result() -> VideoProcessingResult:
    """Fixture providing a populated VideoProcessingResult object."""
    meta = VideoMetadata(
        file_path="/tmp/sample.mp4",
        file_name="sample.mp4",
        file_size_bytes=1048576,
        duration_sec=10.0,
        total_frames=300,
        native_fps=30.0,
        width=640,
        height=480,
        aspect_ratio=1.33,
        codec="h264",
    )
    q = VideoQualityReport(
        average_brightness=120.0,
        average_contrast=45.0,
        average_blur_score=200.0,
        resolution_ok=True,
        hand_visibility_ratio=0.95,
        sampled_frames=15,
        overall_rating="EXCELLENT",
    )
    entry = TimestampedSignEntry(
        start_frame=0,
        end_frame=30,
        start_timestamp_sec=0.0,
        end_timestamp_sec=1.0,
        start_timestamp="00:00.000",
        end_timestamp="00:01.000",
        recognized_gloss="HELLO",
        confidence=0.92,
        top_3_candidates=[("HELLO", 0.92), ("WELCOME", 0.05)],
        primary_signer_track_id=1,
    )
    transcript = SentenceTranscript(
        text="HELLO",
        raw_tokens=[],
        filtered_tokens=[],
        mean_confidence=0.92,
    )
    return VideoProcessingResult(
        video_metadata=meta,
        quality_report=q,
        timeline=[entry],
        transcript=transcript,
        tts_audio=None,
        execution_seconds=1.5,
        processing_fps=200.0,
        frames_processed=300,
        recognition_available=True,
    )


def test_export_json_success(sample_video_result: VideoProcessingResult, tmp_path: Path):
    """Test JSON export creates valid, complete JSON file with expected fields."""
    json_path = tmp_path / "reports" / "report.json"
    exported_file = ReportExporter.export_json(sample_video_result, json_path)

    assert exported_file.exists()
    content = json.loads(exported_file.read_text(encoding="utf-8"))

    assert content["video_metadata"]["file_name"] == "sample.mp4"
    assert content["quality_report"]["overall_rating"] == "EXCELLENT"
    assert len(content["timeline"]) == 1
    assert content["timeline"][0]["recognized_gloss"] == "HELLO"
    assert content["transcript"]["text"] == "HELLO"
    assert content["execution_seconds"] == 1.5
    assert content["recognition_available"] is True


def test_export_json_none_result_raises_error(tmp_path: Path):
    """Test export_json raises DataError if result is None."""
    with pytest.raises(DataError, match="Cannot export JSON for None"):
        ReportExporter.export_json(None, tmp_path / "fail.json")


def test_export_pdf_success(sample_video_result: VideoProcessingResult, tmp_path: Path):
    """Test PDF export creates valid, non-empty PDF file."""
    pdf_path = tmp_path / "reports" / "report.pdf"
    exported_file = ReportExporter.export_pdf(sample_video_result, pdf_path)

    assert exported_file.exists()
    assert exported_file.stat().st_size > 500  # Valid non-empty PDF document

    # Header magic check %PDF
    with open(exported_file, "rb") as f:
        header = f.read(4)
        assert header == b"%PDF"


def test_export_pdf_empty_timeline_and_transcript(sample_video_result: VideoProcessingResult, tmp_path: Path):
    """Test PDF export handles empty timeline and missing transcript without crashing."""
    sample_video_result.timeline = []
    sample_video_result.transcript = None
    sample_video_result.recognition_available = False

    pdf_path = tmp_path / "empty_report.pdf"
    exported_file = ReportExporter.export_pdf(sample_video_result, pdf_path)

    assert exported_file.exists()
    assert exported_file.stat().st_size > 500
