"""Unit tests for Phase 14 URLVideoDownloader & URLVideoMetadata."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import cv2

import numpy as np
import pytest

from src.nlp.types import SentenceTranscript
from src.utils.exceptions import URLExtractionError, VideoProcessingError
from src.video.file_reader import VideoMetadata
from src.video.processor import TimestampedSignEntry, VideoProcessingResult, VideoProcessor
from src.video.quality import VideoQualityReport
from src.video.url_downloader import URLVideoDownloader, URLVideoMetadata


def create_synthetic_video(file_path: Path) -> Path:
    """Helper creating a valid synthetic MP4 file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(file_path), fourcc, 30.0, (320, 240))
    for _ in range(30):
        out.write(np.full((240, 320, 3), 128, dtype=np.uint8))
    out.release()
    return file_path


# --- A. URL Validation Tests ---

def test_validate_url_accepts_valid_urls():
    """Test validate_url accepts supported HTTP/HTTPS web video URLs."""
    downloader = URLVideoDownloader()

    valid_urls = [
        "https://www.youtube.com/watch?v=test",
        "https://youtu.be/test",
        "https://www.instagram.com/reel/test/",
        "https://www.facebook.com/watch/?v=test",
        "https://drive.google.com/file/d/test/view",
        "https://example.com/video.mp4",
    ]
    for url in valid_urls:
        assert downloader.validate_url(url) == url


def test_validate_url_rejects_invalid_urls():
    """Test validate_url rejects empty, malformed, non-HTTP, and host-less URLs."""
    downloader = URLVideoDownloader()

    invalid_urls = [
        "",
        "   ",
        "not_a_url",
        "file:///etc/passwd",
        "ftp://example.com/video.mp4",
        "javascript:alert(1)",
        "http:///path/without/host",
        "C:\\Users\\test\\video.mp4",
    ]
    for url in invalid_urls:
        with pytest.raises(URLExtractionError):
            downloader.validate_url(url)


def test_validate_url_rejects_non_string():
    """Test validate_url rejects non-string inputs."""
    downloader = URLVideoDownloader()
    with pytest.raises(URLExtractionError, match="empty or invalid"):
        downloader.validate_url(12345)  # type: ignore


# --- B. Platform Detection Tests ---

def test_detect_platform_classification():
    """Test deterministic platform detection mapping."""
    downloader = URLVideoDownloader()

    assert downloader.detect_platform("https://www.youtube.com/watch?v=123") == "youtube"
    assert downloader.detect_platform("https://youtu.be/123") == "youtube"
    assert downloader.detect_platform("https://www.youtube.com/shorts/123") == "youtube"
    assert downloader.detect_platform("https://www.youtube.com/embed/123") == "youtube"

    assert downloader.detect_platform("https://www.instagram.com/reel/123/") == "instagram"
    assert downloader.detect_platform("https://www.instagram.com/p/123/") == "instagram"
    assert downloader.detect_platform("https://www.instagram.com/tv/123/") == "instagram"

    assert downloader.detect_platform("https://www.facebook.com/watch/?v=123") == "facebook"
    assert downloader.detect_platform("https://fb.watch/123/") == "facebook"
    assert downloader.detect_platform("https://www.facebook.com/reel/123/") == "facebook"

    assert downloader.detect_platform("https://drive.google.com/file/d/123/view") == "gdrive"

    assert downloader.detect_platform("https://example.com/clip.mp4") == "direct"
    assert downloader.detect_platform("https://example.com/clip.mov") == "direct"
    assert downloader.detect_platform("https://example.com/clip.avi") == "direct"
    assert downloader.detect_platform("https://example.com/clip.webm") == "direct"
    assert downloader.detect_platform("https://example.com/clip.mkv") == "direct"

    assert downloader.detect_platform("https://generic-domain.org/page") == "unknown"


def test_detect_platform_query_parameter_isolation():
    """Verify platform keywords in query parameters do not cause false classification."""
    downloader = URLVideoDownloader()
    # Direct MP4 with a query parameter referring to youtube
    url = "https://example.com/video.mp4?referrer=youtube.com"
    assert downloader.detect_platform(url) == "direct"


# --- C. Download Tests (Mocked yt-dlp) ---

def test_download_video_success(tmp_path: Path):
    """Test download_video handles successful yt-dlp retrieval."""
    vid_file = tmp_path / "mock_123.mp4"
    create_synthetic_video(vid_file)

    downloader = URLVideoDownloader()
    mock_info = {"title": "Sample Video Title", "duration": 15.5}

    with patch("yt_dlp.YoutubeDL") as MockYDL:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        MockYDL.return_value.__enter__.return_value = mock_instance

        metadata = downloader.download_video("https://example.com/sample.mp4", output_dir=tmp_path)

        assert metadata.original_url == "https://example.com/sample.mp4"
        assert metadata.platform_name == "direct"
        assert metadata.video_title == "Sample Video Title"
        assert metadata.duration_sec == 15.5
        assert metadata.is_temporary is False


def test_download_video_private_restricted_failure(tmp_path: Path):
    """Test download_video maps private/restricted errors to URLExtractionError."""
    downloader = URLVideoDownloader()

    with patch("yt_dlp.YoutubeDL") as MockYDL:
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = Exception("This video is private or login required")
        MockYDL.return_value.__enter__.return_value = mock_instance

        with pytest.raises(URLExtractionError, match="private or requires authentication"):
            downloader.download_video("https://www.youtube.com/watch?v=private", output_dir=tmp_path)


def test_download_video_timeout_failure(tmp_path: Path):
    """Test download_video maps socket timeouts to URLExtractionError."""
    downloader = URLVideoDownloader()

    with patch("yt_dlp.YoutubeDL") as MockYDL:
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = TimeoutError("Connection timed out")
        MockYDL.return_value.__enter__.return_value = mock_instance

        with pytest.raises(URLExtractionError, match="timed out"):
            downloader.download_video("https://example.com/timeout.mp4", output_dir=tmp_path)


def test_download_video_resource_constraints(tmp_path: Path):
    """Verify max_filesize constraint is passed to yt-dlp configuration."""
    downloader = URLVideoDownloader()

    with patch("yt_dlp.YoutubeDL") as MockYDL:
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = Exception("Cancelled")
        MockYDL.return_value.__enter__.return_value = mock_instance

        with pytest.raises(URLExtractionError):
            downloader.download_video("https://example.com/video.mp4", output_dir=tmp_path)

        args, kwargs = MockYDL.call_args
        ydl_opts = args[0]
        assert ydl_opts.get("max_filesize") == 150 * 1024 * 1024
        assert ydl_opts.get("noplaylist") is True


# --- D. Temporary Lifecycle Tests ---

def test_temporary_lifecycle_omitted_output_dir():
    """Verify temporary directory creation and cleanup when output_dir is omitted."""
    downloader = URLVideoDownloader()

    with patch.object(downloader, "download_video") as mock_download:
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_lifecycle_"))
        vid_file = tmp_dir / "sample.mp4"
        create_synthetic_video(vid_file)

        meta = URLVideoMetadata(
            original_url="https://example.com/test.mp4",
            platform_name="direct",
            video_title="test",
            retrieved_file_path=str(vid_file),
            duration_sec=1.0,
            is_temporary=True,
            temp_dir=str(tmp_dir),
        )

        assert meta.is_temporary is True
        assert Path(meta.temp_dir).exists()

        # Idempotent cleanup test
        meta.cleanup()
        assert not Path(meta.temp_dir).exists()
        meta.cleanup()  # Second call does not crash


def test_temporary_lifecycle_supplied_output_dir(tmp_path: Path):
    """Verify caller-owned output_dir is not deleted by cleanup."""
    vid_file = tmp_path / "caller_owned.mp4"
    create_synthetic_video(vid_file)

    meta = URLVideoMetadata(
        original_url="https://example.com/test.mp4",
        platform_name="direct",
        video_title="test",
        retrieved_file_path=str(vid_file),
        duration_sec=1.0,
        is_temporary=False,
        temp_dir=None,
    )

    meta.cleanup()
    assert vid_file.exists()


# --- E. process_url() Tests ---

def test_process_url_success(tmp_path: Path):
    """Test process_url downloads video, invokes VideoProcessor, attaches provenance, and cleans up."""
    downloader = URLVideoDownloader()

    tmp_dir = Path(tempfile.mkdtemp(prefix="test_proc_url_"))
    vid_file = tmp_dir / "sample.mp4"
    create_synthetic_video(vid_file)

    meta = URLVideoMetadata(
        original_url="https://example.com/sample.mp4",
        platform_name="direct",
        video_title="Sample Title",
        retrieved_file_path=str(vid_file),
        duration_sec=1.0,
        is_temporary=True,
        temp_dir=str(tmp_dir),
    )

    processor = VideoProcessor()

    with patch.object(downloader, "download_video", return_value=meta):
        result = downloader.process_url("https://example.com/sample.mp4", processor=processor)

        assert isinstance(result, VideoProcessingResult)
        assert result.additional_metadata.get("source_type") == "url"
        assert result.additional_metadata.get("original_url") == "https://example.com/sample.mp4"
        assert result.additional_metadata.get("platform_name") == "direct"
        assert result.additional_metadata.get("video_title") == "Sample Title"

        # Verify temporary directory was cleaned up
        assert not tmp_dir.exists()


def test_process_url_processor_failure_cleanup(tmp_path: Path):
    """Verify temporary directory is cleaned up even if VideoProcessor raises an exception."""
    downloader = URLVideoDownloader()

    tmp_dir = Path(tempfile.mkdtemp(prefix="test_proc_fail_"))
    vid_file = tmp_dir / "sample.mp4"
    create_synthetic_video(vid_file)

    meta = URLVideoMetadata(
        original_url="https://example.com/sample.mp4",
        platform_name="direct",
        video_title="Sample Title",
        retrieved_file_path=str(vid_file),
        duration_sec=1.0,
        is_temporary=True,
        temp_dir=str(tmp_dir),
    )

    mock_processor = MagicMock()
    mock_processor.process_file.side_effect = VideoProcessingError("Simulated processing failure")

    with patch.object(downloader, "download_video", return_value=meta):
        with pytest.raises(VideoProcessingError, match="Simulated processing failure"):
            downloader.process_url("https://example.com/sample.mp4", processor=mock_processor)

        # Temp directory must be cleaned up in finally block
        assert not tmp_dir.exists()


def test_process_url_extraction_failure_prevents_processor():
    """Verify URL extraction failure raises URLExtractionError without calling VideoProcessor."""
    downloader = URLVideoDownloader()
    mock_processor = MagicMock()

    with pytest.raises(URLExtractionError):
        downloader.process_url("invalid_url", processor=mock_processor)

    mock_processor.process_file.assert_not_called()


# --- F. Security & Privacy Tests ---

def test_security_and_privacy_clean_metadata(tmp_path: Path):
    """Verify additional_metadata contains no sensitive credentials or raw cookies."""
    downloader = URLVideoDownloader()

    tmp_dir = Path(tempfile.mkdtemp(prefix="test_privacy_"))
    vid_file = tmp_dir / "privacy.mp4"
    create_synthetic_video(vid_file)

    meta = URLVideoMetadata(
        original_url="https://example.com/privacy.mp4",
        platform_name="direct",
        video_title="Privacy Title",
        retrieved_file_path=str(vid_file),
        duration_sec=1.0,
        is_temporary=True,
        temp_dir=str(tmp_dir),
    )

    processor = VideoProcessor()

    with patch.object(downloader, "download_video", return_value=meta):
        result = downloader.process_url("https://example.com/privacy.mp4", processor=processor)

        add_meta = result.additional_metadata
        assert "cookies" not in add_meta
        assert "headers" not in add_meta
        assert "auth" not in add_meta
        assert add_meta["source_type"] == "url"
        assert add_meta["original_url"] == "https://example.com/privacy.mp4"
