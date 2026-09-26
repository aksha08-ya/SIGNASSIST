"""Smoke test script for SIGNASSIST Phase 14 Video URL Ingestion & Graceful Fallback Pipeline."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.exceptions import URLExtractionError, VideoProcessingError
from src.video import URLVideoDownloader, URLVideoMetadata, VideoProcessingResult, VideoProcessor


def create_synthetic_video(file_path: Path) -> Path:
    """Helper creating a synthetic MP4 video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(file_path), fourcc, 30.0, (640, 480))
    for i in range(30):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        shift = (i * 5) % 520
        cv2.rectangle(frame, (10 + shift, 10), (110 + shift, 110), (255, 0, 0), -1)
        out.write(frame)
    out.release()
    return file_path


def main() -> None:
    print("=== SIGNASSIST Phase 14 Video URL Ingestion Smoke Test ===")

    downloader = URLVideoDownloader()
    processor = VideoProcessor()

    # 1. URL Validation & Platform Detection Checks
    print("\n[1/5] Testing URL validation and deterministic platform detection...")
    assert downloader.validate_url("https://www.youtube.com/watch?v=sample") == "https://www.youtube.com/watch?v=sample"
    assert downloader.detect_platform("https://www.youtube.com/watch?v=sample") == "youtube"
    assert downloader.detect_platform("https://www.instagram.com/reel/sample/") == "instagram"
    assert downloader.detect_platform("https://fb.watch/sample/") == "facebook"
    assert downloader.detect_platform("https://drive.google.com/file/d/sample/view") == "gdrive"
    assert downloader.detect_platform("https://example.com/sample.mp4") == "direct"
    print("      URL validation and platform detection passed!")

    # 2. URL -> VideoProcessor Pipeline Execution
    print("\n[2/5] Testing URL -> VideoProcessor pipeline with synthetic file...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="signassist_smoke_url_"))
    vid_file = tmp_dir / "sample.mp4"
    create_synthetic_video(vid_file)

    meta = URLVideoMetadata(
        original_url="https://example.com/sample.mp4",
        platform_name="direct",
        video_title="Synthetic Sample Clip",
        retrieved_file_path=str(vid_file),
        duration_sec=1.0,
        is_temporary=True,
        temp_dir=str(tmp_dir),
    )

    with patch.object(downloader, "download_video", return_value=meta):
        result: VideoProcessingResult = downloader.process_url(
            "https://example.com/sample.mp4",
            processor=processor,
        )

    print(f"      Source Type: {result.additional_metadata.get('source_type')}")
    print(f"      Original URL: {result.additional_metadata.get('original_url')}")
    print(f"      Platform Name: {result.additional_metadata.get('platform_name')}")
    print(f"      Frames Processed: {result.frames_processed}")

    assert result.additional_metadata.get("source_type") == "url"
    assert result.additional_metadata.get("original_url") == "https://example.com/sample.mp4"
    assert result.additional_metadata.get("platform_name") == "direct"

    # 3. Temporary Directory Cleanup Verification
    print("\n[3/5] Verifying temporary directory cleanup after processing...")
    assert not tmp_dir.exists()
    print("      Temporary directory cleaned up successfully!")

    # 4. Processor Exception Cleanup Safety
    print("\n[4/5] Verifying temporary directory cleanup on VideoProcessor exception...")
    fail_tmp_dir = Path(tempfile.mkdtemp(prefix="signassist_smoke_fail_"))
    fail_vid_file = fail_tmp_dir / "sample.mp4"
    create_synthetic_video(fail_vid_file)

    fail_meta = URLVideoMetadata(
        original_url="https://example.com/sample.mp4",
        platform_name="direct",
        video_title="Synthetic Fail Clip",
        retrieved_file_path=str(fail_vid_file),
        duration_sec=1.0,
        is_temporary=True,
        temp_dir=str(fail_tmp_dir),
    )

    mock_fail_processor = MagicMock()
    mock_fail_processor.process_file.side_effect = VideoProcessingError("Simulated processor error")

    with patch.object(downloader, "download_video", return_value=fail_meta):
        try:
            downloader.process_url("https://example.com/sample.mp4", processor=mock_fail_processor)
            assert False, "Should have raised VideoProcessingError"
        except VideoProcessingError:
            print("      Caught expected VideoProcessingError.")

    assert not fail_tmp_dir.exists()
    print("      Temporary directory cleaned up despite processing error!")

    # 5. Invalid URL Extraction Error Path
    print("\n[5/5] Testing invalid URL extraction error path...")
    try:
        downloader.process_url("ftp://invalid-scheme.com/video.mp4")
        assert False, "Should have raised URLExtractionError"
    except URLExtractionError as e:
        print(f"      Caught expected URLExtractionError: {e.message}")

    print("\n=== Phase 14 Video URL Ingestion Smoke Test Passed Successfully! ===")


if __name__ == "__main__":
    main()
