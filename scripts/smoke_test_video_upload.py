"""Smoke test script for SIGNASSIST Phase 13 Video File Upload & Chunking Subsystem."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.report_exporter import ReportExporter
from src.video import VideoFileReader, VideoProcessingResult, VideoProcessor, VideoQualityAssessor


def create_synthetic_video(
    file_path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    duration_sec: float = 2.0,
) -> Path:
    """Helper creating a synthetic MP4 video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(round(fps * duration_sec))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        shift = (i * 4) % (width - 120)
        cv2.rectangle(frame, (10 + shift, 10), (110 + shift, 110), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return file_path


def main() -> None:
    print("=== SIGNASSIST Phase 13 Video Upload & Processing Smoke Test ===")

    tmp_dir = Path(tempfile.gettempdir()) / "signassist_phase13_smoke"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create short synthetic video (2.0 seconds, 60 frames)
    print("\n[1/6] Generating synthetic test video (2.0s @ 30 FPS)...")
    vid_path = tmp_dir / "short_sample.mp4"
    create_synthetic_video(vid_path, width=640, height=480, fps=30.0, duration_sec=2.0)
    print(f"      Created video file at: {vid_path}")

    # 2. Extract VideoMetadata
    print("\n[2/6] Extracting video metadata...")
    reader = VideoFileReader(vid_path)
    meta = reader.load_metadata()
    print(f"      File Name: {meta.file_name}")
    print(f"      Resolution: {meta.width}x{meta.height}")
    print(f"      Native FPS: {meta.native_fps}")
    print(f"      Total Frames: {meta.total_frames}")
    print(f"      Duration: {meta.duration_sec:.2f}s")
    assert meta.total_frames == 60
    assert meta.width == 640

    # 3. Assess Quality
    print("\n[3/6] Assessing video quality...")
    quality_assessor = VideoQualityAssessor()
    quality = quality_assessor.assess_video(reader)
    print(f"      Brightness: {quality.average_brightness:.1f}")
    print(f"      Contrast: {quality.average_contrast:.1f}")
    print(f"      Blur Score: {quality.average_blur_score:.1f}")
    print(f"      Resolution OK: {quality.resolution_ok}")
    print(f"      Overall Rating: {quality.overall_rating}")
    assert quality.resolution_ok is True

    # 4. Run VideoProcessor in recognition-unavailable mode
    print("\n[4/6] Processing video with VideoProcessor (Recognition Unavailable Mode)...")
    processor = VideoProcessor(recognition_pipeline=None)
    result = processor.process_file(vid_path)

    print(f"      Frames Processed: {result.frames_processed}")
    print(f"      Execution Seconds: {result.execution_seconds:.3f}s")
    print(f"      Processing Rate: {result.processing_fps:.1f} FPS")
    print(f"      Recognition Available: {result.recognition_available}")
    print(f"      Timeline Entry Count: {len(result.timeline)}")

    assert result.recognition_available is False
    assert len(result.timeline) == 0
    assert result.transcript is None

    # 5. Export JSON and PDF reports
    print("\n[5/6] Exporting JSON and PDF reports...")
    json_path = tmp_dir / "report.json"
    pdf_path = tmp_dir / "report.pdf"

    ReportExporter.export_json(result, json_path)
    ReportExporter.export_pdf(result, pdf_path)

    assert json_path.exists()
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 500

    # Read JSON back and verify fields
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["video_metadata"]["file_name"] == "short_sample.mp4"
    assert json_data["quality_report"]["resolution_ok"] is True
    assert json_data["recognition_available"] is False
    print("      JSON & PDF Export Verified Successfully!")

    # 6. Performance Validation on 60-Second Video (~1,800 Frames)
    print("\n[6/6] Performance Benchmark on 60-Second Video (1,800 Frames @ 30 FPS)...")
    long_vid_path = tmp_dir / "bench_60s.mp4"
    create_synthetic_video(long_vid_path, width=640, height=480, fps=30.0, duration_sec=60.0)

    bench_start = time.perf_counter()
    bench_result = processor.process_file(long_vid_path)
    bench_time = time.perf_counter() - bench_start

    print(f"      Measured Processing Time: {bench_time:.3f} seconds")
    print(f"      Frames Processed: {bench_result.frames_processed}")
    print(f"      Effective Throughput: {bench_result.processing_fps:.1f} FPS")

    print("\n=== Phase 13 Video Upload & Processing Smoke Test Passed Successfully! ===")


if __name__ == "__main__":
    main()
