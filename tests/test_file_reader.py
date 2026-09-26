"""Unit tests for Phase 13 VideoFileReader & VideoMetadata."""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pytest

from src.utils.config import VideoConfig
from src.utils.exceptions import VideoProcessingError
from src.video.file_reader import DecodedFrame, VideoFileReader, VideoMetadata, format_timestamp


def create_synthetic_video(
    file_path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    duration_sec: float = 2.0,
    color: Tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Helper function creating a synthetic MP4 video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(round(fps * duration_sec))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        shift = (i * 3) % (width - 100)
        cv2.rectangle(frame, (10 + shift, 10), (100 + shift, 100), (255, 0, 0), -1)
        out.write(frame)
    out.release()
    return file_path


def test_timestamp_formatting():
    """Verify format_timestamp outputs correct MM:SS.mmm format."""
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(65.432) == "01:05.432"
    assert format_timestamp(3605.123) == "60:05.123"
    assert format_timestamp(-5.0) == "00:00.000"


def test_valid_video_metadata_extraction(tmp_path: Path):
    """Test loading metadata from a valid synthetic MP4 video."""
    vid_path = tmp_path / "test_video.mp4"
    create_synthetic_video(vid_path, width=640, height=480, fps=30.0, duration_sec=2.0)

    reader = VideoFileReader(file_path=vid_path)
    meta = reader.load_metadata()

    assert isinstance(meta, VideoMetadata)
    assert meta.file_name == "test_video.mp4"
    assert meta.width == 640
    assert meta.height == 480
    assert meta.native_fps == pytest.approx(30.0, abs=0.1)
    assert meta.total_frames == 60
    assert meta.duration_sec == pytest.approx(2.0, abs=0.1)
    assert meta.aspect_ratio == pytest.approx(640 / 480, abs=1e-2)
    assert meta.file_size_bytes > 0
    assert meta.frame_count_consistent is True


def test_exact_timestamp_behavior(tmp_path: Path):
    """Test native timestamp calculation: timestamp_sec == frame_index / native_fps."""
    vid_path = tmp_path / "ts_video.mp4"
    create_synthetic_video(vid_path, width=320, height=240, fps=25.0, duration_sec=1.0)

    reader = VideoFileReader(file_path=vid_path)
    frames = list(reader.decode_frames())

    assert len(frames) == 25
    for idx, df in enumerate(frames):
        assert isinstance(df, DecodedFrame)
        assert df.frame_index == idx
        expected_sec = idx / 25.0
        assert df.timestamp_sec == pytest.approx(expected_sec, abs=1e-4)
        assert df.timestamp == format_timestamp(expected_sec)


def test_lazy_frame_decoding_properties(tmp_path: Path):
    """Test decode_frames yields frames sequentially with expected properties."""
    vid_path = tmp_path / "lazy_video.mp4"
    create_synthetic_video(vid_path, width=320, height=240, fps=30.0, duration_sec=1.0)

    reader = VideoFileReader(file_path=vid_path)
    count = 0
    for df in reader.decode_frames(stride=2):
        assert df.frame.shape == (240, 320, 3)
        assert df.frame_index == count * 2
        count += 1
    assert count == 15  # 30 total frames / stride 2 = 15 frames


def test_chunked_frame_decoding(tmp_path: Path):
    """Test decode_frames_chunked groups frames into specified chunk sizes."""
    vid_path = tmp_path / "chunk_video.mp4"
    # Create 75 frames (2.5 sec at 30 fps)
    create_synthetic_video(vid_path, width=320, height=240, fps=30.0, duration_sec=2.5)

    reader = VideoFileReader(file_path=vid_path)
    chunks = list(reader.decode_frames_chunked(chunk_size=30))

    assert len(chunks) == 3  # 30 + 30 + 15
    assert len(chunks[0]) == 30
    assert len(chunks[1]) == 30
    assert len(chunks[2]) == 15

    total_yielded = sum(len(c) for c in chunks)
    assert total_yielded == 75


def test_sample_frames(tmp_path: Path):
    """Test sample_frames extracts evenly spaced frames."""
    vid_path = tmp_path / "sample_video.mp4"
    create_synthetic_video(vid_path, width=320, height=240, fps=30.0, duration_sec=3.0)

    reader = VideoFileReader(file_path=vid_path)
    samples = reader.sample_frames(num_samples=5)

    assert len(samples) == 5
    indices = [s.frame_index for s in samples]
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 89  # total_frames - 1 = 89


def test_invalid_extension_rejection(tmp_path: Path):
    """Test rejection of unsupported file extension."""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("dummy content")

    reader = VideoFileReader(file_path=txt_path)
    with pytest.raises(VideoProcessingError, match="Unsupported video file extension"):
        reader.validate_file()


def test_missing_file_rejection(tmp_path: Path):
    """Test rejection of missing file path."""
    missing_path = tmp_path / "non_existent.mp4"
    reader = VideoFileReader(file_path=missing_path)
    with pytest.raises(VideoProcessingError, match="Video file does not exist"):
        reader.validate_file()


def test_corrupt_unreadable_video_rejection(tmp_path: Path):
    """Test rejection of corrupt or invalid video file content."""
    fake_vid = tmp_path / "corrupt.mp4"
    fake_vid.write_bytes(b"NOT A VALID MP4 FILE CONTENT")

    reader = VideoFileReader(file_path=fake_vid)
    with pytest.raises(VideoProcessingError, match="Failed to open video file"):
        reader.validate_file()


def test_duration_limit_enforcement(tmp_path: Path):
    """Test rejection of videos exceeding max configured duration."""
    vid_path = tmp_path / "long_video.mp4"
    # Create 5-second video but configure max 2-second limit
    create_synthetic_video(vid_path, width=320, height=240, fps=30.0, duration_sec=5.0)

    cfg = VideoConfig(max_video_length_sec=2)
    reader = VideoFileReader(file_path=vid_path, config=cfg)

    with pytest.raises(VideoProcessingError, match="exceeds max configured limit"):
        reader.validate_file()
