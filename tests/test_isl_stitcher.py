"""Unit tests for SIGNASSIST Phase 15 Block 3 ISLVideoStitcher."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.nlp.text_to_isl import (
    DemonstrationSource,
    ISLDemonstration,
    ISLDemonstrationSequence,
)
from src.utils.exceptions import VideoProcessingError
from src.video.isl_stitcher import ISLStitchResult, ISLVideoStitcher


def create_synthetic_mp4(
    path: Path,
    num_frames: int = 15,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Path:
    """Helper utility to generate small valid synthetic MP4 video clips at runtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        img = np.full((height, width, 3), color, dtype=np.uint8)
        cv2.putText(
            img, str(i), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        writer.write(img)
    writer.release()
    return path


def test_dictionary_only_stitching(tmp_path: Path) -> None:
    """Tests stitching sequence containing only valid dictionary video clips."""
    clip1 = create_synthetic_mp4(tmp_path / "hello.mp4", num_frames=15, fps=30.0)
    clip2 = create_synthetic_mp4(tmp_path / "world.mp4", num_frames=15, fps=30.0)

    demo1 = ISLDemonstration(
        token="HELLO",
        normalized_token="hello",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(clip1),
        dictionary_gloss="HELLO",
    )
    demo2 = ISLDemonstration(
        token="WORLD",
        normalized_token="world",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(clip2),
        dictionary_gloss="WORLD",
    )

    sequence = ISLDemonstrationSequence(
        original_text="HELLO WORLD",
        normalized_text="hello world",
        demonstrations=[demo1, demo2],
    )

    output_path = tmp_path / "output_dictionary.mp4"
    stitcher = ISLVideoStitcher(output_fps=30.0, output_width=640, output_height=480)
    result = stitcher.stitch_sequence(sequence, output_path)

    assert isinstance(result, ISLStitchResult)
    assert Path(result.output_path).exists()
    assert result.total_clips == 2
    assert result.included_tokens == ["HELLO", "WORLD"]
    assert result.skipped_tokens == []
    assert result.total_frames == 30
    assert result.duration_sec == 1.0
    assert result.output_fps == 30.0
    assert result.output_width == 640
    assert result.output_height == 480

    # Verify readable by OpenCV
    cap = cv2.VideoCapture(result.output_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480
    cap.release()


def test_fingerspelling_clip_ordering(tmp_path: Path) -> None:
    """Tests that fingerspelling fallback clips are stitched in exact character order with frame content verification."""
    # Distinct solid BGR colors: P = RED (0,0,255), Y = GREEN (0,255,0), T = BLUE (255,0,0)
    c_p = create_synthetic_mp4(
        tmp_path / "P.mp4", num_frames=10, fps=30.0, color=(0, 0, 255)
    )
    c_y = create_synthetic_mp4(
        tmp_path / "Y.mp4", num_frames=10, fps=30.0, color=(0, 255, 0)
    )
    c_t = create_synthetic_mp4(
        tmp_path / "T.mp4", num_frames=10, fps=30.0, color=(255, 0, 0)
    )

    demo = ISLDemonstration(
        token="PYT",
        normalized_token="pyt",
        source=DemonstrationSource.FINGERSPELL,
        characters=["P", "Y", "T"],
        alphabet_video_paths=[str(c_p), str(c_y), str(c_t)],
    )

    sequence = ISLDemonstrationSequence(
        original_text="PYT",
        normalized_text="pyt",
        demonstrations=[demo],
    )

    output_path = tmp_path / "output_fingerspell.mp4"
    stitcher = ISLVideoStitcher(output_fps=30.0, output_width=320, output_height=240)
    result = stitcher.stitch_sequence(sequence, output_path)

    assert result.total_clips == 3
    assert result.included_tokens == ["PYT"]
    assert result.skipped_tokens == []
    assert result.total_frames == 30
    assert result.duration_sec == 1.0

    # Read output video frames and verify P -> Y -> T order by checking pixel colors
    cap = cv2.VideoCapture(result.output_path)
    assert cap.isOpened()
    output_frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        output_frames.append(frame)
    cap.release()

    assert len(output_frames) == 30

    # Frame 5 (Middle of P clip): expect RED color (BGR: B=0, G=0, R=255)
    frame_p = output_frames[5]
    assert frame_p[10, 10, 2] > 200  # Red channel
    assert frame_p[10, 10, 0] < 50   # Blue channel

    # Frame 15 (Middle of Y clip): expect GREEN color (BGR: B=0, G=255, R=0)
    frame_y = output_frames[15]
    assert frame_y[10, 10, 1] > 200  # Green channel
    assert frame_y[10, 10, 0] < 50   # Blue channel

    # Frame 25 (Middle of T clip): expect BLUE color (BGR: B=255, G=0, R=0)
    frame_t = output_frames[25]
    assert frame_t[10, 10, 0] > 200  # Blue channel
    assert frame_t[10, 10, 2] < 50   # Red channel


def test_mixed_dictionary_fingerspelling_unknown_input(tmp_path: Path) -> None:
    """Tests mixed sequence with dictionary, fingerspelled, and unknown demonstrations."""
    clip_dict = create_synthetic_mp4(tmp_path / "hello.mp4", num_frames=15)
    c_a = create_synthetic_mp4(tmp_path / "A.mp4", num_frames=10)
    c_b = create_synthetic_mp4(tmp_path / "B.mp4", num_frames=10)

    demo_dict = ISLDemonstration(
        token="HELLO",
        normalized_token="hello",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(clip_dict),
    )
    demo_finger = ISLDemonstration(
        token="AB",
        normalized_token="ab",
        source=DemonstrationSource.FINGERSPELL,
        characters=["A", "B"],
        alphabet_video_paths=[str(c_a), str(c_b)],
    )
    demo_unknown = ISLDemonstration(
        token="123",
        normalized_token="123",
        source=DemonstrationSource.UNKNOWN,
        characters=[],
        alphabet_video_paths=[],
    )

    sequence = ISLDemonstrationSequence(
        original_text="HELLO AB 123",
        normalized_text="hello ab 123",
        demonstrations=[demo_dict, demo_finger, demo_unknown],
    )

    output_path = tmp_path / "output_mixed.mp4"
    stitcher = ISLVideoStitcher()
    result = stitcher.stitch_sequence(sequence, output_path)

    assert result.total_clips == 3  # 1 dict clip + 2 alphabet clips
    assert result.included_tokens == ["HELLO", "AB"]
    assert result.skipped_tokens == ["123"]


def test_unknown_only_input_behavior(tmp_path: Path) -> None:
    """Tests graceful handling when all input tokens are UNKNOWN or missing without creating empty file."""
    demo_unk1 = ISLDemonstration(
        token="XYZ",
        normalized_token="xyz",
        source=DemonstrationSource.UNKNOWN,
    )
    demo_unk2 = ISLDemonstration(
        token="999",
        normalized_token="999",
        source=DemonstrationSource.UNKNOWN,
    )

    sequence = ISLDemonstrationSequence(
        original_text="XYZ 999",
        normalized_text="xyz 999",
        demonstrations=[demo_unk1, demo_unk2],
    )

    output_path = tmp_path / "output_unknown.mp4"
    stitcher = ISLVideoStitcher()
    result = stitcher.stitch_sequence(sequence, output_path)

    assert result.total_clips == 0
    assert result.total_frames == 0
    assert result.duration_sec == 0.0
    assert result.included_tokens == []
    assert result.skipped_tokens == ["XYZ", "999"]
    assert not Path(result.output_path).exists()  # File is NOT created on disk


def test_invalid_and_missing_source_clip_handling(tmp_path: Path) -> None:
    """Tests skipping missing or corrupt clip files without crashing."""
    valid_clip = create_synthetic_mp4(tmp_path / "valid.mp4", num_frames=15)
    invalid_clip = tmp_path / "missing_clip_123.mp4"
    corrupt_clip = tmp_path / "corrupt.mp4"
    corrupt_clip.write_text("not a video file", encoding="utf-8")

    demo_valid = ISLDemonstration(
        token="VALID",
        normalized_token="valid",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(valid_clip),
    )
    demo_missing = ISLDemonstration(
        token="MISSING",
        normalized_token="missing",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(invalid_clip),
    )
    demo_corrupt = ISLDemonstration(
        token="CORRUPT",
        normalized_token="corrupt",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(corrupt_clip),
    )

    sequence = ISLDemonstrationSequence(
        original_text="VALID MISSING CORRUPT",
        normalized_text="valid missing corrupt",
        demonstrations=[demo_valid, demo_missing, demo_corrupt],
    )

    output_path = tmp_path / "output_resilient.mp4"
    stitcher = ISLVideoStitcher()
    result = stitcher.stitch_sequence(sequence, output_path)

    assert result.total_clips == 1
    assert result.included_tokens == ["VALID"]
    assert result.skipped_tokens == ["MISSING", "CORRUPT"]


def test_differing_resolution_and_fps_normalization(tmp_path: Path) -> None:
    """Tests normalization of source clips with different resolutions and frame rates."""
    # Clip A: 15 FPS, 320x240, 15 frames (1.0 sec)
    clip_a = create_synthetic_mp4(
        tmp_path / "clip_a.mp4", num_frames=15, fps=15.0, width=320, height=240
    )
    # Clip B: 60 FPS, 1280x720, 60 frames (1.0 sec)
    clip_b = create_synthetic_mp4(
        tmp_path / "clip_b.mp4", num_frames=60, fps=60.0, width=1280, height=720
    )

    demo_a = ISLDemonstration(
        token="A",
        normalized_token="a",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(clip_a),
    )
    demo_b = ISLDemonstration(
        token="B",
        normalized_token="b",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(clip_b),
    )

    sequence = ISLDemonstrationSequence(
        original_text="A B",
        normalized_text="a b",
        demonstrations=[demo_a, demo_b],
    )

    output_path = tmp_path / "output_normalized.mp4"
    stitcher = ISLVideoStitcher(output_fps=30.0, output_width=640, output_height=480)
    result = stitcher.stitch_sequence(sequence, output_path)

    assert result.total_clips == 2
    assert result.output_fps == 30.0
    assert result.output_width == 640
    assert result.output_height == 480
    assert result.total_frames == 60
    assert result.duration_sec == 2.0

    cap = cv2.VideoCapture(result.output_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480
    cap.release()


def test_base_dir_relative_path_resolution(tmp_path: Path) -> None:
    """Tests resolving relative clip paths against a base_dir."""
    create_synthetic_mp4(tmp_path / "videos" / "hello.mp4", num_frames=15)

    demo = ISLDemonstration(
        token="HELLO",
        normalized_token="hello",
        source=DemonstrationSource.DICTIONARY,
        video_path="videos/hello.mp4",
    )
    sequence = ISLDemonstrationSequence(
        original_text="HELLO",
        normalized_text="hello",
        demonstrations=[demo],
    )

    output_path = tmp_path / "output_base_dir.mp4"
    stitcher = ISLVideoStitcher(base_dir=tmp_path)
    assert stitcher.validate_clip("videos/hello.mp4")

    result = stitcher.stitch_sequence(sequence, output_path)
    assert result.total_clips == 1
    assert result.included_tokens == ["HELLO"]


def test_videowriter_initialization_failure(tmp_path: Path) -> None:
    """Tests raising VideoProcessingError when cv2.VideoWriter fails to open."""
    clip = create_synthetic_mp4(tmp_path / "clip.mp4", num_frames=15)
    demo = ISLDemonstration(
        token="HELLO",
        normalized_token="hello",
        source=DemonstrationSource.DICTIONARY,
        video_path=str(clip),
    )
    sequence = ISLDemonstrationSequence(
        original_text="HELLO",
        normalized_text="hello",
        demonstrations=[demo],
    )

    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = False

    with patch("cv2.VideoWriter", return_value=mock_writer):
        stitcher = ISLVideoStitcher()
        with pytest.raises(VideoProcessingError, match="Failed to initialize cv2.VideoWriter"):
            stitcher.stitch_sequence(sequence, tmp_path / "fail_output.mp4")
    mock_writer.release.assert_called_once()
