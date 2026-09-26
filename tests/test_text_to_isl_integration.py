"""Integration test for SIGNASSIST Phase 15 Text-to-ISL Demonstration Engine."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.nlp import (
    DemonstrationSource,
    ISLDemonstrationSequence,
    ISLDictionaryEntry,
    ISLDictionaryIndex,
    TextToISLConverter,
)
from src.video import ISLStitchResult, ISLVideoStitcher


def create_synthetic_mp4(
    path: Path,
    num_frames: int = 15,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Path:
    """Helper utility generating valid synthetic MP4 video clips at runtime."""
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


def test_full_text_to_isl_pipeline_integration(tmp_path: Path) -> None:
    """End-to-end integration test: Text -> Converter -> Sequence -> Stitcher -> Validated MP4."""
    # 1. Setup synthetic video dictionary & alphabet clips
    create_synthetic_mp4(
        tmp_path / "videos" / "hello.mp4", num_frames=15, fps=30.0
    )

    alphabet_map: dict[str, str] = {}
    for char in ["P", "Y", "T", "H", "O", "N"]:
        create_synthetic_mp4(
            tmp_path / "videos" / "alphabet" / f"{char}.mp4",
            num_frames=10,
            fps=30.0,
        )
        alphabet_map[char] = f"videos/alphabet/{char}.mp4"

    # 2. Setup ISLDictionaryIndex
    entry_hello = ISLDictionaryEntry(
        gloss="HELLO",
        video_path="videos/hello.mp4",
        aliases=["hi"],
    )
    dict_index = ISLDictionaryIndex(entries=[entry_hello])

    # 3. Convert text phrase
    converter = TextToISLConverter(
        dictionary=dict_index, alphabet_video_mapping=alphabet_map
    )
    sequence = converter.convert("Hello, python 999!")

    assert sequence.original_text == "Hello, python 999!"
    assert sequence.dictionary_count == 1
    assert sequence.fingerspell_count == 1
    assert sequence.unknown_count == 1

    # 4. Stitch sequence into playable MP4
    output_mp4 = tmp_path / "output_pipeline_integration.mp4"
    stitcher = ISLVideoStitcher(
        output_fps=30.0, output_width=640, output_height=480, base_dir=tmp_path
    )
    result = stitcher.stitch_sequence(sequence, output_mp4)

    assert isinstance(result, ISLStitchResult)
    assert Path(result.output_path).exists()
    assert result.included_tokens == ["Hello", "python"]
    assert result.skipped_tokens == ["999"]
    assert result.total_clips == 7
    assert result.total_frames == 75
    assert result.duration_sec == 2.5

    # 5. OpenCV Validation
    cap = cv2.VideoCapture(result.output_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
    cap.release()

    assert frame_count == 75
