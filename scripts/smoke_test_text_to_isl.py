"""Smoke test script for SIGNASSIST Phase 15 Text-to-ISL Demonstration Engine Pipeline."""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def main() -> None:
    print("=== SIGNASSIST Phase 15 Text-to-ISL Demonstration Engine Smoke Test ===")

    tmp_dir = Path(tempfile.mkdtemp(prefix="signassist_smoke_isl_"))
    try:
        # Step 1: Create synthetic dictionary & alphabet video clips
        print("\n[1/5] Setting up synthetic dictionary entries and alphabet assets...")
        dict_video = create_synthetic_mp4(
            tmp_dir / "videos" / "hello.mp4", num_frames=15, color=(0, 0, 255)
        )

        alphabet_map: dict[str, str] = {}
        for char in ["P", "Y", "T", "H", "O", "N"]:
            alp_path = create_synthetic_mp4(
                tmp_dir / "videos" / "alphabet" / f"{char}.mp4",
                num_frames=10,
                color=(0, 255, 0),
            )
            alphabet_map[char] = f"videos/alphabet/{char}.mp4"

        # Create dictionary index
        entry_hello = ISLDictionaryEntry(
            gloss="HELLO",
            video_path="videos/hello.mp4",
            aliases=["hi"],
            category="greetings",
            description="Standard wave greeting",
        )
        dict_index = ISLDictionaryIndex(entries=[entry_hello])
        print("      Synthetic dictionary & alphabet setup completed!")

        # Step 2: Convert text phrase to demonstration sequence
        print("\n[2/5] Converting text phrase to ISLDemonstrationSequence...")
        converter = TextToISLConverter(
            dictionary=dict_index, alphabet_video_mapping=alphabet_map
        )
        input_text = "Hello, python 123!"
        seq = converter.convert(input_text)

        assert seq.original_text == input_text
        assert seq.dictionary_count == 1
        assert seq.fingerspell_count == 1
        assert seq.unknown_count == 1
        assert len(seq.demonstrations) == 3

        demo_hello = seq.demonstrations[0]
        assert demo_hello.token == "Hello"
        assert demo_hello.normalized_token == "hello"
        assert demo_hello.source == DemonstrationSource.DICTIONARY
        assert demo_hello.video_path == "videos/hello.mp4"
        assert demo_hello.alphabet_video_paths == []

        demo_python = seq.demonstrations[1]
        assert demo_python.token == "python"
        assert demo_python.normalized_token == "python"
        assert demo_python.source == DemonstrationSource.FINGERSPELL
        assert demo_python.video_path is None
        assert len(demo_python.alphabet_video_paths) == 6

        demo_123 = seq.demonstrations[2]
        assert demo_123.token == "123"
        assert demo_123.normalized_token == "123"
        assert demo_123.source == DemonstrationSource.UNKNOWN
        assert demo_123.video_path is None
        assert demo_123.alphabet_video_paths == []
        print("      TextToISLConverter conversion passed!")

        # Step 3: Stitch demonstration sequence to MP4
        print("\n[3/5] Stitching demonstration sequence into playable MP4...")
        output_mp4 = tmp_dir / "output_stitched.mp4"
        stitcher = ISLVideoStitcher(
            output_fps=30.0, output_width=640, output_height=480, base_dir=tmp_dir
        )
        result = stitcher.stitch_sequence(seq, output_mp4)

        assert isinstance(result, ISLStitchResult)
        assert Path(result.output_path).exists()
        assert result.included_tokens == ["Hello", "python"]
        assert result.skipped_tokens == ["123"]
        assert result.total_clips == 1 + 6  # 1 dictionary clip + 6 letter clips
        assert result.total_frames == 15 + (6 * 10)  # 75 frames total
        assert result.duration_sec == 2.5
        assert result.output_fps == 30.0
        assert result.output_width == 640
        assert result.output_height == 480
        print("      ISLVideoStitcher stitching passed!")

        # Step 4: Validate output video with OpenCV VideoCapture
        print("\n[4/5] Validating output MP4 using OpenCV VideoCapture...")
        cap = cv2.VideoCapture(result.output_path)
        assert cap.isOpened(), "Failed to open output stitched video"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert width == 640
        assert height == 480

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_count += 1
        cap.release()

        assert frame_count == result.total_frames == 75
        print(
            f"      OpenCV validation passed! Total frames verified: {frame_count} ({result.duration_sec}s)"
        )

        # Step 5: Verify zero valid clips handling (unknown only)
        print("\n[5/5] Verifying zero valid clips / unknown-only non-touch behavior...")
        unk_seq = converter.convert("XYZ 999")
        unk_output_mp4 = tmp_dir / "output_unk.mp4"
        unk_result = stitcher.stitch_sequence(unk_seq, unk_output_mp4)

        assert unk_result.total_clips == 0
        assert unk_result.total_frames == 0
        assert unk_result.duration_sec == 0.0
        assert unk_result.included_tokens == []
        assert unk_result.skipped_tokens == ["XYZ", "999"]
        assert not Path(unk_result.output_path).exists()
        print("      Zero-clip handling verified!")

        print(
            "\n=== Phase 15 Text-to-ISL Demonstration Engine Smoke Test Passed Successfully! ==="
        )

    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
