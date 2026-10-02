import tempfile
from pathlib import Path
import cv2
import pytest

from src.nlp.text_to_isl import (
    DemonstrationSource,
    ISLDemonstration,
    ISLDemonstrationSequence,
)
from src.video.isl_stitcher import ISLVideoStitcher


def test_isl_stitcher_h264_transcoding_and_validation():
    """Verify that ISLVideoStitcher creates a valid, non-zero duration,

    H.264 web-compatible MP4 clip with at least one decodable frame.
    """
    dict_dir = Path("data/external/isl_dictionary")
    stitcher = ISLVideoStitcher(base_dir=dict_dir)

    # Use Plane clip ('10. Plane')
    demo = ISLDemonstration(
        token="plane",
        normalized_token="plane",
        source=DemonstrationSource.DICTIONARY,
        video_path="videos/class_000_MVI_3113.MOV",
        dictionary_gloss="10. Plane",
    )
    sequence = ISLDemonstrationSequence(
        original_text="plane",
        normalized_text="plane",
        demonstrations=[demo],
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = Path(tmp.name)

    try:
        stitch_res = stitcher.stitch_sequence(sequence, out_path)
        assert stitch_res.total_clips == 1
        assert stitch_res.total_frames > 0
        assert stitch_res.duration_sec > 0.0
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Media inspection via OpenCV
        cap = cv2.VideoCapture(str(out_path))
        assert cap.isOpened(), f"Generated MP4 video '{out_path}' is unreadable"
        ret, frame = cap.read()
        cap.release()

        assert ret is True, "Failed to decode frame from generated MP4"
        assert frame is not None and frame.size > 0, "Decoded frame is empty"
    finally:
        if out_path.exists():
            out_path.unlink()
