"""Real Text to ISL E2E verification smoke test script for SIGNASSIST."""

from pathlib import Path
import json
import cv2

from src.nlp.isl_dictionary import ISLDictionaryIndex
from src.ui.services import convert_text_to_isl


def main() -> None:
    print("=================================================================")
    print("REAL TEXT -> ISL SMOKE TEST & ASSET PROVENANCE VERIFICATION")
    print("=================================================================")

    dict_dir = Path("data/external/isl_dictionary")
    assert dict_dir.exists(), f"Missing dictionary directory: {dict_dir}"

    index = ISLDictionaryIndex.load(dict_dir, validate_paths=True)
    print(f"Loaded ISL Dictionary Index with {len(index.entries())} verified entries.")

    test_words = "plane cow slow xyz123unsupported"
    print(f"\nTesting Input Text: '{test_words}'")

    out_video_path = Path("/tmp/real_text_to_isl_smoke_test.mp4")
    if out_video_path.exists():
        out_video_path.unlink()

    res = convert_text_to_isl(
        text=test_words,
        dictionary=index,
        stitch_video=True,
        output_video_path=out_video_path,
    )

    print(f"\n--- Translation Summary ---")
    print(f"Original Text: '{res.sequence.original_text}'")
    print(f"Normalized Text: '{res.sequence.normalized_text}'")
    print(f"Dictionary Signs Matched: {res.dictionary_count}")
    print(f"Fingerspelled Words: {res.fingerspell_count}")
    print(f"Unknown / Unsupported Words: {res.unknown_count}")
    print(f"Included Tokens: {res.included_tokens}")
    print(f"Skipped / Missing Tokens: {res.skipped_tokens}")

    print(f"\n--- Individual Asset Provenance Verification ---")
    for d in res.sequence.demonstrations:
        if d.video_path:
            abs_p = dict_dir / d.video_path
            assert abs_p.exists(), f"Asset missing: {abs_p}"
            cap = cv2.VideoCapture(str(abs_p))
            assert cap.isOpened(), f"Cannot open video: {abs_p}"
            f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"  ✅ Matched Token '{d.token}' -> Gloss '{d.dictionary_gloss}'")
            print(f"     Path: {abs_p}")
            print(f"     Resolution: {w}x{h}, Frames: {f_count}")
        else:
            print(f"  ⚠️ Unsupported Token '{d.token}' -> Demonstration Unavailable (No fake clips generated)")

    print(f"\n--- Stitched Video Compilation Verification ---")
    assert out_video_path.exists(), f"Missing output video: {out_video_path}"
    assert out_video_path.stat().st_size > 0, "Output video is 0 bytes"

    cap_out = cv2.VideoCapture(str(out_video_path))
    assert cap_out.isOpened(), "Stitched MP4 cannot be opened by OpenCV"
    out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap_out.get(cv2.CAP_PROP_FPS)
    cap_out.release()

    print(f"  ✅ Stitched Output Video: {out_video_path}")
    print(f"     File Size: {out_video_path.stat().st_size} bytes")
    print(f"     Properties: {out_w}x{out_h} @ {out_fps} FPS, Total Frames: {out_frames}")
    print("\nALL REAL TEXT -> ISL SMOKE TEST ASSERTIONS PASSED CLEANLY!")


if __name__ == "__main__":
    main()
