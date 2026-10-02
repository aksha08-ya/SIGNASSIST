"""Real local webcam session smoke test script for SIGNASSIST.

Consumes a real demo MP4 video from disk, decodes frame-by-frame using OpenCV,
and feeds the frames into WebcamRecognitionSession to verify real pipeline operation.
"""

import sys
import os
import time
from pathlib import Path
import cv2

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.services import get_default_recognition_pipeline
from src.ui.webcam import WebcamRecognitionSession


def main():
    print("=" * 65)
    print("SIGNASSIST WEBCAM SESSION SMOKE TEST (REAL VIDEO FRAME INGESTION)")
    print("=" * 65)

    # Locate verified demo asset
    demo_path = Path("data/raw/include/videos/Means_of_Transportation/10. Plane/MVI_3114.MOV")
    if not demo_path.exists():
        # Fallback search in data/
        matches = list(Path("data").glob("**/*.MOV")) + list(Path("data").glob("**/*.mp4"))
        if matches:
            demo_path = matches[0]
        else:
            print("ERROR: No demo video assets found in data/ directory.")
            sys.exit(1)

    print(f"Loading video asset: {demo_path}")

    # Initialize RecognitionPipeline and WebcamRecognitionSession
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    if not pipeline.is_available():
        print("ERROR: RecognitionPipeline is unavailable (checkpoints missing or standby).")
        sys.exit(1)

    session = WebcamRecognitionSession(recognition_pipeline=pipeline, window_size=60, stride=15)

    cap = cv2.VideoCapture(str(demo_path))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video file {demo_path}")
        sys.exit(1)

    frame_count = 0
    t0 = time.time()

    print("Feeding video frames into WebcamRecognitionSession...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        res = session.add_frame(frame)

        if frame_count % 15 == 0:
            latest = res.latest_prediction
            latest_gloss = "N/A"
            conf_pct = "N/A"
            if latest and latest.prediction:
                latest_gloss = getattr(latest.prediction, "predicted_gloss", "N/A")
                conf_val = getattr(latest.prediction, "confidence", 0.0)
                conf_float = float(conf_val.item()) if hasattr(conf_val, "item") else float(conf_val)
                conf_pct = f"{conf_float * 100:.1f}%"

            print(
                f"  [Frame {frame_count:03d}] Received: {res.total_frames_received:03d} | "
                f"Windows: {res.processed_windows} | FPS: {res.processing_fps:.1f} | "
                f"Latest: '{latest_gloss}' ({conf_pct})"
            )

    cap.release()
    t1 = time.time()

    final_res = session.get_result()
    print("\n" + "=" * 65)
    print("FINAL SESSION RESULT SUMMARY")
    print("=" * 65)
    print(f"Status:                      {final_res.status}")
    print(f"Total Frames Ingested:       {final_res.total_frames_received}")
    print(f"Processed 60-Frame Windows:  {final_res.processed_windows}")
    print(f"Average Ingestion FPS:       {final_res.processing_fps:.2f}")
    print(f"Last Window Latency:         {final_res.last_latency_ms:.1f} ms")
    print(f"Total Execution Time:        {t1 - t0:.2f} s")
    print(f"Prediction History Count:    {len(final_res.prediction_history)}")

    if final_res.latest_prediction and final_res.latest_prediction.prediction:
        p = final_res.latest_prediction.prediction
        print("\nLatest Prediction Details:")
        print(f"  Gloss:            {getattr(p, 'predicted_gloss', 'N/A')}")
        print(f"  Confidence:       {float(p.confidence.item() if hasattr(p.confidence, 'item') else p.confidence):.4f}")
        print(f"  Accepted:         {bool(p.is_accepted.item() if hasattr(p.is_accepted, 'item') else p.is_accepted)}")
        print(f"  Top-3 Glosses:    {getattr(p, 'top_k_glosses', [])}")
    print("=" * 65)

    assert final_res.total_frames_received > 0, "No frames were processed by session."
    assert final_res.processed_windows > 0, "No 60-frame inference windows were processed."
    print("SUCCESS: Real webcam smoke test passed cleanly!")


if __name__ == "__main__":
    main()
