"""
INCLUDE feature preprocessing for SIGNASSIST.

Converts clean INCLUDE videos into the canonical SIGNASSIST feature format:

    static_features : (225,)
    static_mask     : (75,)
    dynamic_tensor  : (1, 5, 60, 115)
    dynamic_mask    : (1, 60, 115)

Each video is stored independently as a compressed NPZ so preprocessing
is resumable and individual failures do not invalidate the full dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.extractor import MediaPipeHolisticExtractor
from src.features.normalizer import LandmarkNormalizer
from src.features.sequence import SequenceBuffer
from src.features.static_builder import StaticFeatureBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/include/manifest.csv"
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data/raw/include/videos"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/processed/include/features"
DEFAULT_LOG = PROJECT_ROOT / "data/processed/include/logs/preprocess.log"


def make_output_id(video_path: str) -> str:
    """Creates a stable filename from the relative video path."""
    return hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:24]


def process_video(
    video_path: Path,
    label_id: int,
    output_path: Path,
    extractor: MediaPipeHolisticExtractor,
    normalizer: LandmarkNormalizer,
    static_builder: StaticFeatureBuilder,
    sequence_length: int = 60,
) -> dict:
    """Extracts and serializes SIGNASSIST features from one video."""

    # Critical: temporal velocity state must never leak between videos.
    normalizer.reset()
    sequence = SequenceBuffer(sequence_length=sequence_length)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_metadata = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    processed_frames = 0
    valid_landmark_frames = 0
    midpoint_graph = None

    # Use a centered temporal window rather than always taking the final
    # 60 frames. For videos shorter than 60 frames, the full sequence is used.
    if frame_count_metadata > 0:
        midpoint_index = (frame_count_metadata - 1) // 2
        window_start = max(0, midpoint_index - sequence_length // 2)
        window_end = min(
            frame_count_metadata,
            window_start + sequence_length,
        )
        if window_end - window_start < sequence_length:
            window_start = max(0, window_end - sequence_length)
    else:
        midpoint_index = None
        window_start = 0
        window_end = sequence_length

    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            processed_frames += 1

            raw = extractor.extract(frame)
            graph = normalizer.normalize(raw)

            if graph.mask.any():
                valid_landmark_frames += 1

            # Select the actual temporal midpoint for the static branch.
            if midpoint_index is not None and processed_frames - 1 == midpoint_index:
                midpoint_graph = graph

            # Keep only the centered temporal window for the dynamic branch.
            frame_index = processed_frames - 1
            if window_start <= frame_index < window_end:
                sequence.add_frame(graph)

    finally:
        cap.release()

    if processed_frames == 0:
        raise RuntimeError(f"Video contained no readable frames: {video_path}")

    # SequenceBuffer provides canonical left-padding when the video has
    # fewer than sequence_length frames.
    dynamic = sequence.get_tensor(batch_size=1)

    if midpoint_graph is None:
        raise RuntimeError(f"No graph features produced: {video_path}")

    static = static_builder.build_from_graph(midpoint_graph)

    elapsed = time.perf_counter() - start_time

    np.savez_compressed(
        output_path,
        static_features=static.features.astype(np.float32),
        static_mask=static.mask.astype(bool),
        dynamic_tensor=dynamic.tensor.astype(np.float32),
        dynamic_mask=dynamic.mask.astype(bool),
        label_id=np.int64(label_id),
        valid_frames=np.int64(dynamic.valid_frames_count),
        processed_frames=np.int64(processed_frames),
        source_frame_count=np.int64(frame_count_metadata),
        fps=np.float32(fps),
    )

    return {
        "processed_frames": processed_frames,
        "source_frame_count": frame_count_metadata,
        "valid_landmark_frames": valid_landmark_frames,
        "valid_sequence_frames": dynamic.valid_frames_count,
        "fps": fps,
        "seconds": elapsed,
        "output": str(output_path),
    }


def load_mapping(path: Path) -> dict[str, int]:
    mapping = json.loads(path.read_text())
    return {
        str(label): int(class_id)
        for label, class_id in mapping["class_to_id"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "data/processed/include/class_mapping.json",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    # Only clean rows should enter the training feature set.
    df = df[df["status"].astype(str) == "clean"].copy()
    df = df.drop_duplicates(subset=["video_path"], keep="first")
    df = df.sort_values(["label", "video_path"]).reset_index(drop=True)

    label_to_id = load_mapping(args.mapping)

    # Verify manifest/mapping consistency before expensive processing.
    unknown_labels = sorted(set(df["label"].astype(str)) - set(label_to_id))
    if unknown_labels:
        raise ValueError(
            f"Manifest contains labels absent from class mapping: {unknown_labels[:10]}"
        )

    if args.limit > 0:
        df = df.head(args.limit)

    extractor = MediaPipeHolisticExtractor(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    normalizer = LandmarkNormalizer()
    static_builder = StaticFeatureBuilder()

    successes = 0
    skipped = 0
    failures = 0

    with args.log.open("a", encoding="utf-8") as log:
        try:
            for row in df.itertuples(index=False):
                relative_path = str(row.video_path)
                output_id = make_output_id(relative_path)
                output_path = args.output_root / f"{output_id}.npz"

                if output_path.exists() and output_path.stat().st_size > 0:
                    skipped += 1
                    print(f"[SKIP] {relative_path}")
                    continue

                video_path = args.video_root / relative_path
                label = str(row.label)
                label_id = label_to_id[label]

                print(f"[PROCESS] {relative_path} -> class {label_id}")

                try:
                    result = process_video(
                        video_path=video_path,
                        label_id=label_id,
                        output_path=output_path,
                        extractor=extractor,
                        normalizer=normalizer,
                        static_builder=static_builder,
                    )

                    successes += 1

                    log.write(
                        f"SUCCESS\t{relative_path}\tclass={label_id}\t"
                        f"frames={result['processed_frames']}\t"
                        f"valid={result['valid_landmark_frames']}\t"
                        f"seconds={result['seconds']:.2f}\n"
                    )
                    log.flush()

                    print(
                        f"  frames={result['processed_frames']} "
                        f"valid={result['valid_landmark_frames']} "
                        f"time={result['seconds']:.2f}s"
                    )

                except Exception as exc:
                    failures += 1
                    message = f"FAIL\t{relative_path}\t{type(exc).__name__}: {exc}\n"
                    log.write(message)
                    log.flush()
                    print(f"  [FAILED] {exc}")

        finally:
            extractor.close()

    print("\n===== INCLUDE PREPROCESSING SUMMARY =====")
    print(f"Requested : {len(df)}")
    print(f"Processed : {successes}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failures}")
    print(f"Output    : {args.output_root}")
    print(f"Log       : {args.log}")


if __name__ == "__main__":
    main()
