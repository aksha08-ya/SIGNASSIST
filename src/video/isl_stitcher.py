"""ISL Video Demonstration Stitcher module for SIGNASSIST Phase 15 Block 3.

Stitches sequence of ISL dictionary video clips and fingerspelling alphabet clips
into a single normalized playable MP4 file using OpenCV.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from src.nlp.text_to_isl import DemonstrationSource, ISLDemonstrationSequence
from src.utils.exceptions import VideoProcessingError


@dataclass
class ISLStitchResult:
    """Structured result metadata for an ISL video stitching operation.

    Attributes:
        output_path: Absolute or relative path to the generated stitched MP4 video file.
        included_tokens: List of tokens whose video clips were successfully stitched.
        skipped_tokens: List of tokens whose clips were missing, skipped, or unknown.
        total_clips: Total number of individual source video clips stitched.
        total_frames: Total number of frames written to the output video.
        duration_sec: Total duration of the stitched video in seconds.
        output_fps: Output video frame rate in frames per second.
        output_width: Output frame width in pixels.
        output_height: Output frame height in pixels.
    """

    output_path: str
    included_tokens: List[str] = field(default_factory=list)
    skipped_tokens: List[str] = field(default_factory=list)
    total_clips: int = 0
    total_frames: int = 0
    duration_sec: float = 0.0
    output_fps: float = 30.0
    output_width: int = 640
    output_height: int = 480

    def __post_init__(self) -> None:
        self.output_path = str(self.output_path)
        self.total_clips = int(self.total_clips)
        self.total_frames = int(self.total_frames)
        self.duration_sec = float(self.duration_sec)
        self.output_fps = float(self.output_fps)
        self.output_width = int(self.output_width)
        self.output_height = int(self.output_height)


class ISLVideoStitcher:
    """Stitches ISLDemonstrationSequence video assets into a playable MP4 clip."""

    def __init__(
        self,
        output_fps: float = 30.0,
        output_width: int = 640,
        output_height: int = 480,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if output_fps <= 0:
            raise ValueError("output_fps must be positive")
        if output_width <= 0 or output_height <= 0:
            raise ValueError("output_width and output_height must be positive integers")

        self.output_fps = float(output_fps)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.base_dir = Path(base_dir) if base_dir else None

    def _resolve_clip_path(self, path_str: str) -> Path:
        """Resolves clip path using base_dir if relative."""
        p = Path(path_str)
        if p.is_absolute() or self.base_dir is None:
            return p
        return self.base_dir / p

    def validate_clip(self, clip_path: Union[str, Path]) -> bool:
        """Validates that a video clip exists, is non-empty, and can be read by OpenCV."""
        p = self._resolve_clip_path(str(clip_path))
        if not p.exists() or not p.is_file():
            return False
        if p.stat().st_size == 0:
            return False

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            cap.release()
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            return False

        return True

    def _read_and_resample_clip(
        self, clip_path: Path
    ) -> List[np.ndarray]:
        """Reads all frames from a clip and resamples/resizes them to target FPS and dimensions."""
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return []

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0 or math.isnan(src_fps):
            src_fps = self.output_fps

        raw_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            return []

        n_src = len(raw_frames)
        duration = n_src / src_fps
        n_out = max(1, int(round(duration * self.output_fps)))

        processed_frames: List[np.ndarray] = []
        for k in range(n_out):
            t = k / self.output_fps
            src_idx = min(n_src - 1, max(0, int(math.floor(t * src_fps))))
            frame = raw_frames[src_idx]

            h, w = frame.shape[:2]
            if w != self.output_width or h != self.output_height:
                frame = cv2.resize(
                    frame,
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            processed_frames.append(frame)

        return processed_frames

    def stitch_sequence(
        self,
        sequence: ISLDemonstrationSequence,
        output_path: Union[str, Path],
    ) -> ISLStitchResult:
        """Stitches an ISLDemonstrationSequence into a single MP4 video file.

        Args:
            sequence: The ISLDemonstrationSequence containing demonstrations to stitch.
            output_path: Destination path for the output stitched MP4 file.

        Returns:
            ISLStitchResult containing operational metadata and video properties.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        included_tokens: List[str] = []
        skipped_tokens: List[str] = []
        clips_to_stitch: List[Path] = []

        if sequence is not None and sequence.demonstrations:
            for demo in sequence.demonstrations:
                if demo.source == DemonstrationSource.DICTIONARY:
                    if demo.video_path and self.validate_clip(demo.video_path):
                        resolved = self._resolve_clip_path(demo.video_path)
                        clips_to_stitch.append(resolved)
                        included_tokens.append(demo.token)
                    else:
                        skipped_tokens.append(demo.token)

                elif demo.source == DemonstrationSource.FINGERSPELL:
                    if demo.alphabet_video_paths and all(
                        self.validate_clip(p) for p in demo.alphabet_video_paths
                    ):
                        for path_str in demo.alphabet_video_paths:
                            resolved = self._resolve_clip_path(path_str)
                            clips_to_stitch.append(resolved)
                        included_tokens.append(demo.token)
                    else:
                        skipped_tokens.append(demo.token)

                else:
                    # UNKNOWN or unrecognized demonstration source
                    skipped_tokens.append(demo.token)

        total_clips = len(clips_to_stitch)

        if total_clips == 0:
            return ISLStitchResult(
                output_path=str(out_p),
                included_tokens=included_tokens,
                skipped_tokens=skipped_tokens,
                total_clips=0,
                total_frames=0,
                duration_sec=0.0,
                output_fps=self.output_fps,
                output_width=self.output_width,
                output_height=self.output_height,
            )

        # OpenCV VideoWriter initialization
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_p),
            fourcc,
            self.output_fps,
            (self.output_width, self.output_height),
        )

        if not writer.isOpened():
            writer.release()
            raise VideoProcessingError(
                f"Failed to initialize cv2.VideoWriter for output path '{out_p}'"
            )

        total_written_frames = 0
        try:
            for clip_p in clips_to_stitch:
                frames = self._read_and_resample_clip(clip_p)
                for frame in frames:
                    writer.write(frame)
                    total_written_frames += 1
        finally:
            writer.release()

        duration_sec = round(total_written_frames / self.output_fps, 4)

        return ISLStitchResult(
            output_path=str(out_p),
            included_tokens=included_tokens,
            skipped_tokens=skipped_tokens,
            total_clips=total_clips,
            total_frames=total_written_frames,
            duration_sec=duration_sec,
            output_fps=self.output_fps,
            output_width=self.output_width,
            output_height=self.output_height,
        )
