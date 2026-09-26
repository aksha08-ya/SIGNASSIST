"""Video File Reader & Metadata Extractor for SIGNASSIST (Phase 13).

Provides:
- VideoMetadata typed dataclass
- DecodedFrame typed dataclass
- VideoFileReader for OpenCV video loading, metadata extraction, validation,
  lazy frame decoding, chunking, and timestamp generation.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Union
import cv2
import numpy as np

from src.utils.config import AppConfig, VideoConfig
from src.utils.exceptions import VideoProcessingError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS.mmm timestamp string."""
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        seconds = 0.0

    total_millis = int(round(seconds * 1000))
    minutes = total_millis // 60000
    remainder = total_millis % 60000
    secs = remainder // 1000
    millis = remainder % 1000

    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class VideoMetadata:
    """Structured container holding extracted video file metadata."""

    file_path: str
    file_name: str
    file_size_bytes: int
    duration_sec: float
    total_frames: int
    native_fps: float
    width: int
    height: int
    aspect_ratio: float
    codec: str
    frame_count_consistent: bool = True


@dataclass
class DecodedFrame:
    """Container for a single decoded video frame and its temporal metadata."""

    frame: np.ndarray
    frame_index: int
    timestamp_sec: float
    timestamp: str  # Formatted MM:SS.mmm


class VideoFileReader:
    """Reads, decodes, validates, and samples local video files using OpenCV."""

    def __init__(
        self,
        file_path: Union[str, Path],
        config: Optional[Union[VideoConfig, AppConfig]] = None,
    ) -> None:
        self.file_path = Path(file_path).resolve()

        if isinstance(config, AppConfig):
            self.config = config.video
        elif isinstance(config, VideoConfig):
            self.config = config
        elif config is None:
            self.config = VideoConfig()
        else:
            raise TypeError(
                f"Expected config to be VideoConfig or AppConfig, got {type(config)}"
            )

        self._metadata: Optional[VideoMetadata] = None

    def validate_file(self) -> None:
        """Validates video file existence, extension, readability, FPS, and max duration."""
        if not self.file_path.exists():
            raise VideoProcessingError(
                f"Video file does not exist: '{self.file_path}'",
                details={"file_path": str(self.file_path)},
            )
        if not self.file_path.is_file():
            raise VideoProcessingError(
                f"Specified path is not a file: '{self.file_path}'",
                details={"file_path": str(self.file_path)},
            )

        ext = self.file_path.suffix.lower()
        supported = [e.lower() for e in self.config.supported_extensions]
        if ext not in supported:
            raise VideoProcessingError(
                f"Unsupported video file extension '{ext}'. Supported: {supported}",
                details={
                    "file_path": str(self.file_path),
                    "extension": ext,
                    "supported": supported,
                },
            )

        cap = cv2.VideoCapture(str(self.file_path))
        try:
            if not cap.isOpened():
                raise VideoProcessingError(
                    f"Failed to open video file (corrupt or unreadable): '{self.file_path}'",
                    details={"file_path": str(self.file_path)},
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or math.isnan(fps) or math.isinf(fps):
                raise VideoProcessingError(
                    f"Invalid FPS ({fps}) in video file: '{self.file_path}'",
                    details={"file_path": str(self.file_path), "fps": fps},
                )

            if total_frames <= 0:
                raise VideoProcessingError(
                    f"Invalid frame count ({total_frames}) in video file: '{self.file_path}'",
                    details={"file_path": str(self.file_path), "total_frames": total_frames},
                )

            duration = total_frames / fps
            if duration > self.config.max_video_length_sec:
                raise VideoProcessingError(
                    f"Video duration ({duration:.1f}s) exceeds max configured limit ({self.config.max_video_length_sec}s).",
                    details={
                        "file_path": str(self.file_path),
                        "duration_sec": duration,
                        "max_allowed_sec": self.config.max_video_length_sec,
                    },
                )
        finally:
            cap.release()

    def load_metadata(self) -> VideoMetadata:
        """Extracts and caches detailed video metadata."""
        if self._metadata is not None:
            return self._metadata

        self.validate_file()

        cap = cv2.VideoCapture(str(self.file_path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

            codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()
            if not codec or any(ord(c) < 32 or ord(c) > 126 for c in codec):
                codec = "UNKNOWN"

            duration_sec = float(total_frames / fps) if fps > 0 else 0.0
            aspect_ratio = float(width / height) if height > 0 else 0.0
            file_size_bytes = self.file_path.stat().st_size

            self._metadata = VideoMetadata(
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                file_size_bytes=file_size_bytes,
                duration_sec=duration_sec,
                total_frames=total_frames,
                native_fps=fps,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                codec=codec,
                frame_count_consistent=True,
            )
            return self._metadata
        finally:
            cap.release()

    def decode_frames(
        self,
        stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[DecodedFrame, None, None]:
        """Lazy generator yielding decoded BGR frames with exact native timestamps.

        Args:
            stride: Step size between yielded frames (default 1 = every frame).
            max_frames: Optional maximum number of frames to yield.

        Yields:
            DecodedFrame containing BGR image array, frame index, timestamp_sec, and timestamp string.
        """
        metadata = self.load_metadata()
        fps = metadata.native_fps

        cap = cv2.VideoCapture(str(self.file_path))
        try:
            frame_idx = 0
            yielded_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % stride == 0:
                    ts_sec = float(frame_idx / fps)
                    ts_str = format_timestamp(ts_sec)

                    yield DecodedFrame(
                        frame=frame,
                        frame_index=frame_idx,
                        timestamp_sec=ts_sec,
                        timestamp=ts_str,
                    )
                    yielded_count += 1
                    if max_frames is not None and yielded_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

    def decode_frames_chunked(
        self,
        chunk_size: int = 60,
        stride: int = 1,
    ) -> Generator[List[DecodedFrame], None, None]:
        """Lazy generator yielding lists of DecodedFrame objects in chunks of chunk_size."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive integer > 0, got {chunk_size}")

        chunk: List[DecodedFrame] = []
        for decoded_frame in self.decode_frames(stride=stride):
            chunk.append(decoded_frame)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk

    def sample_frames(
        self,
        num_samples: int = 10,
    ) -> List[DecodedFrame]:
        """Samples up to num_samples evenly spaced frames across video without loading full video into memory."""
        metadata = self.load_metadata()
        total = metadata.total_frames
        if total <= 0:
            return []

        if num_samples >= total:
            return list(self.decode_frames())

        indices = np.linspace(0, total - 1, num_samples, dtype=int)
        target_indices = set(int(idx) for idx in indices)

        samples: List[DecodedFrame] = []
        fps = metadata.native_fps

        cap = cv2.VideoCapture(str(self.file_path))
        try:
            curr_idx = 0
            while cap.isOpened() and len(samples) < len(target_indices):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if curr_idx in target_indices:
                    ts_sec = float(curr_idx / fps)
                    samples.append(
                        DecodedFrame(
                            frame=frame,
                            frame_index=curr_idx,
                            timestamp_sec=ts_sec,
                            timestamp=format_timestamp(ts_sec),
                        )
                    )
                curr_idx += 1
            return samples
        finally:
            cap.release()
