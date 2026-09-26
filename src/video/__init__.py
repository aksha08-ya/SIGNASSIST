"""SIGNASSIST Video Package (Phase 12: Signer Detection & Tracking, Phase 13: Video Ingestion)."""

from src.video.file_reader import DecodedFrame, VideoFileReader, VideoMetadata
from src.video.processor import (
    RecognitionPipelineAdapter,
    TimestampedSignEntry,
    VideoProcessingResult,
    VideoProcessor,
)
from src.video.quality import VideoQualityAssessor, VideoQualityReport
from src.video.signer_tracker import (
    BoundingBox,
    KalmanBoxFilter,
    SignerTrack,
    SignerTracker,
    SignerTrackerOutput,
    TrackState,
)

__all__ = [
    "BoundingBox",
    "TrackState",
    "SignerTrack",
    "SignerTrackerOutput",
    "KalmanBoxFilter",
    "SignerTracker",
    "VideoMetadata",
    "DecodedFrame",
    "VideoFileReader",
    "VideoQualityReport",
    "VideoQualityAssessor",
    "TimestampedSignEntry",
    "VideoProcessingResult",
    "RecognitionPipelineAdapter",
    "VideoProcessor",
]