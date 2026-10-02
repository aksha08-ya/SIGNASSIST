"""Video Processing Pipeline Orchestrator for SIGNASSIST (Phase 13).

Provides:
- TimestampedSignEntry dataclass
- VideoProcessingResult dataclass
- RecognitionPipelineAdapter protocol/adapter boundary
- VideoProcessor class orchestrating video loading, quality assessment,
  signer tracking (Phase 12), recognition adapter integration,
  sentence reconstruction (Phase 10), and TTS synthesis (Phase 11).
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np

from src.nlp.sentence_reconstructor import SentenceReconstructor
from src.nlp.types import BoundedSignPrediction, SentenceTranscript
from src.pipeline.boundary_detector import SignSegment
from src.tts.tts_engine import TTSAudioOutput, TTSEngine
from src.utils.config import AppConfig, VideoConfig
from src.utils.exceptions import VideoProcessingError
from src.utils.logger import setup_logger
from src.video.file_reader import DecodedFrame, VideoFileReader, VideoMetadata, format_timestamp
from src.video.quality import VideoQualityAssessor, VideoQualityReport
from src.video.signer_tracker import BoundingBox, SignerTracker, SignerTrackerOutput

logger = setup_logger(__name__)


@dataclass
class TimestampedSignEntry:
    """Represents a recognized sign token bound to frame indices and formatted timestamps."""

    start_frame: int
    end_frame: int
    start_timestamp_sec: float
    end_timestamp_sec: float
    start_timestamp: str  # Formatted MM:SS.mmm
    end_timestamp: str  # Formatted MM:SS.mmm
    recognized_gloss: str
    confidence: float
    top_3_candidates: List[Tuple[str, float]] = field(default_factory=list)
    primary_signer_track_id: Optional[int] = None
    class_index: Optional[int] = None
    is_accepted: bool = False

    def __post_init__(self) -> None:
        self.start_frame = int(self.start_frame)
        self.end_frame = int(self.end_frame)
        self.start_timestamp_sec = float(self.start_timestamp_sec)
        self.end_timestamp_sec = float(self.end_timestamp_sec)
        self.confidence = float(self.confidence)
        self.is_accepted = bool(self.is_accepted)
        if self.class_index is not None:
            self.class_index = int(self.class_index)
        if self.primary_signer_track_id is not None:
            self.primary_signer_track_id = int(self.primary_signer_track_id)


@dataclass
class VideoProcessingResult:
    """Master container holding the complete output of video file processing."""

    video_metadata: VideoMetadata
    quality_report: VideoQualityReport
    timeline: List[TimestampedSignEntry] = field(default_factory=list)
    transcript: Optional[SentenceTranscript] = None
    tts_audio: Optional[TTSAudioOutput] = None
    execution_seconds: float = 0.0
    processing_fps: float = 0.0
    frames_processed: int = 0
    recognition_available: bool = False
    additional_metadata: Dict[str, Any] = field(default_factory=dict)


class RecognitionPipelineAdapter(Protocol):
    """Protocol boundary for perception and model recognition integration."""

    def is_available(self) -> bool:
        """Returns True if trained model weights and recognition pipeline are ready."""
        ...

    def process_frames(
        self,
        frames: List[DecodedFrame],
        primary_tracks: List[Optional[BoundingBox]],
    ) -> List[BoundedSignPrediction]:
        """Processes decoded frame chunks and yields validated single-sample BoundedSignPredictions."""
        ...


class VideoProcessor:
    """Orchestrates video loading, quality assessment, tracking, recognition, reconstruction, and TTS."""

    def __init__(
        self,
        config: Optional[Union[VideoConfig, AppConfig]] = None,
        tracker: Optional[SignerTracker] = None,
        recognition_pipeline: Optional[Union[RecognitionPipelineAdapter, Callable]] = None,
        reconstructor: Optional[SentenceReconstructor] = None,
        tts_engine: Optional[TTSEngine] = None,
    ) -> None:
        if isinstance(config, AppConfig):
            self.config = config.video
            self.app_config = config
        elif isinstance(config, VideoConfig):
            self.config = config
            self.app_config = None
        elif config is None:
            self.config = VideoConfig()
            self.app_config = None
        else:
            raise TypeError(
                f"Expected config to be VideoConfig or AppConfig, got {type(config)}"
            )

        self.tracker = tracker if tracker is not None else SignerTracker(config=self.config)
        self.recognition_pipeline = recognition_pipeline
        self.reconstructor = (
            reconstructor if reconstructor is not None else SentenceReconstructor(config=self.app_config)
        )
        self.tts_engine = tts_engine if tts_engine is not None else TTSEngine(config=self.app_config)
        self.quality_assessor = VideoQualityAssessor(config=self.config)

    def process_file(
        self,
        file_path: Union[str, Path],
        manual_signer_id: Optional[int] = None,
        enable_tts: bool = False,
        chunk_size: int = 60,
    ) -> VideoProcessingResult:
        """Processes a local video file end-to-end through SIGNASSIST video pipeline."""
        start_time = time.perf_counter()

        reader = VideoFileReader(file_path=file_path, config=self.config)
        metadata = reader.load_metadata()

        # 1. Assess video quality
        quality_report = self.quality_assessor.assess_video(reader)

        # 2. Configure signer tracker
        self.tracker.reset()
        if manual_signer_id is not None:
            self.tracker.select_signer(manual_signer_id)

        # Check if recognition pipeline is supplied and available
        rec_available = False
        if self.recognition_pipeline is not None:
            if hasattr(self.recognition_pipeline, "is_available"):
                rec_available = bool(self.recognition_pipeline.is_available())
            else:
                rec_available = True

        timeline: List[TimestampedSignEntry] = []
        bounded_predictions: List[BoundedSignPrediction] = []
        frames_processed = 0

        # 3. Process video in chunks
        for chunk in reader.decode_frames_chunked(chunk_size=chunk_size):
            chunk_primary_tracks: List[Optional[BoundingBox]] = []
            last_tracker_output: Optional[SignerTrackerOutput] = None

            for df in chunk:
                frames_processed += 1
                tracker_output = self.tracker.process_frame([], frame_index=df.frame_index)
                last_tracker_output = tracker_output
                primary_bbox = (
                    tracker_output.primary_track.bbox if tracker_output.primary_track else None
                )
                chunk_primary_tracks.append(primary_bbox)

            if rec_available and self.recognition_pipeline is not None:
                try:
                    if callable(self.recognition_pipeline):
                        preds = self.recognition_pipeline(chunk, chunk_primary_tracks)
                    else:
                        preds = self.recognition_pipeline.process_frames(
                            chunk, chunk_primary_tracks
                        )

                    if preds:
                        for pred in preds:
                            if isinstance(pred, BoundedSignPrediction):
                                pred.validate()
                                bounded_predictions.append(pred)
                                entry = self._prediction_to_timestamped_entry(
                                    pred, metadata.native_fps, last_tracker_output
                                )
                                timeline.append(entry)
                except Exception as e:
                    logger.warning(f"Error during recognition chunk processing: {e}")

        # 4. Sentence Reconstruction (Phase 10)
        transcript: Optional[SentenceTranscript] = None
        if bounded_predictions and self.reconstructor is not None:
            segments = [bp.segment for bp in bounded_predictions]
            predictions = [bp.prediction for bp in bounded_predictions]
            transcript = self.reconstructor.process_batch(segments, predictions)
        elif not rec_available:
            transcript = None


        # 5. Text-to-Speech (Phase 11)
        tts_output: Optional[TTSAudioOutput] = None
        if (
            enable_tts
            and transcript is not None
            and transcript.text.strip()
            and self.tts_engine is not None
        ):
            try:
                tts_output = self.tts_engine.synthesize_transcript(transcript)
            except Exception as e:
                logger.warning(f"TTS synthesis failed during video processing: {e}")

        exec_time = time.perf_counter() - start_time
        proc_fps = float(frames_processed / exec_time) if exec_time > 0 else 0.0

        return VideoProcessingResult(
            video_metadata=metadata,
            quality_report=quality_report,
            timeline=timeline,
            transcript=transcript,
            tts_audio=tts_output,
            execution_seconds=exec_time,
            processing_fps=proc_fps,
            frames_processed=frames_processed,
            recognition_available=rec_available,
            additional_metadata={
                "chunk_size": chunk_size,
                "manual_signer_id": manual_signer_id,
            },
        )

    def _prediction_to_timestamped_entry(
        self,
        pred: BoundedSignPrediction,
        fps: float,
        tracker_output: Optional[SignerTrackerOutput] = None,
    ) -> TimestampedSignEntry:
        """Converts BoundedSignPrediction into TimestampedSignEntry."""
        seg = pred.segment
        p = pred.prediction

        start_frame = seg.start_frame
        end_frame = seg.end_frame
        start_sec = float(start_frame / fps)
        end_sec = float(end_frame / fps)

        start_ts = format_timestamp(start_sec)
        end_ts = format_timestamp(end_sec)

        cls_val = getattr(p, "predicted_class", 0)
        if hasattr(cls_val, "item"):
            cls_idx = int(cls_val.item())
        elif isinstance(cls_val, (list, tuple, np.ndarray)) and len(cls_val) > 0:
            cls_idx = int(cls_val[0])
        else:
            cls_idx = int(cls_val)

        gloss = getattr(p, "predicted_gloss", None) or str(cls_idx)
        conf_val = getattr(p, "confidence", 0.0)
        conf = float(conf_val.item()) if hasattr(conf_val, "item") else float(conf_val)

        acc_val = getattr(p, "is_accepted", conf >= 0.70)
        is_accepted = bool(acc_val.item()) if hasattr(acc_val, "item") else bool(acc_val)

        top_3: List[Tuple[str, float]] = []
        if p.top_k_indices is not None and p.top_k_probabilities is not None:
            indices = p.top_k_indices
            probs = p.top_k_probabilities
            glosses = getattr(p, "top_k_glosses", None)
            if hasattr(indices, "tolist"):
                indices = indices.tolist()
            if hasattr(probs, "tolist"):
                probs = probs.tolist()
            if indices and isinstance(indices[0], (list, tuple)):
                indices = indices[0]
            if probs and isinstance(probs[0], (list, tuple)):
                probs = probs[0]

            for idx, (cid, prob) in enumerate(zip(indices[:3], probs[:3])):
                g_label = glosses[idx] if (glosses and idx < len(glosses)) else str(cid)
                top_3.append((g_label, float(prob)))

        primary_id = (
            tracker_output.primary_track.track_id
            if (tracker_output and tracker_output.primary_track)
            else None
        )

        return TimestampedSignEntry(
            start_frame=start_frame,
            end_frame=end_frame,
            start_timestamp_sec=start_sec,
            end_timestamp_sec=end_sec,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            recognized_gloss=gloss,
            confidence=conf,
            top_3_candidates=top_3,
            primary_signer_track_id=primary_id,
            class_index=cls_idx,
            is_accepted=is_accepted,
        )
