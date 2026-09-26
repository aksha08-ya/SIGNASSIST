"""UI Service layer for SIGNASSIST Phase 17 Block 2.

Provides deterministic, testable wrappers connecting backend modules
(VideoProcessor, URLVideoDownloader, TextToISLConverter, ISLVideoStitcher, MetricsEvaluator)
to the presentation layer without embedding business logic in app.py.

This module MUST NOT import Streamlit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.nlp.isl_dictionary import ISLDictionaryIndex
from src.nlp.text_to_isl import (
    DemonstrationSource,
    ISLDemonstrationSequence,
    TextToISLConverter,
)
from src.utils.metrics import EvaluationResult, MetricsEvaluator
from src.video.isl_stitcher import ISLStitchResult, ISLVideoStitcher
from src.video.processor import VideoProcessingResult, VideoProcessor
from src.video.url_downloader import URLVideoDownloader


# --- Service 1: Video File Processing ---

def process_video_file(
    file_path: Union[str, Path],
    processor: Optional[VideoProcessor] = None,
    manual_signer_id: Optional[int] = None,
    enable_tts: bool = False,
    chunk_size: int = 60,
) -> VideoProcessingResult:
    """Processes a local video file using the backend VideoProcessor.

    Args:
        file_path: Path to local video file.
        processor: Optional VideoProcessor instance. If None, default VideoProcessor is created.
        manual_signer_id: Optional manual signer track ID override.
        enable_tts: Whether to enable TTS synthesis for transcript text.
        chunk_size: Processing frame chunk size.

    Returns:
        VideoProcessingResult from the backend VideoProcessor.
    """
    if processor is None:
        processor = VideoProcessor()

    return processor.process_file(
        file_path=file_path,
        manual_signer_id=manual_signer_id,
        enable_tts=enable_tts,
        chunk_size=chunk_size,
    )


# --- Service 2: Video URL Processing ---

def process_video_url(
    url: str,
    downloader: Optional[URLVideoDownloader] = None,
    processor: Optional[VideoProcessor] = None,
    output_dir: Optional[Union[str, Path]] = None,
    manual_signer_id: Optional[int] = None,
    enable_tts: bool = False,
    chunk_size: int = 60,
) -> VideoProcessingResult:
    """Processes a public web video URL using backend URLVideoDownloader and VideoProcessor.

    Args:
        url: Web video URL (YouTube, Instagram, Facebook, Google Drive, Direct file link).
        downloader: Optional URLVideoDownloader instance.
        processor: Optional VideoProcessor instance.
        output_dir: Optional destination directory for downloaded video.
        manual_signer_id: Optional manual signer track ID override.
        enable_tts: Whether to enable TTS synthesis.
        chunk_size: Processing frame chunk size.

    Returns:
        VideoProcessingResult from backend downloader/processor pipeline.

    Raises:
        URLExtractionError: If URL extraction/download fails.
    """
    if downloader is None:
        downloader = URLVideoDownloader()

    return downloader.process_url(
        url=url,
        processor=processor,
        output_dir=output_dir,
        manual_signer_id=manual_signer_id,
        enable_tts=enable_tts,
        chunk_size=chunk_size,
    )


# --- Service 3: Text → ISL Conversion ---

@dataclass
class TextToISLResult:
    """Container for UI-facing Text to ISL conversion result."""

    sequence: ISLDemonstrationSequence
    stitch_result: Optional[ISLStitchResult]
    output_video_path: Optional[str]
    included_tokens: List[str]
    skipped_tokens: List[str]
    dictionary_count: int
    fingerspell_count: int
    unknown_count: int


def convert_text_to_isl(
    text: str,
    dictionary: Optional[ISLDictionaryIndex] = None,
    index_path: Optional[Union[str, Path]] = None,
    alphabet_video_mapping: Optional[Dict[str, str]] = None,
    stitch_video: bool = False,
    output_video_path: Optional[Union[str, Path]] = None,
    stitcher: Optional[ISLVideoStitcher] = None,
) -> TextToISLResult:
    """Translates input text into ISL demonstration sequence and optional stitched video clip.

    Args:
        text: Input natural language text string.
        dictionary: Optional pre-loaded ISLDictionaryIndex instance.
        index_path: Optional path to JSON dictionary index file (used if dictionary is None).
        alphabet_video_mapping: Optional Dict mapping letters 'A'-'Z' to clip file paths.
        stitch_video: Whether to stitch sequence clips into an output MP4 video.
        output_video_path: Destination path for output stitched MP4 video file.
        stitcher: Optional ISLVideoStitcher instance.

    Returns:
        TextToISLResult dataclass containing sequence, counts, tokens, and optional video path.
    """
    # 1. Resolve or instantiate ISLDictionaryIndex
    if dictionary is None:
        if index_path is not None and Path(index_path).exists():
            dictionary = ISLDictionaryIndex.load(index_path)
        else:
            dictionary = ISLDictionaryIndex(entries=[])

    # 2. Convert text to ISLDemonstrationSequence
    converter = TextToISLConverter(
        dictionary=dictionary,
        alphabet_video_mapping=alphabet_video_mapping,
    )
    sequence: ISLDemonstrationSequence = converter.convert(text)

    # 3. Optional video stitching
    stitch_res: Optional[ISLStitchResult] = None
    real_out_path: Optional[str] = None
    included_tokens: List[str] = []
    skipped_tokens: List[str] = [
        d.token for d in sequence.demonstrations if d.source == DemonstrationSource.UNKNOWN
    ]

    if stitch_video and output_video_path is not None:
        if stitcher is None:
            stitcher = ISLVideoStitcher()

        stitch_res = stitcher.stitch_sequence(sequence, output_path=output_video_path)
        included_tokens = list(stitch_res.included_tokens)
        skipped_tokens = list(stitch_res.skipped_tokens)

        # Only expose output_video_path if clips were actually stitched and file exists
        if stitch_res.total_clips > 0 and Path(output_video_path).exists():
            real_out_path = str(output_video_path)

    return TextToISLResult(
        sequence=sequence,
        stitch_result=stitch_res,
        output_video_path=real_out_path,
        included_tokens=included_tokens,
        skipped_tokens=skipped_tokens,
        dictionary_count=sequence.dictionary_count,
        fingerspell_count=sequence.fingerspell_count,
        unknown_count=sequence.unknown_count,
    )


# --- Service 4: Research Analytics ---

def evaluate_classification(
    y_true: Any,
    y_pred: Any,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    zero_division: float = 0.0,
) -> EvaluationResult:
    """Evaluates multiclass classification metrics using MetricsEvaluator.

    Args:
        y_true: Ground truth integer-like labels.
        y_pred: Predicted integer-like labels.
        num_classes: Optional explicit number of target classes.
        class_names: Optional list of class string names.
        zero_division: Fallback metric value when division by zero occurs.

    Returns:
        EvaluationResult dataclass with overall, per-class, and matrix results.
    """
    evaluator = MetricsEvaluator(zero_division=zero_division)
    return evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )


# --- Service 5: Model / Recognition Readiness ---

def get_recognition_readiness(
    models_dir: Optional[Union[str, Path]] = None,
    pipeline_instance: Optional[Any] = None,
    explicit_pipeline_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Determines model architecture, trained checkpoint, and recognition pipeline readiness.

    Strict semantics:
    - architecture_available: True (PyTorch ST-GCN, GNN, Transformer, Fusion modules exist)
    - checkpoint_available: True only if trained checkpoint files exist on disk
    - recognition_pipeline_available: False unless an explicit usable pipeline is verified/supplied

    Args:
        models_dir: Optional directory path to check for trained model checkpoints.
        pipeline_instance: Optional instantiated pipeline/adapter object.
        explicit_pipeline_available: Optional explicit boolean flag for pipeline readiness.

    Returns:
        Dictionary containing detailed availability status flags.
    """
    m_dir = Path(models_dir) if models_dir is not None else Path("models")

    checkpoint_files = []
    if m_dir.exists() and m_dir.is_dir():
        for ext in ("*.pt", "*.pth", "*.bin", "*.ckpt"):
            checkpoint_files.extend(list(m_dir.glob(ext)))

    checkpoint_files = sorted(list(set(checkpoint_files)))
    checkpoint_available = len(checkpoint_files) > 0
    architecture_available = True

    if explicit_pipeline_available is not None:
        recognition_pipeline_available = bool(explicit_pipeline_available)
    elif (
        pipeline_instance is not None
        and hasattr(pipeline_instance, "is_available")
        and callable(getattr(pipeline_instance, "is_available"))
    ):
        recognition_pipeline_available = bool(pipeline_instance.is_available())
    else:
        recognition_pipeline_available = False

    return {
        "architecture_available": architecture_available,
        "checkpoint_available": checkpoint_available,
        "recognition_pipeline_available": recognition_pipeline_available,
        "checkpoint_count": len(checkpoint_files),
        "models_dir": str(m_dir),
    }
