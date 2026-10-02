"""UI Service layer for SIGNASSIST Phase 17 Block 2.

Provides deterministic, testable wrappers connecting backend modules
(VideoProcessor, URLVideoDownloader, TextToISLConverter, ISLVideoStitcher, MetricsEvaluator)
to the presentation layer without embedding business logic in app.py.

This module MUST NOT import Streamlit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.pipeline.recognition import RecognitionPipeline
from src.video.processor import VideoProcessingResult, VideoProcessor
from src.video.url_downloader import URLVideoDownloader
from src.video.isl_stitcher import ISLStitchResult, ISLVideoStitcher
from src.nlp.isl_dictionary import ISLDictionaryIndex
from src.nlp.text_to_isl import (
    DemonstrationSource,
    ISLDemonstrationSequence,
    TextToISLConverter,
)
from src.utils.metrics import EvaluationResult, MetricsEvaluator


# --- Private Helper: Default VideoProcessor Factory ---

# --- Private Helper: Default VideoProcessor Factory ---

def get_default_recognition_pipeline(
    mode: str = "dynamic_only",
    backend: str = "experimental_42",
    models_dir: Optional[Union[str, Path]] = None,
    dynamic_checkpoint_path: Optional[Union[str, Path]] = None,
    class_mapping_path: Optional[Union[str, Path]] = None,
    calibration_path: Optional[Union[str, Path]] = None,
    use_boundary_trimming: Optional[bool] = None,
) -> RecognitionPipeline:
    """Instantiates and returns the default RecognitionPipeline adapter."""
    if use_boundary_trimming is None:
        eff_trimming = False if backend == "experimental_42" else True
    else:
        eff_trimming = bool(use_boundary_trimming)

    return RecognitionPipeline(
        mode=mode,
        backend=backend,
        models_dir=models_dir,
        dynamic_checkpoint_path=dynamic_checkpoint_path,
        class_mapping_path=class_mapping_path,
        calibration_path=calibration_path,
        use_boundary_trimming=eff_trimming,
    )


def _create_default_video_processor(
    mode: str = "dynamic_only",
    backend: str = "experimental_42",
    models_dir: Optional[Union[str, Path]] = None,
    dynamic_checkpoint_path: Optional[Union[str, Path]] = None,
    class_mapping_path: Optional[Union[str, Path]] = None,
    calibration_path: Optional[Union[str, Path]] = None,
    use_boundary_trimming: Optional[bool] = None,
) -> VideoProcessor:
    """Creates default VideoProcessor initialized with RecognitionPipeline adapter."""
    pipeline = get_default_recognition_pipeline(
        mode=mode,
        backend=backend,
        models_dir=models_dir,
        dynamic_checkpoint_path=dynamic_checkpoint_path,
        class_mapping_path=class_mapping_path,
        calibration_path=calibration_path,
        use_boundary_trimming=use_boundary_trimming,
    )
    return VideoProcessor(recognition_pipeline=pipeline)



# --- Service 1: Video File Processing ---

def process_video_file(
    file_path: Union[str, Path],
    processor: Optional[VideoProcessor] = None,
    manual_signer_id: Optional[int] = None,
    enable_tts: bool = False,
    chunk_size: int = 60,
    mode: str = "dynamic_only",
    backend: str = "experimental_42",
    use_boundary_trimming: Optional[bool] = None,
) -> VideoProcessingResult:
    """Processes a local video file using the backend VideoProcessor.

    Args:
        file_path: Path to local video file.
        processor: Optional VideoProcessor instance. If None, default VideoProcessor is created.
        manual_signer_id: Optional manual signer track ID override.
        enable_tts: Whether to enable TTS synthesis for transcript text.
        chunk_size: Processing frame chunk size.
        mode: Pipeline mode ("dynamic_only" or "multimodal").
        backend: Recognition backend ("experimental_42" or "production").
        use_boundary_trimming: Whether to enable boundary trimming.

    Returns:
        VideoProcessingResult from the backend VideoProcessor.
    """
    if processor is None:
        processor = _create_default_video_processor(
            mode=mode,
            backend=backend,
            use_boundary_trimming=use_boundary_trimming,
        )

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
    mode: str = "dynamic_only",
    backend: str = "experimental_42",
    use_boundary_trimming: Optional[bool] = None,
) -> VideoProcessingResult:
    """Processes a public web video URL using backend URLVideoDownloader and VideoProcessor.

    Args:
        url: Web video URL (YouTube, Instagram, Facebook, Google Drive, Direct file link).
        downloader: Optional URLVideoDownloader instance.
        processor: Optional VideoProcessor instance. If None, default VideoProcessor is created.
        output_dir: Optional destination directory for downloaded video.
        manual_signer_id: Optional manual signer track ID override.
        enable_tts: Whether to enable TTS synthesis.
        chunk_size: Processing frame chunk size.
        mode: Pipeline mode ("dynamic_only" or "multimodal").
        backend: Recognition backend ("experimental_42" or "production").
        use_boundary_trimming: Whether to enable boundary trimming.

    Returns:
        VideoProcessingResult from backend downloader/processor pipeline.

    Raises:
        URLExtractionError: If URL extraction/download fails.
    """
    if downloader is None:
        downloader = URLVideoDownloader()

    if processor is None:
        processor = _create_default_video_processor(
            mode=mode,
            backend=backend,
            use_boundary_trimming=use_boundary_trimming,
        )

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
        if index_path is None:
            index_path = Path("data/external/isl_dictionary/index.json")
        dict_path = Path(index_path)
        if dict_path.exists():
            dictionary = ISLDictionaryIndex.load(dict_path)
            dict_base_dir = dict_path.parent
        else:
            dictionary = ISLDictionaryIndex(entries=[])
            dict_base_dir = None
    else:
        dict_base_dir = getattr(dictionary, "index_path", None)
        if dict_base_dir and Path(dict_base_dir).is_file():
            dict_base_dir = Path(dict_base_dir).parent

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
            stitcher = ISLVideoStitcher(base_dir=dict_base_dir)

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
    mode: str = "dynamic_only",
) -> Dict[str, Any]:
    """Determines model architecture, trained checkpoint, and recognition pipeline readiness.

    Strict semantics:
    - architecture_available: True (PyTorch ST-GCN, GNN, Transformer, Fusion modules exist)
    - checkpoint_available: True if trained checkpoint files exist on disk
    - recognition_pipeline_available: Evaluated via RecognitionPipeline.is_available() for requested mode
    """
    m_dir = Path(models_dir) if models_dir is not None else Path("models")

    if pipeline_instance is not None:
        rec_avail = (
            bool(pipeline_instance.is_available())
            if hasattr(pipeline_instance, "is_available") and callable(getattr(pipeline_instance, "is_available"))
            else False
        )
        pipe_mode = getattr(pipeline_instance, "mode", mode)
    elif explicit_pipeline_available is not None:
        rec_avail = bool(explicit_pipeline_available)
        pipe_mode = mode
    else:
        try:
            p = RecognitionPipeline(models_dir=m_dir, mode=mode)
            rec_avail = p.is_available()
            pipe_mode = p.mode
        except Exception:
            rec_avail = False
            pipe_mode = mode

    checkpoint_files = []
    if m_dir.exists() and m_dir.is_dir():
        for ext in ("*.pt", "*.pth", "*.bin", "*.ckpt"):
            checkpoint_files.extend(list(m_dir.glob(ext)))
        if (m_dir / "checkpoints").exists():
            for ext in ("*.pt", "*.pth", "*.bin", "*.ckpt"):
                checkpoint_files.extend(list((m_dir / "checkpoints").glob(f"**/{ext}")))

    checkpoint_files = sorted(list(set(checkpoint_files)))

    return {
        "architecture_available": True,
        "checkpoint_available": len(checkpoint_files) > 0,
        "recognition_pipeline_available": rec_avail,
        "mode": pipe_mode,
        "checkpoint_count": len(checkpoint_files),
        "models_dir": str(m_dir),
    }


# --- Service 6: Research Analytics Artifact Parsing ---

def parse_research_artifact(data: Any) -> Dict[str, Any]:
    """Parses and validates JSON evaluation result artifacts for Research Analytics dashboard.

    Supported types:
    - "evaluation": Standard EvaluationResult JSON (contains accuracy, macro_f1, per_class_metrics, confusion_matrix)
    - "loso": LOSOEvaluationResult JSON (contains evaluation_type="leave_one_signer_out" or folds & aggregate_metrics)
    - "ablation": AblationStudyResult JSON (contains experiments list or num_experiments)

    Returns:
        Dict containing:
            - "artifact_type": "evaluation" | "loso" | "ablation"
            - "payload": Dict data

    Raises:
        ValueError: If data structure is unsupported or missing required fields.
    """
    if not isinstance(data, dict):
        raise ValueError("JSON artifact must be a top-level dictionary object.")

    # 1. Ablation Study Result
    eval_type = str(data.get("evaluation_type", "")).lower()
    if "experiments" in data and isinstance(data["experiments"], list):
        if len(data["experiments"]) == 0:
            raise ValueError("Ablation study artifact contains no experiment results.")
        return {"artifact_type": "ablation", "payload": data}

    # 2. LOSO Evaluation Result
    if eval_type == "leave_one_signer_out" or ("folds" in data and "aggregate_metrics" in data):
        if not isinstance(data.get("folds"), list):
            raise ValueError("LOSO evaluation artifact is missing a valid 'folds' list.")
        if not isinstance(data.get("aggregate_metrics"), dict):
            raise ValueError("LOSO evaluation artifact is missing valid 'aggregate_metrics'.")
        return {"artifact_type": "loso", "payload": data}

    # 3. Standard Evaluation Result
    required_eval_keys = {"accuracy", "macro_f1", "per_class_metrics", "confusion_matrix"}
    if required_eval_keys.issubset(data.keys()):
        return {"artifact_type": "evaluation", "payload": data}

    # Single Ablation Result dictionary fallback
    if "name" in data and "parameters" in data and "metrics" in data:
        return {
            "artifact_type": "ablation",
            "payload": {
                "evaluation_type": "standard",
                "num_experiments": 1,
                "experiments": [data],
            },
        }

    raise ValueError(
        "Unsupported research evaluation artifact structure. Expected a valid EvaluationResult, "
        "LOSOEvaluationResult, or AblationStudyResult JSON file."
    )
