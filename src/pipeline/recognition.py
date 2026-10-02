"""
End-to-End Recognition Pipeline for SIGNASSIST (Phase 17 Block 6).

Integrates feature extraction, landmark normalization, static GCN classification,
dynamic ST-GCN + Transformer classification, learned routing, multimodal fusion,
confidence estimation with rejection thresholding, and continuous boundary detection.

Supports modes:
- mode="multimodal"   (default: requires static_gcn, dynamic_classifier, router, fusion checkpoints)
- mode="dynamic_only" (requires ONLY dynamic classifier checkpoint & class_mapping.json)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.features.extractor import MediaPipeHolisticExtractor
from src.features.normalizer import LandmarkNormalizer
from src.features.sequence import SequenceBuffer
from src.features.static_builder import StaticFeatureBuilder
from src.features.types import RawLandmarks, SelectedGraphLandmarks, StaticFeatureVector
from src.models.confidence import ConfidenceEstimator, ConfidenceOutput
from src.models.dynamic_classifier import DynamicSignClassifier
from src.models.fusion import MultimodalFusionEngine
from src.models.router import LearnedStaticDynamicRouter
from src.models.static_gcn import StaticGraphGCN
from src.pipeline.boundary_detector import SignBoundaryDetector, SignSegment
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier
from src.utils.config import AppConfig, load_config

if TYPE_CHECKING:
    from src.nlp.types import BoundedSignPrediction
    from src.video.file_reader import DecodedFrame
    from src.video.signer_tracker import BoundingBox

logger = logging.getLogger(__name__)


class RecognitionPipeline:
    """End-to-end ISL recognition pipeline conforming to RecognitionPipelineAdapter protocol."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        models_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        mode: str = "multimodal",
        backend: str = "production",
        dynamic_checkpoint_path: Optional[Union[str, Path]] = None,
        class_mapping_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        use_boundary_trimming: bool = True,
    ) -> None:
        """Initializes configuration-aware end-to-end recognition pipeline.

        Args:
            config: Optional AppConfig instance. If None, loads system configuration.
            models_dir: Optional path to model checkpoints directory. If None, uses config default.
            device: Optional PyTorch device string or torch.device. Defaults to CPU.
            mode: Pipeline mode ("multimodal" or "dynamic_only"). Defaults to "multimodal".
            backend: Pipeline backend ("production" or "experimental_42"). Defaults to "production".
            dynamic_checkpoint_path: Optional explicit path to dynamic classifier checkpoint.
            class_mapping_path: Optional path to class mapping JSON file.
            calibration_path: Optional path to temperature calibration JSON file.
            use_boundary_trimming: Whether to enable dynamic BoundaryDetector trimming (default: True).
        """
        if mode not in ("multimodal", "dynamic_only"):
            raise ValueError(
                f"Invalid pipeline mode '{mode}'. Supported modes: 'multimodal', 'dynamic_only'."
            )
        if backend not in ("production", "experimental_42"):
            raise ValueError(
                f"Invalid backend '{backend}'. Supported backends: 'production', 'experimental_42'."
            )
        self.mode = mode
        self.backend = backend
        self.use_boundary_trimming = bool(use_boundary_trimming)

        if config is None:
            self.config = load_config()
        else:
            self.config = config

        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if models_dir is None:
            models_path_str = getattr(self.config.paths, "models_dir", "models")
            self.models_dir = Path(models_path_str)
        else:
            self.models_dir = Path(models_dir)

        if class_mapping_path is None:
            if self.backend == "experimental_42":
                self.class_mapping_path = Path("data/processed/include/vocabulary_initial.json")
            else:
                self.class_mapping_path = Path("data/processed/include/class_mapping.json")
        else:
            self.class_mapping_path = Path(class_mapping_path)

        if dynamic_checkpoint_path is None:
            if self.backend == "experimental_42":
                search_cands = [
                    self.models_dir / "checkpoints" / "experimental" / "controlled_stgcn_best.pt",
                    self.models_dir / "experimental" / "controlled_stgcn_best.pt",
                    self.models_dir / "controlled_stgcn_best.pt",
                ]
                if str(self.models_dir) == "models":
                    search_cands.extend([
                        Path("models/checkpoints/experimental/controlled_stgcn_best.pt"),
                        Path("models/experimental/controlled_stgcn_best.pt"),
                    ])
                self.dynamic_checkpoint_path = None
                for cand in search_cands:
                    if cand.is_file():
                        self.dynamic_checkpoint_path = cand
                        break
                if self.dynamic_checkpoint_path is None:
                    self.dynamic_checkpoint_path = self.models_dir / "checkpoints" / "experimental" / "controlled_stgcn_best.pt"
            else:
                self.dynamic_checkpoint_path = None
        else:
            self.dynamic_checkpoint_path = Path(dynamic_checkpoint_path)
            if self.backend == "experimental_42" and not self.dynamic_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Explicitly provided dynamic checkpoint file not found: '{self.dynamic_checkpoint_path}'"
                )

        if calibration_path is None:
            if self.backend == "experimental_42":
                calib_cands = [
                    self.models_dir / "checkpoints" / "experimental" / "calibration.json",
                    self.models_dir / "experimental" / "calibration.json",
                    self.models_dir / "calibration.json",
                ]
                if str(self.models_dir) == "models":
                    calib_cands.extend([
                        Path("models/checkpoints/experimental/calibration.json"),
                        Path("models/experimental/calibration.json"),
                    ])
                self.calibration_path = None
                for cand in calib_cands:
                    if cand.is_file():
                        self.calibration_path = cand
                        break
                if self.calibration_path is None:
                    self.calibration_path = self.models_dir / "checkpoints" / "experimental" / "calibration.json"
            else:
                self.calibration_path = None
        else:
            self.calibration_path = Path(calibration_path)

        self.temperature = 1.0
        if self.backend == "experimental_42" and self.calibration_path is not None and self.calibration_path.is_file():
            try:
                with open(self.calibration_path, "r", encoding="utf-8") as f:
                    calib_meta = json.load(f)
                    self.temperature = float(calib_meta.get("temperature", 1.0))
            except Exception as e:
                logger.warning(f"Failed to load temperature calibration from {self.calibration_path}: {e}")

        self._is_ready = False
        self._readiness_reason = "Checkpoints not checked"

        # Load canonical class mapping if available
        self.id_to_class: Optional[Dict[str, str]] = self._load_class_mapping()

        # 1. Feature Extraction & Preprocessing Components
        self.extractor = MediaPipeHolisticExtractor()
        self.normalizer = LandmarkNormalizer(
            target_fps=float(self.config.video.target_fps)
        )
        self.static_builder = StaticFeatureBuilder()
        self.sequence_buffer = SequenceBuffer(
            sequence_length=self.config.model.dynamic_model.sequence_length
        )

        # 2. PyTorch Model Components
        static_cfg = self.config.model.static_model
        self.static_gcn = StaticGraphGCN(
            input_dim=static_cfg.input_dim,
            num_nodes=static_cfg.num_nodes,
            spatial_channels=static_cfg.spatial_channels,
            hidden_dims=static_cfg.hidden_dims,
            num_classes=static_cfg.num_classes,
            dropout=static_cfg.dropout,
        ).to(self.device).eval()

        if self.backend == "experimental_42":
            self.dynamic_classifier = ControlledSTGCNClassifier(num_classes=42).to(self.device).eval()
        else:
            self.dynamic_classifier = DynamicSignClassifier.from_config(
                self.config.model.dynamic_model
            ).to(self.device).eval()

        self.router = LearnedStaticDynamicRouter.from_config(
            self.config.model.routing
        ).to(self.device).eval()

        self.fusion_engine = MultimodalFusionEngine.from_config(
            self.config.model.fusion
        ).to(self.device).eval()

        self.confidence_estimator = ConfidenceEstimator.from_config(
            self.config.model.inference
        ).to(self.device).eval()

        self.boundary_detector = SignBoundaryDetector(
            config=self.config.model.boundary_detection
        ).to(self.device).eval()

        # 3. Discover and load model checkpoints
        self._discover_and_load_checkpoints()

    def _load_class_mapping(self) -> Optional[Dict[str, str]]:
        """Loads canonical class mapping mapping string class IDs to glosses."""
        if not self.class_mapping_path.exists():
            if self.mode == "dynamic_only" or self.backend == "experimental_42":
                raise FileNotFoundError(
                    f"Class mapping file not found at '{self.class_mapping_path}'."
                )
            return None

        try:
            with open(self.class_mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError(
                    f"Class mapping JSON must be a dict, got {type(data)}"
                )

            if self.backend == "experimental_42":
                if "controlled_id_to_label" in data and isinstance(data["controlled_id_to_label"], dict):
                    return {str(k): str(v) for k, v in data["controlled_id_to_label"].items()}
                elif "id_to_class" in data and isinstance(data["id_to_class"], dict):
                    return {str(k): str(v) for k, v in data["id_to_class"].items()}
                else:
                    raise ValueError(
                        f"Vocabulary JSON missing 'controlled_id_to_label' dict at '{self.class_mapping_path}'"
                    )

            if "id_to_class" not in data or not isinstance(data["id_to_class"], dict):
                raise ValueError(
                    f"Class mapping JSON missing valid 'id_to_class' dict at '{self.class_mapping_path}'"
                )

            return {str(k): str(v) for k, v in data["id_to_class"].items()}
        except Exception as e:
            if self.mode == "dynamic_only" or self.backend == "experimental_42":
                logger.error(
                    f"Failed to load class mapping from {self.class_mapping_path}: {e}"
                )
                raise ValueError(
                    f"Failed to load class mapping from {self.class_mapping_path}: {e}"
                ) from e
            return None

    def _extract_state_dict(self, obj: Any) -> Dict[str, torch.Tensor]:
        """Extracts tensor state dict from a raw state dict or checkpoint dictionary."""
        if not isinstance(obj, dict):
            raise TypeError(f"Expected checkpoint or state_dict to be dict, got {type(obj)}")

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state_dict = obj["model_state_dict"]
        else:
            state_dict = obj

        if not state_dict or not all(isinstance(k, str) for k in state_dict.keys()):
            raise ValueError(
                "Malformed checkpoint: state_dict must be a non-empty dict with string keys."
            )

        return state_dict

    def _discover_and_load_checkpoints(self) -> None:
        """Scans models_dir or explicit paths for required model state dicts.

        Sets self._is_ready = True ONLY if valid checkpoints are found and loaded.
        Otherwise leaves self._is_ready = False without raising exceptions.
        """
        if self.backend == "experimental_42":
            resolved_dyn_path = self.dynamic_checkpoint_path
            if resolved_dyn_path is None or not resolved_dyn_path.is_file():
                self._is_ready = False
                self._readiness_reason = (
                    f"Missing required experimental 42-class checkpoint file at '{resolved_dyn_path}'."
                )
                return

            try:
                raw_dynamic = torch.load(resolved_dyn_path, map_location=self.device)
                dynamic_state = self._extract_state_dict(raw_dynamic)
                self.dynamic_classifier.load_state_dict(dynamic_state, strict=True)
                self.dynamic_classifier.eval()

                self._is_ready = True
                self._readiness_reason = (
                    f"Experimental 42-class checkpoint loaded successfully from '{resolved_dyn_path}'."
                )
            except Exception as e:
                self._is_ready = False
                self._readiness_reason = (
                    f"Failed to load experimental checkpoint from '{resolved_dyn_path}': {e}"
                )
                if self.mode == "dynamic_only":
                    raise ValueError(self._readiness_reason) from e
            return

        if self.mode == "dynamic_only":
            if self.id_to_class is None:
                self._is_ready = False
                self._readiness_reason = (
                    f"Class mapping file missing or malformed at '{self.class_mapping_path}'."
                )
                return

            resolved_dyn_path = None
            if self.dynamic_checkpoint_path is not None and self.dynamic_checkpoint_path.is_file():
                resolved_dyn_path = self.dynamic_checkpoint_path

            if resolved_dyn_path is None:
                search_candidates = [
                    self.models_dir / "checkpoints" / "dynamic_classifier" / "dynamic_classifier_best.pt",
                    self.models_dir / "checkpoints" / "dynamic_classifier" / "dynamic_classifier_last.pt",
                    self.models_dir / "dynamic_classifier.pt",
                    self.models_dir / "dynamic_classifier_best.pt",
                ]
                if str(self.models_dir) == "models":
                    search_candidates.extend([
                        Path("models/checkpoints/dynamic_classifier/dynamic_classifier_best.pt"),
                        Path("models/checkpoints/dynamic_classifier/dynamic_classifier_last.pt"),
                        Path("models/dynamic_classifier.pt"),
                    ])

                seen = set()
                for cand in search_candidates:
                    if cand in seen:
                        continue
                    seen.add(cand)
                    if cand.is_file():
                        resolved_dyn_path = cand
                        break

            if resolved_dyn_path is None:
                self._is_ready = False
                self._readiness_reason = (
                    f"Missing required dynamic classifier checkpoint file in {self.models_dir}."
                )
                return

            try:
                raw_dynamic = torch.load(resolved_dyn_path, map_location=self.device)
                dynamic_state = self._extract_state_dict(raw_dynamic)
                self.dynamic_classifier.load_state_dict(dynamic_state, strict=True)
                self.dynamic_classifier.eval()

                self._is_ready = True
                self._readiness_reason = (
                    f"Dynamic-only checkpoint loaded successfully from '{resolved_dyn_path}'."
                )
            except Exception as e:
                self._is_ready = False
                self._readiness_reason = (
                    f"Failed to load dynamic classifier checkpoint from '{resolved_dyn_path}': {e}"
                )
            return

        # Multimodal mode
        if not self.models_dir.exists() or not self.models_dir.is_dir():
            self._is_ready = False
            self._readiness_reason = (
                f"Models directory '{self.models_dir}' does not exist or is not a directory."
            )
            return

        checkpoint_files = {
            "static_gcn": self.models_dir / "static_gcn.pt",
            "dynamic_classifier": self.models_dir / "dynamic_classifier.pt",
            "router": self.models_dir / "router.pt",
            "fusion": self.models_dir / "fusion.pt",
        }

        missing = [name for name, path in checkpoint_files.items() if not path.is_file()]
        if missing:
            self._is_ready = False
            self._readiness_reason = (
                f"Missing required checkpoint files: {', '.join(missing)} in {self.models_dir}"
            )
            return

        try:
            raw_static = torch.load(checkpoint_files["static_gcn"], map_location=self.device)
            static_state = self._extract_state_dict(raw_static)
            self.static_gcn.load_state_dict(static_state, strict=True)

            raw_dynamic = torch.load(checkpoint_files["dynamic_classifier"], map_location=self.device)
            dynamic_state = self._extract_state_dict(raw_dynamic)
            self.dynamic_classifier.load_state_dict(dynamic_state, strict=True)

            raw_router = torch.load(checkpoint_files["router"], map_location=self.device)
            router_state = self._extract_state_dict(raw_router)
            self.router.load_state_dict(router_state, strict=True)

            raw_fusion = torch.load(checkpoint_files["fusion"], map_location=self.device)
            fusion_state = self._extract_state_dict(raw_fusion)
            self.fusion_engine.load_state_dict(fusion_state, strict=True)

            conf_ckpt = self.models_dir / "confidence.pt"
            if conf_ckpt.is_file():
                raw_conf = torch.load(conf_ckpt, map_location=self.device)
                conf_state = self._extract_state_dict(raw_conf)
                self.confidence_estimator.load_state_dict(conf_state, strict=True)

            self._is_ready = True
            self._readiness_reason = "All checkpoints loaded successfully."
        except Exception as e:
            self._is_ready = False
            self._readiness_reason = f"Failed to load checkpoint files: {e}"

    def is_available(self) -> bool:
        """Returns True if trained model weights and recognition pipeline are ready."""
        return self._is_ready

    def preprocess_frame(
        self, frame: Union[np.ndarray, DecodedFrame]
    ) -> Tuple[RawLandmarks, SelectedGraphLandmarks, StaticFeatureVector]:
        """Preprocesses a single video frame into landmarks and static feature vector."""
        if hasattr(frame, "frame") and hasattr(frame, "frame_index") and isinstance(frame.frame, np.ndarray):
            bgr_image = frame.frame
        elif isinstance(frame, np.ndarray):
            bgr_image = frame
        else:
            raise TypeError(
                f"Expected frame to be DecodedFrame or np.ndarray, got {type(frame)}"
            )

        if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
            raise ValueError("Provided frame array is None or empty.")

        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError(f"Expected BGR image shape [H, W, 3], got shape {bgr_image.shape}")

        raw_landmarks = self.extractor.extract(bgr_image)
        graph_landmarks = self.normalizer.normalize(raw_landmarks)
        static_vector = self.static_builder.build_from_graph(graph_landmarks)

        return raw_landmarks, graph_landmarks, static_vector

    def build_dynamic_tensor(
        self, graph_landmarks_list: List[SelectedGraphLandmarks]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a sequence of SelectedGraphLandmarks into a PyTorch dynamic sequence tensor."""
        seq_length = self.config.model.dynamic_model.sequence_length
        temp_buffer = SequenceBuffer(sequence_length=seq_length)

        for landmarks in graph_landmarks_list:
            temp_buffer.add_frame(landmarks)

        dyn_seq = temp_buffer.get_tensor(batch_size=1)

        tensor = torch.from_numpy(dyn_seq.tensor).to(
            device=self.device, dtype=torch.float32
        )
        mask = torch.from_numpy(dyn_seq.mask).to(
            device=self.device, dtype=torch.bool
        )

        return tensor, mask

    def get_static_label(self, class_idx: int) -> str:
        """Returns internal identifier for static class index."""
        return f"static_{class_idx}"

    def get_dynamic_label(self, class_idx: int) -> str:
        """Returns canonical gloss for dynamic class index using class_mapping.json."""
        if self.mode == "dynamic_only":
            if self.id_to_class is not None:
                key = str(class_idx)
                if key in self.id_to_class:
                    return self.id_to_class[key]
                raise KeyError(
                    f"Class index {class_idx} not found in class mapping JSON at {self.class_mapping_path}."
                )
            raise FileNotFoundError(
                f"Class mapping file unavailable or invalid at {self.class_mapping_path}. "
                f"Cannot resolve gloss for class index {class_idx}."
            )

        return f"dynamic_{class_idx}"

    def process_frames(
        self,
        frames: List[DecodedFrame],
        primary_tracks: List[Optional[BoundingBox]],
    ) -> List[BoundedSignPrediction]:
        """Processes decoded frame chunks and yields validated single-sample BoundedSignPredictions."""
        if not self.is_available():
            return []

        if not frames:
            return []

        self.normalizer.reset()
        self.sequence_buffer.reset()

        from src.nlp.types import BoundedSignPrediction

        valid_frames: List[DecodedFrame] = []
        graph_landmarks_list: List[SelectedGraphLandmarks] = []
        static_vectors_list: List[StaticFeatureVector] = []

        # 1. Preprocess frames into landmarks and static feature vectors
        for df in frames:
            try:
                _, graph_lm, stat_vec = self.preprocess_frame(df)
                graph_landmarks_list.append(graph_lm)
                static_vectors_list.append(stat_vec)
                valid_frames.append(df)
            except Exception as e:
                logger.warning(f"Skipping frame during preprocessing: {e}")
                continue

        if not graph_landmarks_list:
            return []

        # 2. Build full sequence dynamic tensor [1, 5, T, 115] and mask [1, T, 115]
        dyn_tensor, dyn_mask = self.build_dynamic_tensor(graph_landmarks_list)

        # 3. Detect continuous sign boundaries or bypass trimming
        if not self.use_boundary_trimming:
            start_f = 0
            end_f = len(graph_landmarks_list) - 1
            full_duration = max(1, end_f - start_f + 1)
            sample_segments = [
                SignSegment(
                    start_frame=start_f,
                    end_frame=end_f,
                    duration=full_duration,
                    mean_energy=1.0,
                    boundary_evidence=1.0,
                    is_valid=True,
                )
            ]
        else:
            with torch.no_grad():
                boundary_out = self.boundary_detector(dyn_tensor, valid_mask=dyn_mask)

            sample_segments = boundary_out.segments[0] if boundary_out.segments else []

        if not sample_segments:
            return []

        predictions: List[BoundedSignPrediction] = []

        # 4. Process each detected segment
        with torch.no_grad():
            for segment in sample_segments:
                start_f = max(0, segment.start_frame)
                end_f = min(len(graph_landmarks_list) - 1, segment.end_frame)

                if start_f > end_f:
                    continue

                if not self.use_boundary_trimming:
                    seg_tensor, seg_mask = dyn_tensor, dyn_mask
                else:
                    seg_graph_lms = graph_landmarks_list[start_f : end_f + 1]
                    seg_tensor, seg_mask = self.build_dynamic_tensor(seg_graph_lms)

                # Enforce input tensor shape contract [B, 5, T, 115]
                if seg_tensor.dim() != 4 or seg_tensor.shape[1] != 5 or seg_tensor.shape[3] != 115:
                    raise ValueError(
                        f"Dynamic tensor shape {list(seg_tensor.shape)} is incompatible with expected contract [B, 5, T, 115]."
                    )

                dynamic_out = self.dynamic_classifier(seg_tensor, mask=seg_mask)
                if hasattr(dynamic_out, "logits"):
                    dynamic_logits = dynamic_out.logits
                    dynamic_emb = getattr(dynamic_out, "embedding", None)
                else:
                    dynamic_logits = dynamic_out
                    dynamic_emb = None

                if self.temperature != 1.0:
                    dynamic_logits = dynamic_logits / self.temperature

                if self.mode == "dynamic_only" or self.backend == "experimental_42":
                    selected_logits = dynamic_logits
                else:
                    mid_idx = start_f + (end_f - start_f) // 2
                    stat_feat = static_vectors_list[mid_idx].features  # (225,)
                    stat_tensor = (
                        torch.from_numpy(stat_feat)
                        .unsqueeze(0)
                        .to(device=self.device, dtype=torch.float32)
                    )

                    static_out = self.static_gcn(stat_tensor)
                    static_logits = static_out.logits  # [1, 36]
                    static_emb = static_out.embedding  # [1, 256]

                    router_out = self.router(static_emb, dynamic_emb)
                    static_w = float(router_out.static_weight[0, 0].item())
                    dynamic_w = float(router_out.dynamic_weight[0, 0].item())

                    _ = self.fusion_engine(static_emb, dynamic_emb)

                    if static_w > dynamic_w:
                        selected_logits = static_logits
                    else:
                        selected_logits = dynamic_logits

                # Confidence estimation and rejection thresholding
                conf_out = self.confidence_estimator(selected_logits)

                # Resolve glosses for predicted class and top-K candidates
                pred_cls = (
                    int(conf_out.predicted_class[0].item())
                    if hasattr(conf_out.predicted_class, "item")
                    else int(conf_out.predicted_class)
                )
                conf_out.predicted_gloss = self.get_dynamic_label(pred_cls)

                top_k_ids = (
                    conf_out.top_k_indices[0].tolist()
                    if hasattr(conf_out.top_k_indices, "tolist")
                    else list(conf_out.top_k_indices)
                )
                conf_out.top_k_glosses = [self.get_dynamic_label(int(idx)) for idx in top_k_ids]

                # Map relative segment indices to absolute video frame indices
                abs_start_frame = (
                    valid_frames[start_f].frame_index
                    if hasattr(valid_frames[start_f], "frame_index")
                    else start_f
                )
                abs_end_frame = (
                    valid_frames[end_f].frame_index
                    if hasattr(valid_frames[end_f], "frame_index")
                    else end_f
                )
                abs_duration = max(1, abs_end_frame - abs_start_frame + 1)
                abs_segment = SignSegment(
                    start_frame=abs_start_frame,
                    end_frame=abs_end_frame,
                    duration=abs_duration,
                    mean_energy=segment.mean_energy,
                    boundary_evidence=segment.boundary_evidence,
                    is_valid=segment.is_valid,
                )

                pred = BoundedSignPrediction(segment=abs_segment, prediction=conf_out)
                pred.validate()
                predictions.append(pred)

        return predictions

    def close(self) -> None:
        """Releases underlying resources (e.g. MediaPipe holistic extractor)."""
        if hasattr(self, "extractor") and self.extractor is not None:
            self.extractor.close()

    def __enter__(self) -> "RecognitionPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
