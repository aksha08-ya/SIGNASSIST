"""
End-to-End Recognition Pipeline for SIGNASSIST (Phase 17 Block 6).

Integrates feature extraction, landmark normalization, static GCN classification,
dynamic ST-GCN + Transformer classification, learned routing, multimodal fusion,
confidence estimation with rejection thresholding, and continuous boundary detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import logging
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
    ) -> None:
        """Initializes configuration-aware end-to-end recognition pipeline.

        Args:
            config: Optional AppConfig instance. If None, loads system configuration.
            models_dir: Optional path to model checkpoints directory. If None, uses config default.
            device: Optional PyTorch device string or torch.device. Defaults to CPU.
        """
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

        self._is_ready = False
        self._readiness_reason = "Checkpoints not checked"

        # 1. Feature Extraction & Preprocessing Components
        self.extractor = MediaPipeHolisticExtractor()
        self.normalizer = LandmarkNormalizer(
            target_fps=float(self.config.video.target_fps)
        )
        self.static_builder = StaticFeatureBuilder()
        # Reserved for future streaming/webcam integration; process_frames uses temporary SequenceBuffer instances
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

    def _extract_state_dict(self, obj: Any) -> Dict[str, torch.Tensor]:
        """Extracts tensor state dict from a raw state dict or checkpoint dictionary.

        Accepts:
        a) a raw state_dict mapping parameter names to tensors, or
        b) a checkpoint dict containing 'state_dict' or 'model_state_dict'.

        Raises ValueError/TypeError on malformed checkpoint data.
        """
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
        """Scans models_dir for required model state dicts.

        Sets self._is_ready = True ONLY if valid checkpoints are found and loaded.
        Otherwise leaves self._is_ready = False without raising exceptions.
        """
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
        """Preprocesses a single video frame into landmarks and static feature vector.

        Args:
            frame: Either a DecodedFrame or a BGR numpy image array [H, W, 3].

        Returns:
            Tuple of (RawLandmarks, SelectedGraphLandmarks (115, 5), StaticFeatureVector (225,)).
        """
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
        """Converts a sequence of SelectedGraphLandmarks into a PyTorch dynamic sequence tensor.

        Note: Uses temporary SequenceBuffer instances for stateless batch/segment tensor construction.
        self.sequence_buffer is reserved for future streaming/webcam integration.

        Args:
            graph_landmarks_list: Sequence of (115, 5) node landmarks.

        Returns:
            Tuple of:
            - tensor: torch.Tensor of shape [1, 5, T, 115] float32 on self.device
            - mask: torch.Tensor of shape [1, T, 115] bool on self.device
        """
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
        """Returns internal identifier for dynamic class index."""
        return f"dynamic_{class_idx}"

    def process_frames(
        self,
        frames: List[DecodedFrame],
        primary_tracks: List[Optional[BoundingBox]],
    ) -> List[BoundedSignPrediction]:
        """Processes decoded frame chunks and yields validated single-sample BoundedSignPredictions.

        Note: primary_tracks is accepted to satisfy the RecognitionPipelineAdapter contract,
        but is currently not consumed by the landmark extractor. Signer-track-aware
        cropping/ROI integration is deferred to the signer-tracking integration layer.

        If pipeline is not available (checkpoints missing/not loaded), returns [] immediately.
        """
        if not self.is_available():
            return []

        if not frames:
            return []

        # Reset normalizer and sequence buffer history to prevent state leakage between independent calls
        self.normalizer.reset()
        self.sequence_buffer.reset()

        from src.nlp.types import BoundedSignPrediction

        graph_landmarks_list: List[SelectedGraphLandmarks] = []
        static_vectors_list: List[StaticFeatureVector] = []

        # 1. Preprocess frames into landmarks and static feature vectors
        for df in frames:
            try:
                _, graph_lm, stat_vec = self.preprocess_frame(df)
                graph_landmarks_list.append(graph_lm)
                static_vectors_list.append(stat_vec)
            except Exception as e:
                logger.warning(f"Skipping frame during preprocessing: {e}")
                continue

        if not graph_landmarks_list:
            return []

        # 2. Build full sequence dynamic tensor [1, 5, T, 115] and mask [1, T, 115]
        dyn_tensor, dyn_mask = self.build_dynamic_tensor(graph_landmarks_list)

        # 3. Detect continuous sign boundaries
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
                end_f = min(len(static_vectors_list) - 1, segment.end_frame)

                if start_f > end_f:
                    continue

                # Representative static feature vector at temporal midpoint of segment
                mid_idx = start_f + (end_f - start_f) // 2
                stat_feat = static_vectors_list[mid_idx].features  # (225,)
                stat_tensor = (
                    torch.from_numpy(stat_feat)
                    .unsqueeze(0)
                    .to(device=self.device, dtype=torch.float32)
                )

                # Static GCN forward pass
                static_out = self.static_gcn(stat_tensor)
                static_logits = static_out.logits  # [1, 36]
                static_emb = static_out.embedding  # [1, 256]

                # Dynamic classification on segment sequence window
                seg_graph_lms = graph_landmarks_list[start_f : end_f + 1]
                seg_tensor, seg_mask = self.build_dynamic_tensor(seg_graph_lms)
                dynamic_out = self.dynamic_classifier(seg_tensor, mask=seg_mask)
                dynamic_logits = dynamic_out.logits  # [1, 250]
                dynamic_emb = dynamic_out.embedding  # [1, 256]

                # Learned router gating pass
                router_out = self.router(static_emb, dynamic_emb)
                static_w = float(router_out.static_weight[0, 0].item())
                dynamic_w = float(router_out.dynamic_weight[0, 0].item())

                # Multimodal fusion pass (embedding-level integration/diagnostic component).
                # Note: Executed for embedding diagnostic pass; fused embedding is not yet used for
                # final classification as there is currently no trained fused classification head/checkpoint.
                _ = self.fusion_engine(static_emb, dynamic_emb)

                # Branch selection based on router probabilities
                if static_w > dynamic_w:
                    selected_logits = static_logits
                else:
                    selected_logits = dynamic_logits

                # Confidence estimation and rejection thresholding
                conf_out = self.confidence_estimator(selected_logits)

                # Construct bounded prediction binding segment to single-sample confidence output
                pred = BoundedSignPrediction(segment=segment, prediction=conf_out)
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
