"""
Typed, validated configuration management system for SIGNASSIST.

Loads system, model, and UI configurations from YAML files and environment
variables, validating schemas and supporting flexible runtime access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError


def resolve_device(requested_device: str) -> str:
    """Resolves and validates target hardware device with safe CPU fallback.

    Supports 'auto', 'cpu', 'cuda', and 'mps'. If 'cuda' or 'auto' is specified,
    validates PyTorch CUDA availability and safely falls back to 'cpu' if CUDA is unavailable.
    """
    req = str(requested_device).strip().lower()
    if req == "cpu":
        return "cpu"

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if req == "cuda":
        if cuda_available:
            return "cuda"
        return "cpu"
    elif req == "auto":
        return "cuda" if cuda_available else "cpu"
    elif req == "mps":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    else:
        raise ConfigurationError(
            f"Unsupported device '{requested_device}'. Must be 'auto', 'cpu', 'cuda', or 'mps'."
        )


@dataclass
class SystemConfig:
    name: str = "SIGNASSIST"
    version: str = "0.1.0"
    environment: str = "development"
    seed: int = 42
    requested_device: str = "auto"
    device: str = "cpu"  # Resolved hardware target after CUDA availability check


@dataclass
class PathsConfig:
    base_dir: str = "."
    data_dir: str = "data"
    models_dir: str = "models"
    reports_dir: str = "reports"
    logs_dir: str = "reports/logs"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    log_filename: str = "signassist.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class VideoConfig:
    target_fps: int = 30
    target_width: int = 640
    target_height: int = 480
    supported_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    )
    max_video_length_sec: int = 120


@dataclass
class RoutingConfig:
    mechanism: str = "learned"
    dynamic_threshold: float = 0.5
    enable_fast_path: bool = True
    static_dim: int = 256
    dynamic_dim: int = 256
    hidden_dim: int = 128
    dropout: float = 0.2


@dataclass
class FusionConfig:
    mechanism: str = "weighted_sum"
    input_dim: int = 256
    output_dim: int = 256
    dropout: float = 0.2
    use_refinement: bool = True


@dataclass
class GraphConfig:
    num_nodes: int = 115
    self_loops: bool = True
    normalization: str = "symmetric"
    cross_part_edges: bool = True


@dataclass
class StaticModelConfig:
    name: str = "static_sign_classifier"
    type: str = "gnn"
    input_dim: int = 225  # 75 nodes (21 LH + 21 RH + 33 Pose) x 3 spatial coords (x,y,z)
    num_nodes: int = 75
    spatial_channels: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2
    num_classes: int = 36


@dataclass
class DynamicModelConfig:
    name: str = "dynamic_sign_classifier"
    type: str = "st_gcn_transformer"
    sequence_length: int = 60
    raw_landmarks: int = 543  # Raw MediaPipe Holistic keypoints (468 face + 33 pose + 21 LH + 21 RH)
    num_nodes: int = 115  # Selected recognition graph nodes (21 LH + 21 RH + 33 pose + 40 face contour)
    landmark_channels: int = 5  # Feature channels per node (x, y, z, vx, vy)
    input_landmarks: int = 543  # Alias for backward compatibility
    hidden_dim: int = 256
    num_stgcn_layers: int = 4
    num_transformer_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.3
    num_classes: int = 250


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.70
    rejection_threshold: float = 0.40
    temporal_window_size: int = 5
    duplicate_suppression_window: int = 10
    batch_size: int = 16


@dataclass
class ModelConfig:
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    static_model: StaticModelConfig = field(default_factory=StaticModelConfig)
    dynamic_model: DynamicModelConfig = field(default_factory=DynamicModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass
class UIAppConfig:
    title: str = "SIGNASSIST - Real-Time Indian Sign Language Recognition & Translation"
    theme: str = "dark"
    page_icon: str = "🤟"
    layout: str = "wide"


@dataclass
class UIWebcamConfig:
    fps: int = 30
    width: int = 640
    height: int = 480
    mirror: bool = True
    auto_start: bool = False


@dataclass
class UIDisplayConfig:
    show_landmarks: bool = True
    show_confidence_bars: bool = True
    show_bounding_box: bool = True
    max_history_lines: int = 20


@dataclass
class TTSConfig:
    enabled: bool = True
    auto_play: bool = False
    voice_gender: str = "female"
    speech_rate: float = 1.0


@dataclass
class UITranslationConfig:
    nlp_smoothing: bool = True
    min_confidence: float = 0.70


@dataclass
class UIConfig:
    app: UIAppConfig = field(default_factory=UIAppConfig)
    webcam: UIWebcamConfig = field(default_factory=UIWebcamConfig)
    display: UIDisplayConfig = field(default_factory=UIDisplayConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    translation: UITranslationConfig = field(default_factory=UITranslationConfig)


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)


class ConfigManager:
    """Manages loading, merging, environment variable overrides, and validation of configurations."""

    def __init__(
        self,
        config_dir: Optional[Path | str] = None,
        env_file: Optional[Path | str] = None,
    ) -> None:
        if config_dir is None:
            self.config_dir = Path("configs")
        else:
            self.config_dir = Path(config_dir)

        # Load environment variables from .env if present
        if env_file:
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()

        self._raw_default_cfg: Dict[str, Any] = {}
        self._raw_model_cfg: Dict[str, Any] = {}
        self._raw_ui_cfg: Dict[str, Any] = {}
        self._config: Optional[AppConfig] = None

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Loads a YAML file safely."""
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                details={"file_path": str(file_path)},
            )
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML file {file_path}: {e}",
                details={"file_path": str(file_path), "error": str(e)},
            )

    def load_all(self) -> AppConfig:
        """Loads default, model, and UI configurations, applies env overrides and returns AppConfig."""
        default_yaml_path = self.config_dir / "default.yaml"
        model_yaml_path = self.config_dir / "model.yaml"
        ui_yaml_path = self.config_dir / "ui.yaml"

        if default_yaml_path.exists():
            self._raw_default_cfg = self._load_yaml(default_yaml_path)
        if model_yaml_path.exists():
            self._raw_model_cfg = self._load_yaml(model_yaml_path)
        if ui_yaml_path.exists():
            self._raw_ui_cfg = self._load_yaml(ui_yaml_path)

        system_raw = self._raw_default_cfg.get("system", {})
        paths_raw = self._raw_default_cfg.get("paths", {})
        logging_raw = self._raw_default_cfg.get("logging", {})
        video_raw = self._raw_default_cfg.get("video", {})

        # Environment variable overrides
        system_env = os.getenv("SIGNASSIST_ENV", system_raw.get("environment", "development"))
        device_req = os.getenv("SIGNASSIST_DEVICE", system_raw.get("device", "auto"))
        resolved_device = resolve_device(device_req)

        log_level_env = os.getenv("SIGNASSIST_LOG_LEVEL", logging_raw.get("level", "INFO"))
        data_dir_env = os.getenv("SIGNASSIST_DATA_DIR", paths_raw.get("data_dir", "data"))
        models_dir_env = os.getenv("SIGNASSIST_MODELS_DIR", paths_raw.get("models_dir", "models"))
        reports_dir_env = os.getenv("SIGNASSIST_REPORTS_DIR", paths_raw.get("reports_dir", "reports"))

        system_cfg = SystemConfig(
            name=system_raw.get("name", "SIGNASSIST"),
            version=system_raw.get("version", "0.1.0"),
            environment=system_env,
            seed=int(system_raw.get("seed", 42)),
            requested_device=device_req,
            device=resolved_device,
        )

        paths_cfg = PathsConfig(
            base_dir=paths_raw.get("base_dir", "."),
            data_dir=data_dir_env,
            models_dir=models_dir_env,
            reports_dir=reports_dir_env,
            logs_dir=paths_raw.get("logs_dir", "reports/logs"),
        )

        logging_cfg = LoggingConfig(
            level=log_level_env,
            console_output=bool(logging_raw.get("console_output", True)),
            file_output=bool(logging_raw.get("file_output", True)),
            log_filename=logging_raw.get("log_filename", "signassist.log"),
            max_file_size_mb=int(logging_raw.get("max_file_size_mb", 10)),
            backup_count=int(logging_raw.get("backup_count", 5)),
        )

        video_cfg = VideoConfig(
            target_fps=int(video_raw.get("target_fps", 30)),
            target_width=int(video_raw.get("target_width", 640)),
            target_height=int(video_raw.get("target_height", 480)),
            supported_extensions=video_raw.get(
                "supported_extensions", [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            ),
            max_video_length_sec=int(video_raw.get("max_video_length_sec", 120)),
        )

        # Model configs
        routing_raw = self._raw_model_cfg.get("routing", {})
        fusion_raw = self._raw_model_cfg.get("fusion", {})
        graph_raw = self._raw_model_cfg.get("graph", {})
        static_raw = self._raw_model_cfg.get("static_model", {})
        dynamic_raw = self._raw_model_cfg.get("dynamic_model", {})
        inference_raw = self._raw_model_cfg.get("inference", {})

        conf_env = os.getenv("SIGNASSIST_CONFIDENCE_THRESHOLD")
        conf_threshold = (
            float(conf_env) if conf_env is not None else float(inference_raw.get("confidence_threshold", 0.70))
        )

        model_cfg = ModelConfig(
            routing=RoutingConfig(
                mechanism=routing_raw.get("mechanism", "learned"),
                dynamic_threshold=float(routing_raw.get("dynamic_threshold", 0.5)),
                enable_fast_path=bool(routing_raw.get("enable_fast_path", True)),
                static_dim=int(routing_raw.get("static_dim", 256)),
                dynamic_dim=int(routing_raw.get("dynamic_dim", 256)),
                hidden_dim=int(routing_raw.get("hidden_dim", 128)),
                dropout=float(routing_raw.get("dropout", 0.2)),
            ),
            fusion=FusionConfig(
                mechanism=fusion_raw.get("mechanism", "weighted_sum"),
                input_dim=int(fusion_raw.get("input_dim", 256)),
                output_dim=int(fusion_raw.get("output_dim", 256)),
                dropout=float(fusion_raw.get("dropout", 0.2)),
                use_refinement=bool(fusion_raw.get("use_refinement", True)),
            ),
            graph=GraphConfig(
                num_nodes=int(graph_raw.get("num_nodes", 115)),
                self_loops=bool(graph_raw.get("self_loops", True)),
                normalization=graph_raw.get("normalization", "symmetric"),
                cross_part_edges=bool(graph_raw.get("cross_part_edges", True)),
            ),
            static_model=StaticModelConfig(
                name=static_raw.get("name", "static_sign_classifier"),
                type=static_raw.get("type", "gnn"),
                input_dim=int(static_raw.get("input_dim", 225)),
                num_nodes=int(static_raw.get("num_nodes", 75)),
                spatial_channels=int(static_raw.get("spatial_channels", 3)),
                hidden_dims=static_raw.get("hidden_dims", [512, 256, 128]),
                dropout=float(static_raw.get("dropout", 0.2)),
                num_classes=int(static_raw.get("num_classes", 36)),
            ),
            dynamic_model=DynamicModelConfig(
                name=dynamic_raw.get("name", "dynamic_sign_classifier"),
                type=dynamic_raw.get("type", "st_gcn_transformer"),
                sequence_length=int(dynamic_raw.get("sequence_length", 60)),
                raw_landmarks=int(
                    dynamic_raw.get("raw_landmarks", dynamic_raw.get("input_landmarks", 543))
                ),
                num_nodes=int(dynamic_raw.get("num_nodes", 115)),
                landmark_channels=int(dynamic_raw.get("landmark_channels", 5)),
                input_landmarks=int(dynamic_raw.get("input_landmarks", 543)),
                hidden_dim=int(dynamic_raw.get("hidden_dim", 256)),
                num_stgcn_layers=int(dynamic_raw.get("num_stgcn_layers", 4)),
                num_transformer_layers=int(dynamic_raw.get("num_transformer_layers", 4)),
                num_heads=int(dynamic_raw.get("num_heads", 8)),
                dropout=float(dynamic_raw.get("dropout", 0.3)),
                num_classes=int(dynamic_raw.get("num_classes", 250)),
            ),
            inference=InferenceConfig(
                confidence_threshold=conf_threshold,
                rejection_threshold=float(inference_raw.get("rejection_threshold", 0.40)),
                temporal_window_size=int(inference_raw.get("temporal_window_size", 5)),
                duplicate_suppression_window=int(
                    inference_raw.get("duplicate_suppression_window", 10)
                ),
                batch_size=int(inference_raw.get("batch_size", 16)),
            ),
        )

        # UI configs
        app_ui_raw = self._raw_ui_cfg.get("app", {})
        webcam_ui_raw = self._raw_ui_cfg.get("webcam", {})
        display_ui_raw = self._raw_ui_cfg.get("display", {})
        tts_ui_raw = self._raw_ui_cfg.get("tts", {})
        translation_ui_raw = self._raw_ui_cfg.get("translation", {})

        ui_cfg = UIConfig(
            app=UIAppConfig(
                title=app_ui_raw.get(
                    "title",
                    "SIGNASSIST - Real-Time Indian Sign Language Recognition & Translation",
                ),
                theme=app_ui_raw.get("theme", "dark"),
                page_icon=app_ui_raw.get("page_icon", "🤟"),
                layout=app_ui_raw.get("layout", "wide"),
            ),
            webcam=UIWebcamConfig(
                fps=int(webcam_ui_raw.get("fps", 30)),
                width=int(webcam_ui_raw.get("width", 640)),
                height=int(webcam_ui_raw.get("height", 480)),
                mirror=bool(webcam_ui_raw.get("mirror", True)),
                auto_start=bool(webcam_ui_raw.get("auto_start", False)),
            ),
            display=UIDisplayConfig(
                show_landmarks=bool(display_ui_raw.get("show_landmarks", True)),
                show_confidence_bars=bool(
                    display_ui_raw.get("show_confidence_bars", True)
                ),
                show_bounding_box=bool(display_ui_raw.get("show_bounding_box", True)),
                max_history_lines=int(display_ui_raw.get("max_history_lines", 20)),
            ),
            tts=TTSConfig(
                enabled=bool(tts_ui_raw.get("enabled", True)),
                auto_play=bool(tts_ui_raw.get("auto_play", False)),
                voice_gender=tts_ui_raw.get("voice_gender", "female"),
                speech_rate=float(tts_ui_raw.get("speech_rate", 1.0)),
            ),
            translation=UITranslationConfig(
                nlp_smoothing=bool(translation_ui_raw.get("nlp_smoothing", True)),
                min_confidence=float(translation_ui_raw.get("min_confidence", 0.70)),
            ),
        )

        app_config = AppConfig(
            system=system_cfg,
            paths=paths_cfg,
            logging=logging_cfg,
            video=video_cfg,
            model=model_cfg,
            ui=ui_cfg,
        )

        self._validate(app_config)
        self._config = app_config
        return app_config

    def _validate(self, config: AppConfig) -> None:
        """Validates loaded parameters for bounds and consistency."""
        if config.video.target_fps <= 0:
            raise ConfigurationError(
                f"Invalid target_fps: {config.video.target_fps}. Must be > 0."
            )

        if not (0.0 <= config.model.inference.confidence_threshold <= 1.0):
            raise ConfigurationError(
                f"Invalid confidence_threshold: {config.model.inference.confidence_threshold}. Must be between 0.0 and 1.0."
            )

        if not (0.0 <= config.model.inference.rejection_threshold <= 1.0):
            raise ConfigurationError(
                f"Invalid rejection_threshold: {config.model.inference.rejection_threshold}. Must be between 0.0 and 1.0."
            )

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.logging.level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid logging level: '{config.logging.level}'. Must be one of {valid_levels}."
            )

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            return self.load_all()
        return self._config


def load_config(config_dir: Optional[Path | str] = None) -> AppConfig:
    """Convenience helper to quickly instantiate ConfigManager and load configurations."""
    manager = ConfigManager(config_dir=config_dir)
    return manager.load_all()
