"""
Unit tests for SIGNASSIST configuration manager.
"""

import os
from pathlib import Path
import pytest
import yaml

from src.utils.config import ConfigManager, load_config, resolve_device
from src.utils.exceptions import ConfigurationError


def test_load_default_configs(tmp_path):
    default_yaml = tmp_path / "default.yaml"
    model_yaml = tmp_path / "model.yaml"
    ui_yaml = tmp_path / "ui.yaml"

    default_yaml.write_text(
        yaml.dump(
            {
                "system": {"name": "TestApp", "environment": "test", "device": "cpu"},
                "video": {"target_fps": 30},
                "logging": {"level": "DEBUG"},
            }
        )
    )
    model_yaml.write_text(
        yaml.dump(
            {
                "static_model": {"input_dim": 225, "num_nodes": 75},
                "dynamic_model": {"raw_landmarks": 543, "num_nodes": 115, "landmark_channels": 5},
                "inference": {"confidence_threshold": 0.8},
            }
        )
    )
    ui_yaml.write_text(yaml.dump({"app": {"title": "Test Title"}}))

    manager = ConfigManager(config_dir=tmp_path)
    config = manager.load_all()

    assert config.system.name == "TestApp"
    assert config.system.environment == "test"
    assert config.system.device == "cpu"
    assert config.video.target_fps == 30
    assert config.logging.level == "DEBUG"
    assert config.model.static_model.input_dim == 225
    assert config.model.dynamic_model.raw_landmarks == 543
    assert config.model.dynamic_model.num_nodes == 115
    assert config.model.dynamic_model.landmark_channels == 5
    assert config.model.inference.confidence_threshold == 0.8
    assert config.ui.app.title == "Test Title"


def test_device_resolution():
    assert resolve_device("cpu") == "cpu"
    # When CUDA is unavailable in test environment, 'cuda' and 'auto' safely resolve to 'cpu'
    assert resolve_device("cuda") in ("cuda", "cpu")
    assert resolve_device("auto") in ("cuda", "cpu")

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_device("invalid_device")
    assert "Unsupported device" in str(exc_info.value)


def test_env_var_overrides(tmp_path, monkeypatch):
    default_yaml = tmp_path / "default.yaml"
    default_yaml.write_text(
        yaml.dump(
            {
                "system": {"environment": "development", "device": "auto"},
                "logging": {"level": "INFO"},
            }
        )
    )

    monkeypatch.setenv("SIGNASSIST_ENV", "production")
    monkeypatch.setenv("SIGNASSIST_DEVICE", "cpu")
    monkeypatch.setenv("SIGNASSIST_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SIGNASSIST_CONFIDENCE_THRESHOLD", "0.85")

    manager = ConfigManager(config_dir=tmp_path)
    config = manager.load_all()

    assert config.system.environment == "production"
    assert config.system.requested_device == "cpu"
    assert config.system.device == "cpu"
    assert config.logging.level == "WARNING"
    assert config.model.inference.confidence_threshold == 0.85


def test_validation_errors(tmp_path):
    default_yaml = tmp_path / "default.yaml"
    default_yaml.write_text(yaml.dump({"video": {"target_fps": -5}}))

    manager = ConfigManager(config_dir=tmp_path)
    with pytest.raises(ConfigurationError) as exc_info:
        manager.load_all()
    assert "Invalid target_fps" in str(exc_info.value)


def test_invalid_yaml_file(tmp_path):
    default_yaml = tmp_path / "default.yaml"
    default_yaml.write_text("system: [unclosed_bracket")

    manager = ConfigManager(config_dir=tmp_path)
    with pytest.raises(ConfigurationError) as exc_info:
        manager.load_all()
    assert "Failed to parse YAML file" in str(exc_info.value)
