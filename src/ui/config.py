"""UI configuration helper for SIGNASSIST."""

from pathlib import Path
from typing import Optional, Union

from src.utils.config import AppConfig, ConfigManager, load_config


def get_ui_config(config_dir: Optional[Union[str, Path]] = None) -> AppConfig:
    """Loads and returns the application AppConfig containing UI and system configuration.

    Args:
        config_dir: Optional directory path override for configuration YAML files.

    Returns:
        AppConfig instance containing typed configuration sections.
    """
    if config_dir is not None:
        return ConfigManager(config_dir=config_dir).load_all()
    return load_config()
