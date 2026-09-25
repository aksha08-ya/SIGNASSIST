"""
Structured logging system for SIGNASSIST.

Provides configurable console and file logging with rotating file handlers,
ensuring log directory creation and standardized log message formatting.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.config import ConfigManager

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "signassist",
    log_level: Optional[str] = None,
    log_file: Optional[Path | str] = None,
    console: Optional[bool] = None,
    file_output: Optional[bool] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Sets up and configures a structured logger with console and rotating file handlers."""
    logger = logging.getLogger(name)

    # Return existing configured logger if handlers already exist to avoid duplication
    if logger.handlers:
        return logger

    # Try loading system config defaults if parameters not explicitly provided
    try:
        cfg = ConfigManager().load_all().logging
        paths_cfg = ConfigManager().load_all().paths
        default_level = log_level or cfg.level
        default_console = console if console is not None else cfg.console_output
        default_file_output = file_output if file_output is not None else cfg.file_output
        default_log_file = log_file or Path(paths_cfg.logs_dir) / cfg.log_filename
    except Exception:
        # Fallback to sensible defaults if config fails to load
        default_level = log_level or "INFO"
        default_console = console if console is not None else True
        default_file_output = file_output if file_output is not None else True
        default_log_file = log_file or Path("reports/logs/signassist.log")

    level_num = getattr(logging, str(default_level).upper(), logging.INFO)
    logger.setLevel(level_num)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console Handler
    if default_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level_num)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if default_file_output and default_log_file:
        file_path = Path(default_log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level_num)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log propagation to root logger if custom handlers are attached
    logger.propagate = False

    return logger
