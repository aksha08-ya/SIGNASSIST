"""
Unit tests for SIGNASSIST logging utility.
"""

import logging
from pathlib import Path
import pytest

from src.utils.logger import setup_logger


def test_setup_logger_console_and_file(tmp_path):
    log_file = tmp_path / "test_logs" / "test.log"
    logger_name = "test_logger_unique_1"

    logger = setup_logger(
        name=logger_name,
        log_level="DEBUG",
        log_file=log_file,
        console=True,
        file_output=True,
    )

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # StreamHandler + RotatingFileHandler

    # Log a test message
    logger.debug("Debug test message")
    logger.info("Info test message")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Debug test message" in content
    assert "Info test message" in content


def test_setup_logger_existing_handlers():
    logger_name = "test_logger_unique_2"
    logger1 = setup_logger(name=logger_name)
    num_handlers = len(logger1.handlers)

    # Calling setup_logger again on same logger name should return existing without duplicate handlers
    logger2 = setup_logger(name=logger_name)
    assert logger1 is logger2
    assert len(logger2.handlers) == num_handlers
