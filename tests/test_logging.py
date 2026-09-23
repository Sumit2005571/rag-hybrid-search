"""Unit tests for logging configuration."""

import logging
from app.logging_config import get_logger, setup_logging


def test_setup_logging():
    """Verify setup_logging sets root logger level."""
    setup_logging("DEBUG")
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG

    setup_logging("INFO")
    assert root_logger.level == logging.INFO


def test_get_logger():
    """Verify get_logger returns an instance with matching name."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
