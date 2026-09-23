"""Logging configuration module for RAG Hybrid Search."""

import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:
    """Configure root logger with a standardized format.

    Args:
        log_level: Desired log level string (e.g. 'DEBUG', 'INFO', 'WARNING').
                   Defaults to INFO if not provided.
    """
    level_str = (log_level or "INFO").upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the given name.

    Args:
        name: Module or component name for the logger.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
