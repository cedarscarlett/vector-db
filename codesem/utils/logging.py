# codesem/utils/logging.py

from __future__ import annotations

import logging
import sys
from typing import Optional


# ============================================================
# Logging Configuration
# ============================================================

_LOGGER_NAME = "codesem"


def configure_logging(debug: bool = False) -> logging.Logger:
    """
    Configure root logger for CodeSem.

    - INFO level by default
    - DEBUG level if debug=True
    - StreamHandler to stdout
    - Simple structured format

    Returns configured logger.
    """
    level = logging.DEBUG if debug else logging.INFO

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Prevent duplicate handlers if configure_logging is called multiple times,
    # but always ensure handler levels match the current logger level.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure all existing handlers reflect the current level
    for handler in logger.handlers:
        handler.setLevel(level)

    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a child logger under the main CodeSem namespace.

    Example:
        logger = get_logger("indexer")
    """
    base_logger = logging.getLogger(_LOGGER_NAME)

    if name:
        return base_logger.getChild(name)

    return base_logger
