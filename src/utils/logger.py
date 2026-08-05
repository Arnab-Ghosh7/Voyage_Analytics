"""Small, dependency-free logging helper used across the pipeline."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "voyage", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once.

    Idempotent: repeated calls with the same name do not stack handlers, so
    importing this in a notebook cell you re-run will not duplicate log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
