"""Shared utilities: configuration, logging, IO."""
from .config import CONFIG, PATHS
from .logger import get_logger

__all__ = ["CONFIG", "PATHS", "get_logger"]
