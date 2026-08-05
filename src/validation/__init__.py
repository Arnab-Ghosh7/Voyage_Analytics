"""Data validation package.

All logic lives in the single module :mod:`src.validation.validation`, mirroring
the layout of :mod:`src.data_ingestion` and :mod:`src.features`::

    from src.validation import run_validation, validate_raw, validate_processed

Run from the command line with::

    python -m src.validation                    # raw (pipeline gate)
    python -m src.validation --scope processed  # audit generated tables
"""
from .validation import (
    DataQualityReport,
    EXPECTED_TABLES,
    REPORT_PATH,
    load_raw_frames,
    validate_raw,
    validate_processed,
    run_validation,
    main,
)

__all__ = [
    "DataQualityReport",
    "EXPECTED_TABLES",
    "REPORT_PATH",
    "load_raw_frames",
    "validate_raw",
    "validate_processed",
    "run_validation",
    "main",
]
