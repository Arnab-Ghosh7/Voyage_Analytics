"""Data ingestion package.

All logic lives in the single module :mod:`src.data_ingestion.data_ingestion`
(IO + cleaning + feature build + typed readers + CLI). Data quality checks are
**not** here — see :mod:`src.validation`, which runs as its own pipeline stage.
This ``__init__`` re-exports the public surface so callers can simply write::

    from src.data_ingestion import run_pipeline, load_processed

Run the pipeline from the command line with::

    python -m src.data_ingestion
"""
from .data_ingestion import (
    # IO
    load_raw,
    # cleaning -> data/interim
    clean_flights,
    clean_hotels,
    clean_users,
    # feature build -> data/processed
    build_trips,
    build_user_features,
    build_flight_price_table,
    _attach_hotels,
    # typed readers
    load_interim,
    load_processed,
    load_all_processed,
    # driver
    run_pipeline,
    main,
)

__all__ = [
    "load_raw",
    "clean_flights",
    "clean_hotels",
    "clean_users",
    "build_trips",
    "build_user_features",
    "build_flight_price_table",
    "_attach_hotels",
    "load_interim",
    "load_processed",
    "load_all_processed",
    "run_pipeline",
    "main",
]
