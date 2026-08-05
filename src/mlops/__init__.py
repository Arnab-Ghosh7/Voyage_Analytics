"""MLOps package — experiment tracking and model registry.

All logic lives in :mod:`src.mlops.tracking`, mirroring the layout used by
:mod:`src.data_ingestion`, :mod:`src.features` and :mod:`src.validation`::

    from src.mlops import run, setup_mlflow, list_runs

Run from the command line with::

    python -m src.mlops
"""
from .tracking import (
    EXPERIMENT,
    REGISTERED_NAMES,
    TRACKING_DIR,
    list_runs,
    register_model,
    run,
    setup_mlflow,
    track_flight_price,
    track_gender,
    track_recommender,
    main,
)

__all__ = [
    "EXPERIMENT",
    "REGISTERED_NAMES",
    "TRACKING_DIR",
    "list_runs",
    "register_model",
    "run",
    "setup_mlflow",
    "track_flight_price",
    "track_gender",
    "track_recommender",
    "main",
]
