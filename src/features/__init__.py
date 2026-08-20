"""Feature engineering package.

All logic lives in the single module :mod:`src.features.features`, mirroring the
layout of :mod:`src.data_ingestion`. This ``__init__`` re-exports its public
surface::

    from src.features import build_gender_features, make_preprocessor

Build every feature set from the command line with::

    python -m src.features
"""
from .features import (
    # registry / shared encoding
    FeatureSpec,
    FEATURE_REGISTRY,
    registry_frame,
    registry_names,
    describe,
    make_preprocessor,
    # builders
    build_user_behaviour,
    build_gender_features,
    build_user_hotel_matrix,
    build_hotel_catalog,
    # feature lists
    GENDER_CAT_FEATURES,
    GENDER_NUM_FEATURES,
    GENDER_TARGET,
    # driver
    build_all,
    main,
)

__all__ = [
    "FeatureSpec",
    "FEATURE_REGISTRY",
    "registry_frame",
    "registry_names",
    "describe",
    "make_preprocessor",
    "build_user_behaviour",
    "build_gender_features",
    "build_user_hotel_matrix",
    "build_hotel_catalog",
    "GENDER_CAT_FEATURES",
    "GENDER_NUM_FEATURES",
    "GENDER_TARGET",
    "build_all",
    "main",
]
