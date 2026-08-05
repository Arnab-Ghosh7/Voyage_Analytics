"""Model serving package — Flask REST API.

All logic lives in :mod:`src.serving.app`, mirroring the layout used by the other
packages::

    from src.serving import app, create_app, build_route_reference

Run the development server with::

    python -m src.serving
"""
from .app import (
    ARTIFACTS,
    ApiError,
    app,
    build_route_reference,
    create_app,
    load_model,
    main,
    reference,
)

__all__ = [
    "ARTIFACTS",
    "ApiError",
    "app",
    "build_route_reference",
    "create_app",
    "load_model",
    "main",
    "reference",
]
