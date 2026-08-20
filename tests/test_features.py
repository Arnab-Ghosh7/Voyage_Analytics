"""Tests for the feature engineering layer.

Corrected against the real module API. The previous version asserted
``"first_name" in FEATURE_REGISTRY`` (the registry holds ``FeatureSpec`` objects,
never strings) and called ``make_preprocessor(cat_cols=..., num_cols=...)``
(the parameters are ``cat_features`` / ``num_features``), so both failed on the
API rather than on behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.features import (FEATURE_REGISTRY, FeatureSpec, describe,  # noqa: E402
                                   make_preprocessor, registry_frame, registry_names)


# ---------------------------------------------------------------- registry --
def test_registry_is_populated_with_specs():
    assert len(FEATURE_REGISTRY) > 0
    assert all(isinstance(f, FeatureSpec) for f in FEATURE_REGISTRY)


def test_registry_documents_the_gender_signal():
    """first_name is the only feature that predicts gender, so it must be documented."""
    assert "first_name" in registry_names()
    spec = describe("first_name")
    assert spec is not None and spec.source == "users"


@pytest.mark.parametrize("name", ["age", "company", "home_city", "n_trips",
                                  "recency_days", "total_spend"])
def test_core_features_are_registered(name):
    assert name in registry_names()


def test_every_spec_is_fully_described():
    for f in FEATURE_REGISTRY:
        assert f.name and f.dtype and f.source and f.description, f"incomplete: {f}"


def test_registry_names_are_unique():
    names = [f.name for f in FEATURE_REGISTRY]
    assert len(names) == len(set(names))


def test_registry_frame_shape():
    df = registry_frame()
    assert len(df) == len(FEATURE_REGISTRY)
    assert {"name", "dtype", "source", "description"} <= set(df.columns)


# ------------------------------------------------------------ preprocessor --
def test_make_preprocessor_fits_and_transforms():
    prep = make_preprocessor(cat_features=["agency"], num_features=["distance"])
    X = pd.DataFrame({"agency": ["Rainbow", "CloudFy", "Rainbow"],
                      "distance": [400.0, 900.0, 400.0]})
    out = prep.fit_transform(X)
    # 2 one-hot columns + 1 numeric
    assert out.shape == (3, 3)


def test_make_preprocessor_handles_unseen_categories():
    """Serving must not explode on a category absent from training."""
    prep = make_preprocessor(cat_features=["agency"], num_features=["distance"])
    prep.fit(pd.DataFrame({"agency": ["Rainbow"], "distance": [400.0]}))
    out = prep.transform(pd.DataFrame({"agency": ["BrandNew"], "distance": [500.0]}))
    assert out.shape[0] == 1


def test_make_preprocessor_imputes_missing_numerics():
    prep = make_preprocessor(cat_features=["agency"], num_features=["distance"])
    X = pd.DataFrame({"agency": ["Rainbow", "CloudFy", "Rainbow"],
                      "distance": [400.0, None, 600.0]})
    out = prep.fit_transform(X)
    assert not pd.isna(out).any(), "missing numerics should be imputed, not propagated"


def test_scale_flag_is_respected():
    unscaled = make_preprocessor(cat_features=[], num_features=["distance"], scale=False)
    X = pd.DataFrame({"distance": [100.0, 200.0, 300.0]})
    out = unscaled.fit_transform(X)
    assert out.max() == 300.0, "scale=False should leave values in their original units"
