"""Tests for the ingestion/cleaning layer.

Fixed against the real pipeline. The previous version used ISO dates
(``"2019-01-01"``) while the raw CSVs — and therefore ``CONFIG.date_format`` —
are ``MM/DD/YYYY``, and omitted ``travelCode``, which ``clean_flights`` needs to
tag outbound/return legs. It also asserted a ``rate_per_km`` column; the derived
column is ``price_per_km``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_ingestion.data_ingestion import (clean_flights, clean_hotels,  # noqa: E402
                                               clean_users)


# ---------------------------------------------------------------- flights --
def _raw_flights() -> pd.DataFrame:
    """One round trip: out on the 1st, back on the 4th."""
    return pd.DataFrame({
        "travelCode": [1, 1],
        "userCode": [1, 1],
        "from": ["Florianopolis (SC)", "Recife (PE)"],
        "to": ["Recife (PE)", "Florianopolis (SC)"],
        "flightType": ["economic", "economic"],
        "price": [500.0, 500.0],
        "time": [2.5, 2.5],
        "distance": [1000.0, 1000.0],
        "agency": ["Rainbow", "Rainbow"],
        "date": ["01/01/2019", "01/04/2019"],      # MM/DD/YYYY, as in the raw CSVs
    })


def test_clean_flights_derives_route_and_unit_price():
    df = clean_flights(_raw_flights())
    assert "route" in df.columns
    # The derived unit-price column is `price_per_km`, not `rate_per_km`.
    assert "price_per_km" in df.columns
    assert (df["price_per_km"] == (df["price"] / df["distance"]).round(4)).all()
    assert len(df) == 2
    assert df["route"].iloc[0] == "Florianopolis (SC) -> Recife (PE)"


def test_clean_flights_tags_outbound_and_return():
    """Leg ordering is the assumption every trip-level feature depends on."""
    df = clean_flights(_raw_flights())
    assert list(df["leg"]) == ["outbound", "return"]
    assert df.loc[df["leg"] == "outbound", "from"].iloc[0] == "Florianopolis (SC)"


def test_clean_flights_parses_dates_and_calendar_parts():
    df = clean_flights(_raw_flights())
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["year"].iloc[0] == 2019
    assert df["month"].iloc[0] == 1
    assert df["day_of_week"].iloc[0] == "Tuesday"      # 2019-01-01


def test_clean_flights_uses_categoricals():
    """Categorical dtypes keep the 271k-row frame small."""
    df = clean_flights(_raw_flights())
    for col in ("from", "to", "flightType", "agency", "leg"):
        assert isinstance(df[col].dtype, pd.CategoricalDtype), col


# ----------------------------------------------------------------- hotels --
def test_clean_hotels_preserves_total_and_parses_dates():
    raw = pd.DataFrame({
        "travelCode": [1], "userCode": [1], "name": ["Hotel A"],
        "place": ["Rio de Janeiro (RJ)"], "days": [3], "price": [150.0],
        "total": [450.0], "date": ["05/10/2020"],
    })
    df = clean_hotels(raw)
    assert df["total"].iloc[0] == 450.0
    assert df["total"].iloc[0] == df["price"].iloc[0] * df["days"].iloc[0]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["year"].iloc[0] == 2020 and df["month"].iloc[0] == 5


# ------------------------------------------------------------------ users --
def test_clean_users_normalises_gender():
    raw = pd.DataFrame({
        "code": [101, 102, 103],
        "company": ["Acme Factory", "4You", "Wonka Company"],
        "name": ["Alice Smith", "Bob Jones", "Carol White"],
        "gender": ["FEMALE", " male ", "none"],
        "age": [28, 45, 33],
    })
    df = clean_users(raw)
    assert list(df["gender"]) == ["female", "male", "none"]


def test_clean_users_flags_undisclosed_gender():
    """`none` is an undisclosed label, not a third gender — it drives imputation."""
    raw = pd.DataFrame({
        "code": [1, 2], "company": ["Acme Factory", "4You"],
        "name": ["A B", "C D"], "gender": ["female", "none"], "age": [30, 40],
    })
    df = clean_users(raw)
    assert list(df["gender_known"]) == [True, False]
