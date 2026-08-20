"""Tests for the data-quality validation stage.

Rewritten against the real API. The previous version called
``validate_raw_data(flights, hotels, users)`` expecting a
``(passed, summary, logs)`` tuple — no such function exists. The module exposes
``validate_raw(frames: dict) -> DataQualityReport``.

These tests exercise the behaviour that matters: which violations *stop* the
pipeline (errors) versus which are merely recorded (warnings).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.validation import DataQualityReport, validate_raw  # noqa: E402


def _frames() -> dict[str, pd.DataFrame]:
    """A minimal, fully valid set of raw frames."""
    flights = pd.DataFrame({
        "travelCode": [1, 1],
        "userCode": [1, 1],
        "from": ["Florianopolis (SC)", "Recife (PE)"],
        "to": ["Recife (PE)", "Florianopolis (SC)"],
        "flightType": ["economic", "economic"],
        "price": [500.0, 500.0],
        "time": [2.5, 2.5],
        "distance": [1000.0, 1000.0],
        "agency": ["Rainbow", "Rainbow"],
        "date": ["01/01/2019", "01/04/2019"],
    })
    hotels = pd.DataFrame({
        "travelCode": [1], "userCode": [1], "name": ["Hotel A"],
        "place": ["Recife (PE)"], "days": [3], "price": [150.0],
        "total": [450.0], "date": ["01/01/2019"],
    })
    users = pd.DataFrame({
        "code": [1], "company": ["Acme Factory"], "name": ["Alice Smith"],
        "gender": ["female"], "age": [30],
    })
    return {"flights": flights, "hotels": hotels, "users": users}


# ------------------------------------------------------------- happy path --
def test_valid_frames_pass():
    report = validate_raw(_frames())
    assert isinstance(report, DataQualityReport)
    assert report.ok is True
    assert report.errors == []
    assert len(report.passed) > 0
    assert "PASS" in report.summary()


def test_report_records_row_counts():
    report = validate_raw(_frames())
    assert report.stats["flights"]["rows"] == 2
    assert report.stats["users"]["rows"] == 1


# ----------------------------------------------------------------- errors --
def test_missing_column_is_an_error():
    frames = _frames()
    frames["flights"] = frames["flights"].drop(columns=["price"])
    report = validate_raw(frames)
    assert report.ok is False
    assert any("missing columns" in e for e in report.errors)


def test_out_of_bounds_value_is_an_error():
    frames = _frames()
    frames["flights"].loc[0, "price"] = -999.0
    report = validate_raw(frames)
    assert report.ok is False
    assert any("outside" in e for e in report.errors)


def test_orphan_foreign_key_is_an_error():
    """A booking for a user who does not exist must stop the pipeline."""
    frames = _frames()
    frames["hotels"].loc[0, "userCode"] = 999_999
    report = validate_raw(frames)
    assert report.ok is False
    assert any("not present in users" in e for e in report.errors)


def test_hotel_without_matching_flight_is_an_error():
    frames = _frames()
    frames["hotels"].loc[0, "travelCode"] = 42
    report = validate_raw(frames)
    assert report.ok is False
    assert any("no matching flight" in e for e in report.errors)


# --------------------------------------------------------------- warnings --
@pytest.mark.parametrize("column,value", [("agency", "BrandNewAir"),
                                          ("flightType", "business")])
def test_unknown_category_warns_but_does_not_block(column, value):
    """A new agency or class should be surfaced, not treated as corruption."""
    frames = _frames()
    frames["flights"].loc[0, column] = value
    report = validate_raw(frames)
    assert report.ok is True, "an unseen category must not stop the pipeline"
    assert any(column in w for w in report.warnings)


def test_nulls_warn_but_do_not_block():
    frames = _frames()
    frames["users"].loc[0, "company"] = None
    report = validate_raw(frames)
    assert report.ok is True
    assert any("null" in w for w in report.warnings)


def test_duplicate_rows_warn():
    frames = _frames()
    frames["users"] = pd.concat([frames["users"]] * 2, ignore_index=True)
    report = validate_raw(frames)
    assert any("duplicate" in w for w in report.warnings)


# ----------------------------------------------------------------- report --
def test_report_serialises():
    report = validate_raw(_frames())
    payload = report.as_dict()
    assert payload["ok"] is True
    assert {"scope", "errors", "warnings", "passed", "stats"} <= set(payload)

    frame = report.to_frame()
    assert set(frame["severity"]).issubset({"error", "warning", "passed"})
