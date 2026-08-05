"""Tests for the Flask prediction API.

Run with::

    pytest tests/ -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving import ARTIFACTS, create_app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in ARTIFACTS.values()),
    reason="model artifacts missing — run `python main.py` first",
)

VALID_FLIGHT = {"from": "Sao Paulo (SP)", "to": "Rio de Janeiro (RJ)",
                "flightType": "firstClass", "agency": "Rainbow"}


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


# --------------------------------------------------------------------- meta --
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy"


def test_index_lists_endpoints(client):
    body = client.get("/").get_json()
    assert "/predict/flight-price" in body["endpoints"]
    # the chance-level model must not be advertised as servable
    assert "hotel_attach" in body["not_served"]


def test_reference_exposes_valid_values(client):
    body = client.get("/reference").get_json()
    assert body["n_routes"] == 70
    assert len(body["cities"]) == 9
    assert set(body["flight_types"]) == {"economic", "premium", "firstClass"}


# ------------------------------------------------------------- flight price --
def test_flight_price_returns_plausible_fare(client):
    body = client.post("/predict/flight-price", json=VALID_FLIGHT).get_json()
    assert 100 < body["predicted_price"] < 5000
    assert body["currency"] == "BRL"
    # distance is derived by the API, never supplied by the caller
    assert body["derived"]["distance_km"] > 0


def test_flight_price_respects_class_ladder(client):
    """first class > premium > economic on the same route (the EDA fare ladder)."""
    prices = {}
    for ft in ("economic", "premium", "firstClass"):
        payload = {**VALID_FLIGHT, "flightType": ft}
        prices[ft] = client.post("/predict/flight-price", json=payload
                                 ).get_json()["predicted_price"]
    assert prices["economic"] < prices["premium"] < prices["firstClass"]


@pytest.mark.parametrize("payload,expected", [
    ({"from": "Atlantis", "to": "Natal (RN)", "flightType": "economic",
      "agency": "Rainbow"}, "Invalid city"),
    ({"from": "Sao Paulo (SP)"}, "Missing required field"),
    ({"from": "Natal (RN)", "to": "Natal (RN)", "flightType": "economic",
      "agency": "Rainbow"}, "must differ"),
    ({"from": "Sao Paulo (SP)", "to": "Natal (RN)", "flightType": "business",
      "agency": "Rainbow"}, "Invalid flightType"),
])
def test_flight_price_rejects_bad_input(client, payload, expected):
    r = client.post("/predict/flight-price", json=payload)
    assert r.status_code == 400
    assert expected in r.get_json()["error"]


def test_unflown_route_is_rejected(client):
    """Rio <-> Salvador are real cities but the pair is never flown."""
    r = client.post("/predict/flight-price", json={
        "from": "Rio de Janeiro (RJ)", "to": "Salvador (BH)",
        "flightType": "economic", "agency": "Rainbow"})
    assert r.status_code == 400
    assert "No route" in r.get_json()["error"]


# ------------------------------------------------------------------ gender --
def test_gender_returns_label_and_caveat(client):
    body = client.post("/predict/gender", json={"name": "Charlotte Johnson"}).get_json()
    assert body["predicted_gender"] in {"male", "female"}
    assert body["inputs"]["first_name_used"] == "charlotte"
    # the response must carry what the model actually infers from
    assert "caveat" in body["model"]


def test_gender_requires_name(client):
    r = client.post("/predict/gender", json={})
    assert r.status_code == 400


# ------------------------------------------------------------- recommender --
def test_recommend_with_destination_is_exact(client):
    body = client.post("/recommend/hotels",
                       json={"destination": "Salvador (BH)", "top_k": 3}).get_json()
    assert body["strategy"] == "destination-aware"
    assert body["recommendations"][0]["city"] == "Salvador (BH)"


def test_recommend_without_destination_falls_back(client):
    body = client.post("/recommend/hotels", json={"top_k": 3}).get_json()
    assert body["strategy"] == "popularity"
    assert len(body["recommendations"]) == 3


def test_recommend_rejects_out_of_range_k(client):
    assert client.post("/recommend/hotels", json={"top_k": 99}).status_code == 400


# ----------------------------------------------------------------- latency --
def test_latency_within_prd_budget(client):
    """PRD requires inference under 500 ms; warmup happens at app creation."""
    for _ in range(3):
        client.post("/predict/flight-price", json=VALID_FLIGHT)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        client.post("/predict/flight-price", json=VALID_FLIGHT)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[int(0.95 * len(times))]
    assert p95 < 500, f"p95 latency {p95:.1f} ms exceeds the 500 ms budget"


def test_response_carries_latency_header(client):
    r = client.get("/health")
    assert "X-Response-Time-ms" in r.headers
