import pytest
import pandas as pd
from src.data_ingestion.data_ingestion import clean_flights, clean_hotels, clean_users


def test_clean_flights_derives_route_and_rate():
    raw = pd.DataFrame({
        "userCode": [1, 2],
        "from": ["Florianopolis (SC)", "Sao Paulo (SP)"],
        "to": ["Recife (PE)", "Rio de Janeiro (RJ)"],
        "flightType": ["economic", "firstClass"],
        "price": [500.0, 1200.0],
        "time": [2.5, 1.5],
        "distance": [1000.0, 400.0],
        "agency": ["Gol", "Tam"],
        "date": ["2019-01-01", "2019-01-02"]
    })
    df = clean_flights(raw)
    assert "route" in df.columns
    assert "rate_per_km" in df.columns
    assert len(df) == 2
    assert df["route"].iloc[0] == "Florianopolis (SC) -> Recife (PE)"


def test_clean_hotels_derives_total_spend():
    raw = pd.DataFrame({
        "userCode": [1],
        "name": ["Hotel Copacabana"],
        "place": ["Rio de Janeiro (RJ)"],
        "days": [3],
        "price": [150.0],
        "total": [450.0],
        "date": ["2020-05-10"]
    })
    df = clean_hotels(raw)
    assert "total" in df.columns
    assert df["total"].iloc[0] == 450.0


def test_clean_users_parses_gender():
    raw = pd.DataFrame({
        "code": [101, 102],
        "company": ["Acme", "Beta"],
        "name": ["Alice Smith", "Bob Jones"],
        "gender": ["f", "m"],
        "age": [28, 45]
    })
    df = clean_users(raw)
    assert "gender" in df.columns
    assert list(df["gender"]) == ["f", "m"]
