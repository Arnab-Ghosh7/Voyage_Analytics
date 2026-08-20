import pytest
import pandas as pd
from src.validation.validation import validate_raw_data


def test_validate_raw_data_schema_and_integrity():
    flights = pd.DataFrame({
        "userCode": [1], "from": ["Florianopolis (SC)"], "to": ["Recife (PE)"],
        "flightType": ["economic"], "price": [500.0], "time": [2.5],
        "distance": [1000.0], "agency": ["Gol"], "date": ["2019-01-01"]
    })
    hotels = pd.DataFrame({
        "userCode": [1], "name": ["Hotel Copacabana"], "place": ["Recife (PE)"],
        "days": [3], "price": [150.0], "total": [450.0], "date": ["2019-01-01"]
    })
    users = pd.DataFrame({
        "code": [1], "company": ["Acme"], "name": ["Alice Smith"],
        "gender": ["f"], "age": [30]
    })
    passed, summary, logs = validate_raw_data(flights, hotels, users)
    assert passed is True
    assert summary["n_errors"] == 0
