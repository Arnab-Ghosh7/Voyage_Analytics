import pytest
import pandas as pd
from src.features.features import build_gender_features


def test_build_gender_features_extracts_first_name():
    df = pd.DataFrame({
        "userCode": [1, 2],
        "name": ["Charlotte Johnson", "Lucas Silva"],
        "gender": ["f", "m"],
        "age": [30, 40],
        "company": ["Company A", "Company B"]
    })
    res = build_gender_features(df)
    assert "first_name" in res.columns
    assert res["first_name"].tolist() == ["charlotte", "lucas"]
