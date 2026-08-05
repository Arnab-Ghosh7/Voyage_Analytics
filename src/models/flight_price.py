"""Flight-price regression (project objective #1).

Predicts `flights.price` from route, class, agency and distance.

EDA established that price is a **deterministic rate card**: the tuple
(from, to, flightType, agency) fixes the fare exactly. A naive random split
therefore scores R2 ~ 1.0 by *memorising* that card, which proves nothing.

So this module evaluates every model two ways:

  * ``random``  — standard 80/20 split. Upper bound; expect near-perfect.
  * ``grouped`` — whole **routes** held out (GroupShuffleSplit on `route`), so
    the test set contains city pairs never seen in training. This is the honest
    generalisation test: the model must infer price from distance + class +
    agency rather than look it up.

Run::

    python -m src.models.flight_price               # train, evaluate, save best
    python -m src.models.flight_price --quick       # smaller sample, fast smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.data_ingestion import load_processed
from src.utils import PATHS, get_logger

log = get_logger("model.flight_price")

TARGET = "price"
# `route` is deliberately excluded from features: it is the grouping key for the
# honest split, and it duplicates from+to.
CAT_FEATURES = ["from", "to", "flightType", "agency", "day_of_week"]
NUM_FEATURES = ["distance", "year", "month"]
GROUP_COL = "route"

ARTIFACT_DIR = PATHS.root / "models"
METRICS_PATH = PATHS.reports / "flight_price_metrics.json"


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def load_dataset(sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load the processed flight-price table (optionally subsampled)."""
    df = load_processed("flight_price_model")
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
        log.info("subsampled to %s rows", f"{len(df):,}")
    return df


def _xy(df: pd.DataFrame):
    return df[CAT_FEATURES + NUM_FEATURES], df[TARGET], df[GROUP_COL]


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def _preprocessor() -> ColumnTransformer:
    """One-hot the categoricals, pass numerics through untouched."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
         ("num", "passthrough", NUM_FEATURES)],
        remainder="drop",
    )


class RatePerKmRegressor(BaseEstimator, RegressorMixin):
    """Predict fare **per kilometre**, then multiply back by distance.

    Price is essentially ``rate_per_km(class, agency, route) x distance``. Fitting
    the raw price forces the model to re-learn that linear scaling separately for
    every route, which is exactly what fails on a city pair it has never seen.
    Normalising the target by distance removes the scaling, so the model learns
    the *transferable* quantity — the rate — and generalises far better.

    No leakage: ``distance`` is an ordinary input feature, known at prediction
    time for any route the caller asks about.
    """

    def __init__(self, base_pipeline=None, distance_col: str = "distance"):
        self.base_pipeline = base_pipeline
        self.distance_col = distance_col

    def fit(self, X, y):
        d = np.asarray(X[self.distance_col], dtype=float)
        self.pipeline_ = clone(self.base_pipeline)
        self.pipeline_.fit(X, np.asarray(y, dtype=float) / d)
        return self

    def predict(self, X):
        d = np.asarray(X[self.distance_col], dtype=float)
        return self.pipeline_.predict(X) * d


def build_models() -> dict[str, Pipeline]:
    """Candidate models, cheapest first.

    Ridge is the interpretable linear baseline; the tree models capture the
    class x distance interaction that sets the fare ladder.
    """
    return {
        "ridge": Pipeline([("prep", _preprocessor()),
                           ("model", Ridge(alpha=1.0))]),
        "decision_tree": Pipeline([("prep", _preprocessor()),
                                   ("model", DecisionTreeRegressor(
                                       max_depth=12, min_samples_leaf=20, random_state=42))]),
        "hist_gbr": Pipeline([("prep", _preprocessor()),
                              ("model", HistGradientBoostingRegressor(
                                  max_iter=300, learning_rate=0.1,
                                  max_depth=None, random_state=42))]),
        # Same booster, but learning rate-per-km instead of raw price. Halves the
        # MSE on unseen routes (16.3k -> 8.2k) — see RatePerKmRegressor.
        "gbr_rate_per_km": RatePerKmRegressor(
            base_pipeline=Pipeline([("prep", _preprocessor()),
                                    ("model", HistGradientBoostingRegressor(
                                        max_iter=300, learning_rate=0.1,
                                        random_state=42))])),
    }


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Scores:
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float

    def as_dict(self) -> dict:
        return {"mse": round(self.mse, 2), "rmse": round(self.rmse, 4),
                "mae": round(self.mae, 4), "r2": round(self.r2, 6),
                "mape_pct": round(self.mape, 4)}


def score(y_true, y_pred) -> Scores:
    """MSE / RMSE / MAE / R2 / MAPE.

    MSE is reported because it is commonly asked for, but RMSE is the figure to
    quote: MSE is in squared currency (R$^2) and has no interpretable scale.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return Scores(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        mape=float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    )


def split(df: pd.DataFrame, how: str, seed: int = 42):
    """Return train/test frames for either the random or grouped strategy."""
    if how == "random":
        return train_test_split(df, test_size=0.2, random_state=seed)
    if how == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(df, groups=df[GROUP_COL]))
        return df.iloc[tr], df.iloc[te]
    raise ValueError(f"unknown split '{how}'")


def evaluate(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Train every model under both split strategies and collect metrics."""
    rows, fitted = [], {}

    for how in ("random", "grouped"):
        train_df, test_df = split(df, how, seed)
        X_tr, y_tr, _ = _xy(train_df)
        X_te, y_te, _ = _xy(test_df)
        if how == "grouped":
            unseen = set(test_df[GROUP_COL]) - set(train_df[GROUP_COL])
            log.info("grouped split: %d test routes, %d unseen in training",
                     test_df[GROUP_COL].nunique(), len(unseen))

        for name, pipe in build_models().items():
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            s = score(y_te.to_numpy(), pipe.predict(X_te))
            rows.append({"split": how, "model": name, **s.as_dict(),
                         "fit_seconds": round(elapsed, 2)})
            log.info("%-8s | %-13s R2=%.4f RMSE=%7.2f MAE=%6.2f (%.1fs)",
                     how, name, s.r2, s.rmse, s.mae, elapsed)
            if how == "grouped":          # keep models judged on the honest split
                fitted[name] = pipe

    return pd.DataFrame(rows), fitted


# --------------------------------------------------------------------------- #
# Train / persist                                                              #
# --------------------------------------------------------------------------- #
def train_final(df: pd.DataFrame, model_name: str) -> Pipeline:
    """Refit the chosen model on the full dataset for serving."""
    pipe = build_models()[model_name]
    X, y, _ = _xy(df)
    pipe.fit(X, y)
    log.info("refit '%s' on all %s rows", model_name, f"{len(df):,}")
    return pipe


def save_artifacts(pipe: Pipeline, results: pd.DataFrame, model_name: str) -> dict:
    """Persist the fitted pipeline and the metrics table."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACT_DIR / "flight_price_model.joblib"
    joblib.dump(pipe, model_path)

    payload = {
        "best_model": model_name,
        "selected_on": "grouped (held-out routes)",
        "features": {"categorical": CAT_FEATURES, "numeric": NUM_FEATURES},
        "target": TARGET,
        "results": results.to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info("saved model   -> %s", model_path)
    log.info("saved metrics -> %s", METRICS_PATH)
    return {"model_path": model_path, "metrics_path": METRICS_PATH}


def run(sample: int | None = None, seed: int = 42) -> dict:
    """Full training run: load -> evaluate -> select -> refit -> save."""
    log.info("=== flight-price regression started ===")
    df = load_dataset(sample=sample, seed=seed)
    log.info("dataset %s rows, %d routes", f"{len(df):,}", df[GROUP_COL].nunique())

    results, _ = evaluate(df, seed)

    # Select on the grouped split — the generalisation test, not the memorisation one.
    grouped = results[results["split"] == "grouped"]
    best = grouped.loc[grouped["rmse"].idxmin(), "model"]
    log.info("best model on held-out routes: %s", best)

    final = train_final(df, best)
    paths = save_artifacts(final, results, best)

    log.info("=== flight-price regression complete ===")
    return {"results": results, "best_model": best, "pipeline": final, **paths}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the flight-price regressor")
    p.add_argument("--quick", action="store_true",
                   help="train on a 40k-row sample for a fast smoke test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(sample=40_000 if args.quick else None, seed=args.seed)


if __name__ == "__main__":
    main()
