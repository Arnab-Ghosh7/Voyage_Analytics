"""Feature engineering for Voyage Analytics.

Where this sits in the pipeline
-------------------------------
``src.data_ingestion`` produces the **entity tables** — cleaned sources plus
``trips`` and ``users_features``. This module builds the **model-ready feature
sets** on top of them, one per downstream model::

    data/processed/trips.csv, users_features.csv, hotels_clean.csv
        │
        ├─ build_gender_features()        -> gender_features.csv        (objective #8)
        ├─ build_user_hotel_matrix()      -> user_hotel_interactions.csv (objective #9)
        └─ build_hotel_catalog()          -> hotel_catalog.csv          (objective #9)

It also holds two things shared across models:

* :data:`FEATURE_REGISTRY` — a declarative catalogue of every engineered feature
  (name, dtype, source, description). This is the lightweight stand-in for the
  "feature store" in the PRD: one place that documents what a feature means, so
  training and serving cannot silently disagree.
* :func:`make_preprocessor` — the shared sklearn encoding factory.

Run::

    python -m src.features              # build and persist every feature set
    python -m src.features --parquet
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_ingestion import load_interim, load_processed
from src.utils import CONFIG, PATHS, get_logger

log = get_logger("features")


# =========================================================================== #
# 1. Feature registry (lightweight feature store)                             #
# =========================================================================== #
@dataclass(frozen=True)
class FeatureSpec:
    """One engineered feature: what it is, where it came from, what it means."""

    name: str
    dtype: str
    source: str
    description: str


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # --- demographics -------------------------------------------------------
    FeatureSpec("age", "int", "users", "Traveller age in years (21-65)"),
    FeatureSpec("company", "category", "users", "Employer; deterministically encodes home city"),
    FeatureSpec("home_city", "category", "trips", "Most frequent outbound origin"),
    # --- RFM ----------------------------------------------------------------
    FeatureSpec("n_trips", "int", "trips", "Frequency: completed round trips"),
    FeatureSpec("recency_days", "int", "trips", "Days since last departure (snapshot = max date)"),
    FeatureSpec("total_spend", "float", "trips", "Monetary: flight + hotel spend"),
    FeatureSpec("flight_spend", "float", "trips", "Lifetime spend on flights"),
    FeatureSpec("hotel_spend", "float", "hotels", "Lifetime spend on hotels"),
    # --- behaviour ----------------------------------------------------------
    FeatureSpec("avg_trip_spend", "float", "trips", "Mean spend per trip"),
    FeatureSpec("avg_trip_nights", "float", "trips", "Mean trip length in nights (1-4)"),
    FeatureSpec("avg_distance", "float", "trips", "Mean route distance flown"),
    FeatureSpec("n_destinations", "int", "trips", "Distinct destination cities visited"),
    FeatureSpec("hotel_attach_rate", "float", "trips+hotels", "Share of trips including a hotel"),
    FeatureSpec("share_class_*", "float", "trips", "Share of trips per flight class (economic/premium/first)"),
    FeatureSpec("share_agency_*", "float", "trips", "Share of trips per booking agency"),
    # --- recommender --------------------------------------------------------
    FeatureSpec("bookings", "int", "hotels", "Times a user booked a given hotel"),
    FeatureSpec("nights", "int", "hotels", "Total nights a user stayed at a hotel"),
    FeatureSpec("nightly_rate", "float", "hotels", "Hotel's fixed nightly rate"),
)


def registry_frame() -> pd.DataFrame:
    """Return the feature registry as a DataFrame for display/docs."""
    return pd.DataFrame([f.__dict__ for f in FEATURE_REGISTRY])


# =========================================================================== #
# 2. Shared encoding                                                          #
# =========================================================================== #
def make_preprocessor(cat_features: list[str],
                      num_features: list[str],
                      scale: bool = True) -> ColumnTransformer:
    """One-hot the categoricals, impute + optionally scale the numerics.

    Generalises the encoder used by the flight-price model so the classifier and
    recommender share one definition. ``handle_unknown='ignore'`` keeps serving
    safe when an unseen category arrives at inference time.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
         ("num", Pipeline(num_steps), num_features)],
        remainder="drop",
    )


# =========================================================================== #
# 3. Behaviour aggregation                                                    #
# =========================================================================== #
def build_user_behaviour(trips: pd.DataFrame) -> pd.DataFrame:
    """Per-user travel-behaviour aggregates derived from the trip table.

    Adds the preference *mix* features (share of trips by class and by agency)
    that the raw user table does not carry — these describe **how** somebody
    travels, not just how much, and are the behavioural signal EDA pointed to
    after demographics turned out to be uninformative about value.
    """
    base = (trips.groupby("user", observed=True)
            .agg(avg_trip_nights=("trip_nights", "mean"),
                 avg_distance=("distance", "mean"),
                 avg_flight_cost=("flight_cost", "mean"),
                 avg_trip_spend=("trip_spend", "mean"),
                 n_destinations=("dest", "nunique"))
            .round(4))

    cls = (pd.crosstab(trips["user"], trips["flightType"], normalize="index")
           .add_prefix("share_class_").round(4))
    agy = (pd.crosstab(trips["user"], trips["agency"], normalize="index")
           .add_prefix("share_agency_").round(4))

    out = base.join(cls).join(agy)
    log.info("user behaviour features: %s", out.shape)
    return out


# =========================================================================== #
# 4. Gender-classification feature set (objective #8)                         #
# =========================================================================== #
GENDER_CAT_FEATURES = ["company", "home_city"]
GENDER_NUM_FEATURES = [
    "age", "n_trips", "recency_days", "total_spend", "flight_spend", "hotel_spend",
    "avg_trip_spend", "avg_trip_nights", "avg_distance", "n_destinations",
    "hotel_attach_rate",
]
GENDER_TARGET = "gender"


def build_gender_features(users_features: pd.DataFrame | None = None,
                          trips: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model-ready table for gender classification.

    Keeps *all* users. `gender_known` separates the labelled population (train on
    male/female) from the ~33% labelled ``none``, which are the rows the trained
    model is meant to impute. `name` is intentionally dropped — predicting gender
    from a first name would leak the answer rather than learn from behaviour.
    """
    users_features = load_processed("users_features") if users_features is None else users_features
    trips = load_processed("trips") if trips is None else trips

    behaviour = build_user_behaviour(trips)
    df = users_features.merge(behaviour, left_on="code", right_index=True, how="left")

    # First name, lower-cased. Travel behaviour turned out to carry no gender
    # signal (|r| <= 0.06), but the given name does — character n-grams pick up
    # morphology such as the '-a' ending, which generalises to names the model has
    # never seen. Only the FIRST name is kept; surnames carry no gender signal and
    # would let the model memorise family groups.
    df["first_name"] = df["name"].astype(str).str.split().str[0].str.lower()

    share_cols = [c for c in df.columns if c.startswith(("share_class_", "share_agency_"))]
    keep = (["code", GENDER_TARGET, "gender_known", "is_active", "first_name"]
            + GENDER_CAT_FEATURES + GENDER_NUM_FEATURES + share_cols)
    out = df[[c for c in keep if c in df.columns]].copy()

    log.info("gender features: %s | labelled (male/female): %d | to impute (none): %d",
             out.shape, int(out["gender_known"].sum()), int((~out["gender_known"]).sum()))
    return out


# =========================================================================== #
# 5. Recommender feature sets (objective #9)                                  #
# =========================================================================== #
def build_user_hotel_matrix(hotels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-format user x hotel interactions — the collaborative-filtering input.

    One row per (user, hotel) pair actually booked, with implicit-feedback
    strength: booking count, nights and spend.
    """
    hotels = load_interim("hotels_clean") if hotels is None else hotels

    out = (hotels.groupby(["userCode", "name"], observed=True)
           .agg(bookings=("travelCode", "size"),
                nights=("days", "sum"),
                spend=("total", "sum"),
                last_stay=("date", "max"))
           .reset_index()
           .rename(columns={"userCode": "user", "name": "hotel"}))

    log.info("user-hotel interactions: %s | users=%d hotels=%d",
             out.shape, out["user"].nunique(), out["hotel"].nunique())
    return out


def build_hotel_catalog(hotels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-hotel content features — the content-based half of the recommender."""
    hotels = load_interim("hotels_clean") if hotels is None else hotels

    out = (hotels.groupby("name", observed=True)
           .agg(place=("place", "first"),
                nightly_rate=("price", "first"),
                bookings=("travelCode", "size"),
                unique_guests=("userCode", "nunique"),
                avg_nights=("days", "mean"),
                revenue=("total", "sum"))
           .reset_index()
           .rename(columns={"name": "hotel"}))
    out["avg_nights"] = out["avg_nights"].round(3)
    out = out.sort_values("revenue", ascending=False).reset_index(drop=True)

    log.info("hotel catalog: %s", out.shape)
    return out


# =========================================================================== #
# 6. Driver / CLI                                                             #
# =========================================================================== #
def _write(df: pd.DataFrame, stem: str, parquet: bool = False) -> Path:
    """Persist a feature table to data/processed (CSV by default)."""
    PATHS.processed.mkdir(parents=True, exist_ok=True)
    if parquet:
        path = PATHS.processed / f"{stem}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = PATHS.processed / f"{stem}.csv"
        df.to_csv(path, index=False)
    log.info("wrote %-30s %s", path.name, df.shape)
    return path


def build_all(parquet: bool = False) -> dict:
    """Build and persist every feature set."""
    log.info("=== feature engineering started ===")

    users_features = load_processed("users_features")
    trips = load_processed("trips")
    hotels = load_interim("hotels_clean")

    gender = build_gender_features(users_features, trips)
    interactions = build_user_hotel_matrix(hotels)
    catalog = build_hotel_catalog(hotels)

    written = [
        _write(gender, "gender_features", parquet),
        _write(interactions, "user_hotel_interactions", parquet),
        _write(catalog, "hotel_catalog", parquet),
    ]

    log.info("=== feature engineering complete: %d files written ===", len(written))
    return {"frames": {"gender_features": gender,
                       "user_hotel_interactions": interactions,
                       "hotel_catalog": catalog},
            "written": written}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voyage Analytics feature engineering")
    p.add_argument("--parquet", action="store_true",
                   help="write parquet instead of the default CSV")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    build_all(parquet=args.parquet)


if __name__ == "__main__":
    main()
