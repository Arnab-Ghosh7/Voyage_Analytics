"""Data ingestion for Voyage Analytics — cleaning, feature build, IO.

A single module that takes the raw travel CSVs through::

    raw CSV  ->  clean (data/interim)  ->  entity tables (data/processed)

**Data quality checks live in** :mod:`src.validation`, which runs as its own
pipeline stage before this one. Ingestion assumes its input has already been
gated and concerns itself only with transformation.

Run as a script::

    python -m src.data_ingestion              # CSV outputs
    python -m src.data_ingestion --parquet    # parquet instead of CSV

Or import the pieces (re-exported by the package ``__init__``)::

    from src.data_ingestion import run_pipeline, load_processed
    run_pipeline()                            # writes data/interim + data/processed
    price = load_processed("flight_price_model")   # typed reload (dates/categories)

Sections below: (1) IO, (2) cleaning, (3) feature build,
(4) typed readers, (5) pipeline driver / CLI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import CONFIG, PATHS, get_logger

log = get_logger("ingest")


# =========================================================================== #
# 1. IO                                                                       #
# =========================================================================== #
def load_raw() -> dict[str, pd.DataFrame]:
    """Read the three raw CSVs into a dict of DataFrames."""
    frames = {}
    for name, fname in CONFIG.raw_files.items():
        path = PATHS.raw / fname
        if not path.exists():
            raise FileNotFoundError(f"Raw file missing: {path}")
        frames[name] = pd.read_csv(path)
        log.info("loaded %-8s %s from %s", name, frames[name].shape, path.name)
    return frames


def _write(df: pd.DataFrame, folder: Path, stem: str, parquet: bool = False) -> Path:
    """Persist a frame as CSV (default) or parquet and return the path.

    CSV is the project's chosen format for interim/processed data (human-readable,
    Excel-friendly); dtypes are restored on load via the typed readers below.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if parquet:
        path = folder / f"{stem}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = folder / f"{stem}.csv"
        df.to_csv(path, index=False)
    log.info("wrote %-26s %s", path.name, df.shape)
    return path


# =========================================================================== #
# 2. Cleaning  ->  data/interim                                               #
# =========================================================================== #
def clean_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, tag legs, add calendar + route + unit-price features.

    Encodes the EDA finding that each `travelCode` is a two-leg round trip:
    the earlier leg is `outbound`, the later `return`.
    """
    df = flights.copy()
    df["date"] = pd.to_datetime(df["date"], format=CONFIG.date_format)
    df = df.sort_values(["travelCode", "date"]).reset_index(drop=True)

    df["leg"] = np.where(df.groupby("travelCode").cumcount() == 0, "outbound", "return")
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day_of_week"] = df["date"].dt.day_name()
    df["route"] = df["from"] + " -> " + df["to"]
    df["price_per_km"] = (df["price"] / df["distance"]).round(4)

    # Compact categoricals to shrink the 271k-row frame in memory
    for col in ["from", "to", "flightType", "agency", "leg", "route", "day_of_week"]:
        df[col] = df[col].astype("category")
    return df


def clean_hotels(hotels: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and validate the `total == price * days` identity."""
    df = hotels.copy()
    df["date"] = pd.to_datetime(df["date"], format=CONFIG.date_format)
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")

    recomputed = (df["price"] * df["days"]).round(2)
    mismatch = int((recomputed - df["total"]).abs().gt(0.01).sum())
    if mismatch:
        log.warning("hotels: %d rows where total != price*days", mismatch)

    for col in ["name", "place"]:
        df[col] = df[col].astype("category")
    return df


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Normalise gender and add a `gender_known` flag for the classifier phase."""
    df = users.copy()
    df["gender"] = df["gender"].str.strip().str.lower()
    df["gender_known"] = df["gender"] != CONFIG.missing_gender_label
    for col in ["company", "gender"]:
        df[col] = df[col].astype("category")
    return df


# =========================================================================== #
# 3. Feature build  ->  data/processed                                        #
# =========================================================================== #
def build_trips(flights_clean: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two legs of each trip into one trip-level row."""
    df = flights_clean.sort_values("date")
    trips = (df.groupby("travelCode", observed=True)
             .agg(user=("userCode", "first"),
                  origin=("from", "first"),
                  dest=("to", "first"),
                  flightType=("flightType", "first"),
                  agency=("agency", "first"),
                  depart=("date", "first"),
                  return_date=("date", "last"),
                  distance=("distance", "first"),
                  flight_cost=("price", "sum"),
                  year=("year", "first"))
             .reset_index())
    trips["trip_nights"] = (trips["return_date"] - trips["depart"]).dt.days.astype("int16")
    return trips


def _attach_hotels(trips: pd.DataFrame, hotels_clean: pd.DataFrame) -> pd.DataFrame:
    """Add hotel cost / attach flag per trip (~30% of trips include a stay)."""
    hotel_cost = hotels_clean.groupby("travelCode")["total"].sum().rename("hotel_cost")
    trips = trips.merge(hotel_cost, on="travelCode", how="left")
    trips["has_hotel"] = trips["hotel_cost"].notna()
    trips["hotel_cost"] = trips["hotel_cost"].fillna(0.0)
    trips["trip_spend"] = trips["flight_cost"] + trips["hotel_cost"]
    return trips


def build_user_features(users_clean: pd.DataFrame,
                        trips: pd.DataFrame,
                        flights_clean: pd.DataFrame) -> pd.DataFrame:
    """User-level RFM + behaviour table joined to demographics.

    `home_city` = each user's most frequent outbound origin (EDA showed it is
    effectively fixed per user and per company).
    """
    per_user = (trips.groupby("user", observed=True)
                .agg(n_trips=("travelCode", "size"),
                     flight_spend=("flight_cost", "sum"),
                     hotel_spend=("hotel_cost", "sum"),
                     total_spend=("trip_spend", "sum"),
                     hotels_booked=("has_hotel", "sum"),
                     last_trip=("depart", "max"),
                     first_trip=("depart", "min"))
                .reset_index())

    home = (flights_clean[flights_clean["leg"] == "outbound"]
            .groupby("userCode", observed=True)["from"]
            .agg(lambda s: s.mode().iloc[0])
            .rename("home_city"))

    udf = (users_clean.merge(per_user, left_on="code", right_on="user", how="left")
                      .merge(home, left_on="code", right_index=True, how="left"))

    snapshot = flights_clean["date"].max()
    udf["recency_days"] = (snapshot - udf["last_trip"]).dt.days
    udf["hotel_attach_rate"] = (udf["hotels_booked"] / udf["n_trips"]).round(4)
    udf["age_band"] = pd.cut(udf["age"], [20, 30, 40, 50, 60, 66],
                             labels=["21-30", "31-40", "41-50", "51-60", "61-65"])
    udf["is_active"] = udf["n_trips"].notna()
    return udf


def build_flight_price_table(flights_clean: pd.DataFrame) -> pd.DataFrame:
    """Modelling-ready table for the flight-price regressor.

    Drops `time` (r=0.99999 with `distance`) to remove the redundant collinear
    feature, keeping the compact, leakage-free feature set EDA identified.
    """
    cols = ["from", "to", "route", "flightType", "agency",
            "distance", "year", "month", "day_of_week", "price"]
    return flights_clean[cols].copy()


# =========================================================================== #
# 4. Typed readers (restore dtypes after a CSV round-trip)                     #
# =========================================================================== #
_INTERIM = {"flights_clean", "hotels_clean", "users_clean"}
_PROCESSED = {"trips", "users_features", "flight_price_model",
              # engineered feature sets written by src.features
              "gender_features", "user_hotel_interactions", "hotel_catalog"}


def _read_typed(folder: Path, name: str) -> pd.DataFrame:
    """Read a persisted table as CSV, restoring dates/categoricals from CONFIG."""
    path = folder / f"{name}.csv"
    if not path.exists():  # fall back to parquet if that is what was written
        pq = folder / f"{name}.parquet"
        if pq.exists():
            return pd.read_parquet(pq)
        raise FileNotFoundError(f"No CSV/parquet for '{name}' in {folder}")

    date_cols = CONFIG.date_columns.get(name, [])
    df = pd.read_csv(path, parse_dates=date_cols or None)

    for col in CONFIG.category_columns.get(name, []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    log.info("loaded %-20s %s from %s", name, df.shape, path.name)
    return df


def load_interim(name: str) -> pd.DataFrame:
    """Load a cleaned interim table by stem, e.g. 'flights_clean'."""
    if name not in _INTERIM:
        raise ValueError(f"Unknown interim table '{name}'. Choose from {_INTERIM}.")
    return _read_typed(PATHS.interim, name)


def load_processed(name: str) -> pd.DataFrame:
    """Load a processed feature table by stem, e.g. 'flight_price_model'."""
    if name not in _PROCESSED:
        raise ValueError(f"Unknown processed table '{name}'. Choose from {_PROCESSED}.")
    return _read_typed(PATHS.processed, name)


def load_all_processed() -> dict[str, pd.DataFrame]:
    """Convenience: load every processed table into a dict."""
    return {name: load_processed(name) for name in sorted(_PROCESSED)}


# =========================================================================== #
# 5. Pipeline driver / CLI                                                     #
# =========================================================================== #
def run_pipeline(parquet: bool = False) -> dict:
    """Execute the full ingestion pipeline and persist all outputs.

    Returns a dict with the in-memory frames and the list of written paths, so a
    notebook can inspect results without re-reading them.

    Data quality is *not* checked here — :mod:`src.validation` runs as its own
    stage before this one.
    """
    PATHS.ensure()
    log.info("=== Voyage Analytics ingestion started ===")

    raw = load_raw()

    # ---- clean -> data/interim
    flights_c = clean_flights(raw["flights"])
    hotels_c = clean_hotels(raw["hotels"])
    users_c = clean_users(raw["users"])

    written = []
    written.append(_write(flights_c, PATHS.interim, "flights_clean", parquet))
    written.append(_write(hotels_c, PATHS.interim, "hotels_clean", parquet))
    written.append(_write(users_c, PATHS.interim, "users_clean", parquet))

    # ---- feature tables -> data/processed
    trips = _attach_hotels(build_trips(flights_c), hotels_c)
    udf = build_user_features(users_c, trips, flights_c)
    price_tbl = build_flight_price_table(flights_c)

    written.append(_write(trips, PATHS.processed, "trips", parquet))
    written.append(_write(udf, PATHS.processed, "users_features", parquet))
    written.append(_write(price_tbl, PATHS.processed, "flight_price_model", parquet))

    log.info("=== ingestion complete: %d files written ===", len(written))
    return {
        "frames": {"flights": flights_c, "hotels": hotels_c, "users": users_c,
                   "trips": trips, "users_features": udf,
                   "flight_price_model": price_tbl},
        "written": written,
    }


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voyage Analytics data ingestion")
    p.add_argument("--parquet", action="store_true",
                   help="write parquet instead of the default CSV")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run_pipeline(parquet=args.parquet)


if __name__ == "__main__":
    main()
