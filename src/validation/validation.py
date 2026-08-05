"""Data quality validation for Voyage Analytics.

A standalone pipeline stage — deliberately independent of ingestion, so the same
checks can gate raw input *before* cleaning and audit the generated tables
*after* it. This is the "Data Validation" box in the system design (the role
Great Expectations fills in the target architecture).

Two scopes:

``raw``
    Gates ``data/raw/*.csv`` before anything is cleaned. Schema and numeric
    bounds are **errors** (they stop the pipeline); nulls, duplicates and unseen
    category values are **warnings**.

``processed``
    Audits what the ingestion and feature stages wrote to ``data/interim`` and
    ``data/processed`` — files present, key columns intact, referential
    integrity across tables, and the business invariants EDA established.

Run::

    python -m src.validation                    # raw (the pipeline gate)
    python -m src.validation --scope processed  # audit generated tables
    python -m src.validation --scope all
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import pandas as pd

from src.utils import CONFIG, PATHS, get_logger

log = get_logger("validation")

REPORT_PATH = PATHS.reports / "data_quality_report.json"


# =========================================================================== #
# Report                                                                      #
# =========================================================================== #
@dataclass
class DataQualityReport:
    """Collects check results; ``ok`` is False if any error was recorded."""

    scope: str = "raw"
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    passed: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        log.error(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning(msg)

    def ok_check(self, msg: str) -> None:
        self.passed.append(msg)

    def summary(self) -> str:
        return (f"validation[{self.scope}]: {'PASS' if self.ok else 'FAIL'} | "
                f"{len(self.passed)} passed, {len(self.errors)} error(s), "
                f"{len(self.warnings)} warning(s)")

    def as_dict(self) -> dict:
        return {"scope": self.scope, "ok": self.ok, "stats": self.stats,
                "errors": self.errors, "warnings": self.warnings,
                "passed": self.passed}

    def to_frame(self) -> pd.DataFrame:
        rows = ([{"severity": "error", "message": m} for m in self.errors]
                + [{"severity": "warning", "message": m} for m in self.warnings]
                + [{"severity": "passed", "message": m} for m in self.passed])
        return pd.DataFrame(rows)


# =========================================================================== #
# Raw checks                                                                  #
# =========================================================================== #
def _check_schema(name: str, df: pd.DataFrame, report: DataQualityReport) -> None:
    expected = CONFIG.schema[name]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        report.error(f"[{name}] missing columns: {missing}")
    else:
        report.ok_check(f"[{name}] schema: all {len(expected)} expected columns present")
    if extra:
        report.warn(f"[{name}] unexpected extra columns: {extra}")


def _check_nulls(name: str, df: pd.DataFrame, report: DataQualityReport) -> None:
    nulls = df.isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        report.warn(f"[{name}] null values present: {bad.to_dict()}")
    else:
        report.ok_check(f"[{name}] nulls: none")


def _check_duplicates(name: str, df: pd.DataFrame, report: DataQualityReport) -> None:
    dupes = int(df.duplicated().sum())
    if dupes:
        report.warn(f"[{name}] {dupes} full-row duplicate(s)")
    else:
        report.ok_check(f"[{name}] duplicates: none")


def _check_domains(name: str, df: pd.DataFrame, report: DataQualityReport) -> None:
    domains = {
        ("flights", "flightType"): CONFIG.flight_types,
        ("flights", "agency"): CONFIG.agencies,
        ("users", "gender"): CONFIG.genders,
    }
    for (n, col), allowed in domains.items():
        if n == name and col in df.columns:
            unseen = set(df[col].unique()) - set(allowed)
            if unseen:
                report.warn(f"[{name}] '{col}' has values outside known domain: {unseen}")
            else:
                report.ok_check(f"[{name}] '{col}' domain: within {list(allowed)}")


def _check_bounds(name: str, df: pd.DataFrame, report: DataQualityReport) -> None:
    for key, (lo, hi) in CONFIG.bounds.items():
        n, col = key.split(".")
        if n == name and col in df.columns:
            out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if out:
                report.error(f"[{name}] '{col}' has {out} value(s) outside [{lo}, {hi}]")
            else:
                report.ok_check(f"[{name}] '{col}' within [{lo}, {hi}]")


def _check_referential(frames: dict[str, pd.DataFrame], report: DataQualityReport) -> None:
    if not all(k in frames for k in ("flights", "hotels", "users")):
        return
    user_ids = set(frames["users"]["code"])
    for name, col in (("flights", "userCode"), ("hotels", "userCode")):
        orphans = set(frames[name][col]) - user_ids
        if orphans:
            report.error(f"[{name}] {len(orphans)} {col}(s) not present in users")
        else:
            report.ok_check(f"[{name}] referential: all {col}s exist in users")

    trip_ids = set(frames["flights"]["travelCode"])
    orphan_trips = set(frames["hotels"]["travelCode"]) - trip_ids
    if orphan_trips:
        report.error(f"[hotels] {len(orphan_trips)} travelCode(s) with no matching flight")
    else:
        report.ok_check("[hotels] referential: every travelCode has flights")


def load_raw_frames() -> dict[str, pd.DataFrame]:
    """Read the raw CSVs for validation, independently of the ingestion module."""
    frames = {}
    for name, fname in CONFIG.raw_files.items():
        path = PATHS.raw / fname
        if not path.exists():
            raise FileNotFoundError(f"Raw file missing: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def validate_raw(frames: dict[str, pd.DataFrame] | None = None) -> DataQualityReport:
    """Validate the raw source tables. Errors here should stop the pipeline."""
    frames = load_raw_frames() if frames is None else frames
    report = DataQualityReport(scope="raw")

    for name in ("flights", "hotels", "users"):
        df = frames.get(name)
        if df is None:
            report.error(f"[{name}] frame not loaded")
            continue
        report.stats[name] = {"rows": len(df), "cols": df.shape[1]}
        _check_schema(name, df, report)
        _check_nulls(name, df, report)
        _check_duplicates(name, df, report)
        _check_domains(name, df, report)
        _check_bounds(name, df, report)

    _check_referential(frames, report)
    log.info(report.summary())
    return report


# =========================================================================== #
# Processed checks                                                            #
# =========================================================================== #
# Tables the pipeline is expected to produce, with the columns that must survive.
EXPECTED_TABLES: dict[str, dict] = {
    "interim": {
        "flights_clean": ["travelCode", "userCode", "price", "date", "leg", "route"],
        "hotels_clean": ["travelCode", "userCode", "name", "place", "days", "total"],
        "users_clean": ["code", "gender", "age", "gender_known"],
    },
    "processed": {
        "trips": ["travelCode", "user", "origin", "dest", "trip_nights", "trip_spend"],
        "users_features": ["code", "gender", "age", "n_trips", "recency_days"],
        "flight_price_model": ["from", "to", "flightType", "agency", "distance", "price"],
        "gender_features": ["code", "gender", "gender_known", "age"],
        "user_hotel_interactions": ["user", "hotel", "bookings", "nights"],
        "hotel_catalog": ["hotel", "place", "nightly_rate", "revenue"],
    },
}


def _table_path(folder, stem: str):
    for ext in (".csv", ".parquet"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read_table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate_processed() -> DataQualityReport:
    """Audit the tables written by the ingestion and feature stages."""
    report = DataQualityReport(scope="processed")
    loaded: dict[str, pd.DataFrame] = {}

    for group, tables in EXPECTED_TABLES.items():
        folder = PATHS.interim if group == "interim" else PATHS.processed
        for stem, key_cols in tables.items():
            path = _table_path(folder, stem)
            if path is None:
                report.error(f"[{group}/{stem}] file missing — run the pipeline first")
                continue

            df = _read_table(path)
            loaded[stem] = df
            report.stats[stem] = {"rows": len(df), "cols": df.shape[1]}

            if df.empty:
                report.error(f"[{stem}] table is empty")
                continue
            report.ok_check(f"[{stem}] present, {len(df):,} rows")

            missing = [c for c in key_cols if c not in df.columns]
            if missing:
                report.error(f"[{stem}] missing key columns: {missing}")
            else:
                report.ok_check(f"[{stem}] key columns intact")

    _check_business_invariants(loaded, report)
    log.info(report.summary())
    return report


def _check_business_invariants(t: dict[str, pd.DataFrame], report: DataQualityReport) -> None:
    """Assert the structural facts EDA established still hold after processing."""
    # Every trip is a two-leg round trip
    if "flights_clean" in t:
        legs = t["flights_clean"].groupby("travelCode").size()
        bad = int((legs != CONFIG.n_legs_per_trip).sum())
        if bad:
            report.error(f"[flights_clean] {bad} travelCode(s) without exactly 2 legs")
        else:
            report.ok_check("[flights_clean] every trip has exactly 2 legs")

    # Hotel total must equal nightly rate x nights
    if "hotels_clean" in t:
        h = t["hotels_clean"]
        bad = int(((h["price"] * h["days"] - h["total"]).abs() > 0.01).sum())
        if bad:
            report.error(f"[hotels_clean] {bad} row(s) where total != price * days")
        else:
            report.ok_check("[hotels_clean] total == price * days")

    # Trip length must sit in the observed 1-4 night range
    if "trips" in t:
        tr = t["trips"]
        bad = int((~tr["trip_nights"].between(1, 4)).sum())
        if bad:
            report.warn(f"[trips] {bad} trip(s) outside the 1-4 night range")
        else:
            report.ok_check("[trips] trip_nights within 1-4")

        neg = int((tr["trip_spend"] < 0).sum())
        if neg:
            report.error(f"[trips] {neg} trip(s) with negative spend")
        else:
            report.ok_check("[trips] no negative spend")

    # Model table must carry a usable target
    if "flight_price_model" in t:
        p = t["flight_price_model"]
        if p["price"].isna().any() or (p["price"] <= 0).any():
            report.error("[flight_price_model] target 'price' has null or non-positive values")
        else:
            report.ok_check("[flight_price_model] target 'price' clean and positive")

    # Referential integrity between processed tables
    if "trips" in t and "users_features" in t:
        orphans = set(t["trips"]["user"]) - set(t["users_features"]["code"])
        if orphans:
            report.error(f"[trips] {len(orphans)} user(s) missing from users_features")
        else:
            report.ok_check("[trips] every user exists in users_features")

    # The gender split the classifier depends on
    if "gender_features" in t:
        g = t["gender_features"]
        labelled = int(g["gender_known"].sum())
        if labelled == 0:
            report.error("[gender_features] no labelled rows to train on")
        else:
            report.ok_check(f"[gender_features] {labelled} labelled, "
                            f"{len(g) - labelled} to impute")


# =========================================================================== #
# Driver / CLI                                                                #
# =========================================================================== #
def run_validation(scope: str = "raw", strict: bool = True,
                   save: bool = True) -> dict[str, DataQualityReport]:
    """Run the requested scope(s) and optionally persist the report.

    With ``strict`` (the default) any error raises, so this works as a pipeline
    gate that stops bad data before it reaches the next stage.
    """
    log.info("=== data validation started (scope=%s) ===", scope)
    reports: dict[str, DataQualityReport] = {}

    if scope in ("raw", "all"):
        reports["raw"] = validate_raw()
    if scope in ("processed", "all"):
        reports["processed"] = validate_processed()
    if not reports:
        raise ValueError(f"unknown scope '{scope}' (use raw, processed or all)")

    if save:
        PATHS.reports.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps({k: r.as_dict() for k, r in reports.items()}, indent=2),
            encoding="utf-8")
        log.info("saved report -> %s", REPORT_PATH)

    failed = {k: r for k, r in reports.items() if not r.ok}
    log.info("=== data validation complete ===")
    if failed and strict:
        details = {k: r.errors for k, r in failed.items()}
        raise ValueError(f"Data validation failed: {details}")
    return reports


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voyage Analytics data validation")
    p.add_argument("--scope", choices=["raw", "processed", "all"], default="raw",
                   help="what to validate (default: raw)")
    p.add_argument("--no-strict", action="store_true",
                   help="report problems without raising")
    p.add_argument("--no-save", action="store_true",
                   help="do not write reports/data_quality_report.json")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run_validation(scope=args.scope, strict=not args.no_strict, save=not args.no_save)


if __name__ == "__main__":
    main()
