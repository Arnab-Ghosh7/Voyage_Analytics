"""Central configuration: project paths and dataset schema/constants.

Everything else in the codebase imports paths and constants from here so that
notebooks, scripts, the ingestion pipeline and tests all agree on one layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _find_project_root(marker: str = "requirements.txt") -> Path:
    """Walk upward from this file until the repo root (holding `marker`) is found.

    Falls back to two levels up (`src/utils/config.py` -> repo root) so the module
    still works if the marker is ever moved.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / marker).exists():
            return parent
    return here.parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    external: Path
    reports: Path
    figures: Path

    def ensure(self) -> "Paths":
        """Create the writable data/report directories if they do not yet exist."""
        for p in (self.interim, self.processed, self.external, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


_ROOT = _find_project_root()
PATHS = Paths(
    root=_ROOT,
    data=_ROOT / "data",
    raw=_ROOT / "data" / "raw",
    interim=_ROOT / "data" / "interim",
    processed=_ROOT / "data" / "processed",
    external=_ROOT / "data" / "external",
    reports=_ROOT / "reports",
    figures=_ROOT / "reports" / "figures",
)


@dataclass(frozen=True)
class Config:
    """Dataset-level constants derived and validated during EDA."""

    date_format: str = "%m/%d/%Y"

    # Raw source files (relative to PATHS.raw)
    raw_files: dict = field(default_factory=lambda: {
        "flights": "flights.csv",
        "hotels": "hotels.csv",
        "users": "users.csv",
    })

    # Expected columns per raw table — used by schema validation
    schema: dict = field(default_factory=lambda: {
        "flights": ["travelCode", "userCode", "from", "to", "flightType",
                    "price", "time", "distance", "agency", "date"],
        "hotels": ["travelCode", "userCode", "name", "place", "days",
                   "price", "total", "date"],
        "users": ["code", "company", "name", "gender", "age"],
    })

    # Known categorical domains (from EDA) — validation warns on unseen values
    flight_types: tuple = ("economic", "premium", "firstClass")
    agencies: tuple = ("CloudFy", "Rainbow", "FlyingDrops")
    genders: tuple = ("male", "female", "none")
    missing_gender_label: str = "none"

    # Date columns per persisted table — lets CSV readers restore datetime types
    # (CSV stores everything as text, so dates must be re-parsed on load).
    date_columns: dict = field(default_factory=lambda: {
        "flights_clean": ["date"],
        "hotels_clean": ["date"],
        "users_clean": [],
        "trips": ["depart", "return_date"],
        "users_features": ["last_trip", "first_trip"],
        "flight_price_model": [],
        # engineered feature sets (src/features)
        "gender_features": [],
        "user_hotel_interactions": ["last_stay"],
        "hotel_catalog": [],
    })

    # Categorical columns per persisted table — re-applied as 'category' on load
    # to keep memory small after a CSV round-trip.
    category_columns: dict = field(default_factory=lambda: {
        "flights_clean": ["from", "to", "flightType", "agency", "leg",
                          "route", "day_of_week"],
        "hotels_clean": ["name", "place"],
        "users_clean": ["company", "gender"],
        "trips": ["origin", "dest", "flightType", "agency"],
        "users_features": ["company", "gender", "home_city", "age_band"],
        "flight_price_model": ["from", "to", "route", "flightType",
                               "agency", "day_of_week"],
        # engineered feature sets (src/features)
        # first_name is left as plain text (not category) — the char n-gram
        # vectoriser consumes it as strings.
        "gender_features": ["gender", "company", "home_city"],
        "user_hotel_interactions": ["hotel"],
        "hotel_catalog": ["hotel", "place"],
    })

    # Numeric sanity bounds (from EDA describe()) — validation flags violations
    bounds: dict = field(default_factory=lambda: {
        "flights.price": (0, 5000),
        "flights.time": (0, 10),
        "flights.distance": (0, 3000),
        "hotels.days": (1, 30),
        "hotels.price": (0, 2000),
        "hotels.total": (0, 20000),
        "users.age": (0, 120),
    })

    n_legs_per_trip: int = 2


CONFIG = Config()
