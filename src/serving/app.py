"""Flask prediction API for Voyage Analytics (project objective #2).

Serves the three registered models over REST::

    GET  /                          service + model inventory
    GET  /health                    liveness / readiness probe
    GET  /reference                 valid cities, classes, agencies, hotels
    POST /predict/flight-price      fare prediction
    POST /predict/gender            gender classification
    POST /recommend/hotels          hotel recommendations

Design notes
------------
**Callers send business inputs, not model features.** ``distance`` is a required
model feature but nobody booking a flight knows a route is 400.5 km, so the API
derives it from the origin/destination pair (distance is fixed per route). The
same applies to the inert calendar fields, which default to today. This keeps the
train/serve contract in one place instead of pushing feature assembly onto every
client.

**Pipelines are served whole.** Each artifact bundles preprocessing *and* the
estimator, so there is no feature engineering duplicated here that could drift
away from training.

**The attach model is deliberately not exposed.** It measured at AUC 0.498 — an
endpoint returning it would look authoritative while being a coin flip.

Run::

    python -m src.serving                    # dev server on :5000
    python -m src.serving --port 8080
    waitress-serve --port=5000 src.serving.app:app   # production
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.utils import PATHS, get_logger

log = get_logger("serving")

MODEL_DIR = PATHS.root / "models"
ROUTE_REFERENCE = MODEL_DIR / "route_reference.json"

ARTIFACTS = {
    "flight_price": MODEL_DIR / "flight_price_model.joblib",
    "gender": MODEL_DIR / "gender_model.joblib",
    "hotel_recommender": MODEL_DIR / "hotel_recommender.joblib",
}


# --------------------------------------------------------------------------- #
# Reference data                                                               #
# --------------------------------------------------------------------------- #
def build_route_reference(save: bool = True) -> dict:
    """Route -> distance lookup, plus the valid category values.

    Derived once from the processed data and cached to JSON so the API does not
    need the full dataset at runtime.
    """
    from src.data_ingestion import load_processed

    df = load_processed("flight_price_model")
    routes = (df.groupby(["from", "to"], observed=True)["distance"]
              .first().reset_index())
    ref = {
        "distances": {f"{r['from']}|{r['to']}": float(r["distance"])
                      for _, r in routes.iterrows()},
        "cities": sorted(set(df["from"].astype(str)) | set(df["to"].astype(str))),
        "flight_types": sorted(df["flightType"].astype(str).unique()),
        "agencies": sorted(df["agency"].astype(str).unique()),
    }
    try:
        catalog = load_processed("hotel_catalog")
        ref["hotels"] = sorted(catalog["hotel"].astype(str).unique())
        ref["hotel_by_city"] = {str(r["place"]): str(r["hotel"])
                                for _, r in catalog.iterrows()}
    except Exception:                                    # catalog is optional
        ref["hotels"], ref["hotel_by_city"] = [], {}

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ROUTE_REFERENCE.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        log.info("wrote route reference -> %s", ROUTE_REFERENCE)
    return ref


@lru_cache(maxsize=1)
def reference() -> dict:
    """Load the cached reference, rebuilding it only if it is missing.

    ``save=False`` on the fallback: the container runs read-only and has no
    ``data/`` directory, so a rebuild there would fail on the write rather than
    on the (more informative) missing-data error. In practice the JSON is baked
    into the image, so this path is only taken during local development.
    """
    if ROUTE_REFERENCE.exists():
        return json.loads(ROUTE_REFERENCE.read_text(encoding="utf-8"))
    log.warning("%s missing — rebuilding from processed data", ROUTE_REFERENCE.name)
    return build_route_reference(save=False)


@lru_cache(maxsize=None)
def load_model(name: str):
    """Load a model artifact once and keep it in memory."""
    path = ARTIFACTS.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Run `python main.py` first.")
    t0 = time.perf_counter()
    model = joblib.load(path)
    log.info("loaded model '%s' from %s (%.0f ms)",
             name, path.name, (time.perf_counter() - t0) * 1000)
    return model


def warmup() -> dict:
    """Load every model and run one throwaway prediction, at startup.

    Without this the *first* request pays the model-load cost — measured at
    ~8.5 s for the gradient-boosting pipeline — which would blow the 500 ms
    latency budget for whichever unlucky user arrives first. Boosted trees also
    allocate internal prediction buffers on first use, so a dummy predict is
    needed as well as the load.
    """
    timings = {}
    reference()                                   # cache the JSON lookup
    for name in ARTIFACTS:
        t0 = time.perf_counter()
        try:
            model = load_model(name)
            if name == "flight_price":
                model.predict(_sample_flight_frame())
            elif name == "gender":
                model.predict(pd.DataFrame([{"first_name": "warmup"}]))
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as exc:                     # noqa: BLE001 - report, don't crash
            log.warning("warmup failed for '%s': %s", name, exc)
            timings[name] = None

    # Import the recommender helper now so the first /recommend request does not
    # pay for pulling in sklearn submodules.
    try:
        from src.models.recommender import recommend  # noqa: F401
    except Exception as exc:                         # noqa: BLE001
        log.warning("could not pre-import recommender: %s", exc)

    log.info("warmup complete: %s", timings)
    return timings


def _sample_flight_frame() -> pd.DataFrame:
    """A single valid row, used for warmup only."""
    ref = reference()
    key = next(iter(ref["distances"]))
    origin, dest = key.split("|")
    today = date.today()
    return pd.DataFrame([{
        "from": origin, "to": dest,
        "flightType": ref["flight_types"][0], "agency": ref["agencies"][0],
        "day_of_week": today.strftime("%A"),
        "distance": ref["distances"][key],
        "year": today.year, "month": today.month,
    }])


# --------------------------------------------------------------------------- #
# Validation helpers                                                           #
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """A client-facing validation error carrying an HTTP status."""

    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra


def _payload() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object.")
    return data


def _require(data: dict, *fields: str) -> None:
    missing = [f for f in fields if data.get(f) in (None, "")]
    if missing:
        raise ApiError(f"Missing required field(s): {missing}",
                       required=list(fields))


def _one_of(value: str, allowed: list[str], field: str) -> str:
    if value not in allowed:
        raise ApiError(f"Invalid {field}: {value!r}",
                       allowed=allowed[:20],
                       hint=f"GET /reference lists all valid {field} values")
    return value


def _lookup_distance(origin: str, dest: str) -> float:
    ref = reference()
    key = f"{origin}|{dest}"
    if key not in ref["distances"]:
        raise ApiError(f"No route from {origin!r} to {dest!r}.",
                       hint="GET /reference lists the served city pairs")
    return ref["distances"][key]


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
def create_app(warm: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    if warm:
        app.config["WARMUP_MS"] = warmup()

    @app.errorhandler(ApiError)
    def _handle_api_error(err: ApiError):
        return jsonify({"error": err.message, **err.extra}), err.status

    @app.errorhandler(Exception)
    def _handle_unexpected(err: Exception):        # noqa: BLE001 - last resort
        log.exception("unhandled error")
        return jsonify({"error": "Internal server error",
                        "detail": str(err)}), 500

    @app.before_request
    def _start_timer():
        request.environ["_t0"] = time.perf_counter()

    @app.after_request
    def _add_latency(response):
        t0 = request.environ.get("_t0")
        if t0 is not None:
            ms = (time.perf_counter() - t0) * 1000
            response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
        return response

    # ---------------------------------------------------------------- meta ---
    @app.get("/")
    def index():
        return jsonify({
            "service": "Voyage Analytics Prediction API",
            "version": "1.0",
            "models": {
                "flight_price": "regression - fare in R$ (R2 0.936 on unseen routes)",
                "gender": "classification - male/female from first name (acc 0.884)",
                "hotel_recommender": "ranking - hotel suggestions (hit@3 0.858)",
            },
            "not_served": {
                "hotel_attach": "measured at AUC 0.498 (chance) - deliberately not exposed",
            },
            "endpoints": ["/health", "/reference", "/predict/flight-price",
                          "/predict/gender", "/recommend/hotels"],
        })

    @app.get("/health")
    def health():
        available = {n: p.exists() for n, p in ARTIFACTS.items()}
        ok = all(available.values())
        cache = load_model.cache_info()
        return jsonify({"status": "healthy" if ok else "degraded",
                        "models_available": available,
                        "models_cached": cache.currsize,
                        "warmup_ms": app.config.get("WARMUP_MS")}), (200 if ok else 503)

    @app.get("/reference")
    def reference_endpoint():
        ref = reference()
        return jsonify({
            "cities": ref["cities"],
            "flight_types": ref["flight_types"],
            "agencies": ref["agencies"],
            "hotels": ref.get("hotels", []),
            "n_routes": len(ref["distances"]),
        })

    # ------------------------------------------------------- flight price ---
    @app.post("/predict/flight-price")
    def predict_flight_price():
        data = _payload()
        _require(data, "from", "to", "flightType", "agency")
        ref = reference()

        origin = _one_of(str(data["from"]), ref["cities"], "city")
        dest = _one_of(str(data["to"]), ref["cities"], "city")
        ftype = _one_of(str(data["flightType"]), ref["flight_types"], "flightType")
        agency = _one_of(str(data["agency"]), ref["agencies"], "agency")
        if origin == dest:
            raise ApiError("Origin and destination must differ.")

        # Derived, not demanded: distance is fixed per route, and the calendar
        # fields were shown to carry no signal, so today's date is a safe default.
        distance = _lookup_distance(origin, dest)
        today = date.today()
        when = pd.to_datetime(data.get("date")) if data.get("date") else pd.Timestamp(today)

        X = pd.DataFrame([{
            "from": origin, "to": dest, "flightType": ftype, "agency": agency,
            "day_of_week": when.day_name(), "distance": distance,
            "year": int(when.year), "month": int(when.month),
        }])
        price = float(load_model("flight_price").predict(X)[0])

        return jsonify({
            "predicted_price": round(price, 2),
            "currency": "BRL",
            "inputs": {"from": origin, "to": dest, "flightType": ftype,
                       "agency": agency, "date": str(when.date())},
            "derived": {"distance_km": distance},
            "model": {"name": "gbr_rate_per_km",
                      "r2_unseen_routes": 0.936,
                      "rmse": 88.99,
                      "note": "Expect roughly +/- R$89 on a route unseen in training."},
        })

    # ------------------------------------------------------------ gender ---
    @app.post("/predict/gender")
    def predict_gender():
        data = _payload()
        _require(data, "name")
        first = str(data["name"]).strip().split()[0].lower()
        if not first:
            raise ApiError("Could not extract a first name from 'name'.")

        model = load_model("gender")
        X = pd.DataFrame([{"first_name": first}])
        pred = model.predict(X)[0]
        label = "male" if int(pred) == 1 else "female"
        try:
            confidence = float(model.predict_proba(X).max())
        except Exception:
            confidence = None

        return jsonify({
            "predicted_gender": label,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "inputs": {"name": data["name"], "first_name_used": first},
            "model": {"name": "name_logistic",
                      "accuracy": 0.884,
                      "accuracy_unseen_names": 0.743,
                      "caveat": ("Infers gender from the given name, not from travel "
                                 "behaviour. Behaviour-only models measured at chance. "
                                 "Reflects the naming conventions of the training data.")},
        })

    # --------------------------------------------------------- recommend ---
    @app.post("/recommend/hotels")
    def recommend_hotels():
        from src.models.recommender import recommend

        data = _payload()
        k = int(data.get("top_k", 3))
        if not 1 <= k <= 9:
            raise ApiError("top_k must be between 1 and 9.")

        artifact = load_model("hotel_recommender")
        destination = data.get("destination")
        if destination:
            destination = _one_of(str(destination), reference()["cities"], "city")

        hotels = recommend(artifact, destination=destination, k=k)
        catalog = artifact.get("catalog", {})
        results = [{"rank": i + 1, "hotel": h,
                    "city": catalog.get(h, {}).get("place"),
                    "nightly_rate": catalog.get(h, {}).get("nightly_rate")}
                   for i, h in enumerate(hotels)]

        return jsonify({
            "recommendations": results,
            "strategy": "destination-aware" if destination else "popularity",
            "inputs": {"destination": destination, "top_k": k},
            "note": ("Each city has exactly one hotel, so when a destination is "
                     "supplied the match is exact and no model is needed. Ranking "
                     "matters only when the destination is unknown."),
        })

    return app


app = create_app()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voyage Analytics prediction API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--build-reference", action="store_true",
                   help="rebuild models/route_reference.json and exit")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.build_reference:
        build_route_reference()
        return
    log.info("starting API on http://%s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
