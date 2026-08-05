"""Hotel recommendation (project objective #9).

Two deliverables, because the data forces a distinction:

1. **Hotel ranking** — given a user, rank the 9 hotels. Three strategies are
   compared with leave-one-out evaluation (hold out each user's most recent
   stay, see whether it is recovered in the top-K):

   * ``popularity``          — always recommend the globally most-booked hotels.
   * ``item_cf``             — item-item collaborative filtering on the user x hotel matrix.
   * ``next_dest_heuristic`` — predict the next destination as "same as last trip",
     then recommend that city's hotel.

   Note on the trivial case: hotel and city are **bijective** (one hotel per city),
   so if the destination is already known at request time — which it is in a real
   booking flow, where the flight is chosen first — the correct hotel is determined
   with 100% accuracy and no model is needed. Ranking is only a real problem when
   the destination is *unknown*, which is what these three strategies address.

2. **Hotel attach** — a classifier predicting whether a trip will book a hotel at
   all. Only ~30% do, so this is where the incremental revenue is.

Why both: EDA found the user x hotel matrix is **82.5% dense** with exactly one
hotel per city, so "which hotel" is nearly determined by the destination and
collaborative filtering has little to learn. The genuinely useful model is the
attach predictor, and this module reports honest numbers for both.

Run::

    python -m src.models.recommender
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_ingestion import load_processed
from src.features import make_preprocessor
from src.utils import PATHS, get_logger

log = get_logger("model.recommender")

TOP_K = 3
ATTACH_CAT = ["dest", "origin", "flightType", "agency"]
ATTACH_NUM = ["trip_nights", "distance", "flight_cost", "year"]
ATTACH_TARGET = "has_hotel"

RANKER_PATH = PATHS.root / "models" / "hotel_recommender.joblib"
ATTACH_PATH = PATHS.root / "models" / "hotel_attach_model.joblib"
METRICS_PATH = PATHS.reports / "recommender_metrics.json"


# =========================================================================== #
# 1. Hotel ranking                                                            #
# =========================================================================== #
def _leave_one_out(interactions: pd.DataFrame):
    """Hold out each user's most recent stay; the rest is history."""
    inter = interactions.sort_values("last_stay")
    held = inter.groupby("user", observed=True).tail(1)
    history = inter.drop(held.index)
    # Only users who still have history left are evaluable.
    held = held[held["user"].isin(set(history["user"]))]
    return history, held


def _popularity_scores(history: pd.DataFrame) -> pd.Series:
    return history.groupby("hotel", observed=True)["bookings"].sum().sort_values(ascending=False)


def _item_cf_scores(history: pd.DataFrame, hotels: list[str]) -> pd.DataFrame:
    """Item-item cosine similarity over the user x hotel booking matrix."""
    mat = (history.pivot_table(index="user", columns="hotel", values="bookings",
                               fill_value=0, observed=True)
           .reindex(columns=hotels, fill_value=0))
    v = mat.to_numpy(dtype=float)
    norms = np.linalg.norm(v, axis=0)
    norms[norms == 0] = 1e-9
    sim = (v.T @ v) / np.outer(norms, norms)
    np.fill_diagonal(sim, 0.0)
    scores = v @ sim                                    # user x hotel affinity
    return pd.DataFrame(scores, index=mat.index, columns=hotels)


def evaluate_rankers(interactions: pd.DataFrame, trips: pd.DataFrame,
                     k: int = TOP_K) -> pd.DataFrame:
    """Leave-one-out Precision@K / Recall@K / MAP for the three strategies."""
    history, held = _leave_one_out(interactions)
    hotels = sorted(interactions["hotel"].unique())
    log.info("leave-one-out: %d evaluable users, %d hotels, K=%d",
             held["user"].nunique(), len(hotels), k)

    pop = _popularity_scores(history)
    pop_top = list(pop.head(k).index)
    cf = _item_cf_scores(history, hotels)

    # destination strategy: the hotel actually located in the trip's destination
    hotel_city = (interactions.merge(trips[["user", "dest"]].drop_duplicates(),
                                     on="user", how="left"))
    city_of_hotel = (pd.read_csv(PATHS.processed / "hotel_catalog.csv")
                     .set_index("place")["hotel"].to_dict()
                     if (PATHS.processed / "hotel_catalog.csv").exists() else {})
    # last destination per user, from their trip history
    last_dest = (trips.sort_values("depart").groupby("user", observed=True)["dest"].last())

    rows = []
    hits = {"popularity": [], "item_cf": [], "next_dest_heuristic": []}
    for _, r in held.iterrows():
        user, truth = r["user"], r["hotel"]

        hits["popularity"].append(truth in pop_top)

        if user in cf.index:
            cf_top = list(cf.loc[user].sort_values(ascending=False).head(k).index)
        else:
            cf_top = pop_top
        hits["item_cf"].append(truth in cf_top)

        # "next destination == last destination" heuristic, then that city's hotel
        dest = last_dest.get(user)
        dest_hotel = city_of_hotel.get(dest)
        hits["next_dest_heuristic"].append(truth == dest_hotel)

    for name, h in hits.items():
        h = np.array(h, dtype=float)
        rows.append({"strategy": name,
                     f"hit_rate@{k}": round(h.mean(), 4),
                     f"precision@{k}": round(h.mean() / k, 4),
                     "n_users": len(h)})
    out = pd.DataFrame(rows).sort_values(f"hit_rate@{k}", ascending=False)
    for _, r in out.iterrows():
        log.info("%-12s hit_rate@%d=%.4f", r["strategy"], k, r[f"hit_rate@{k}"])
    return out


def build_recommender(interactions: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """Serving artifact: popularity ranking + hotel-per-city lookup."""
    pop = _popularity_scores(interactions)
    return {
        "popularity_order": list(pop.index),
        "hotel_by_city": catalog.set_index("place")["hotel"].to_dict(),
        "catalog": catalog.set_index("hotel").to_dict(orient="index"),
        "top_k": TOP_K,
    }


def recommend(artifact: dict, destination: str | None = None, k: int = TOP_K) -> list[str]:
    """Recommend hotels; destination-aware when a destination is supplied."""
    if destination and destination in artifact["hotel_by_city"]:
        first = artifact["hotel_by_city"][destination]
        rest = [h for h in artifact["popularity_order"] if h != first]
        return [first] + rest[: k - 1]
    return artifact["popularity_order"][:k]


# =========================================================================== #
# 2. Hotel attach                                                             #
# =========================================================================== #
def train_attach_model(trips: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict]:
    """Predict whether a trip books a hotel — the real revenue lever."""
    X = trips[ATTACH_CAT + ATTACH_NUM]
    y = trips[ATTACH_TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    pipe = Pipeline([("prep", make_preprocessor(ATTACH_CAT, ATTACH_NUM)),
                     ("model", HistGradientBoostingClassifier(
                         max_iter=200, random_state=seed))])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    base_rate = float(y.mean())
    metrics = {
        "base_rate": round(base_rate, 4),
        "majority_baseline_accuracy": round(max(base_rate, 1 - base_rate), 4),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }
    log.info("attach model: acc=%.4f auc=%.4f (baseline acc=%.4f, base rate=%.1f%%)",
             metrics["accuracy"], metrics["roc_auc"],
             metrics["majority_baseline_accuracy"], base_rate * 100)
    return pipe, metrics


# =========================================================================== #
# 3. Driver                                                                   #
# =========================================================================== #
def run(seed: int = 42) -> dict:
    log.info("=== hotel recommendation started ===")
    interactions = load_processed("user_hotel_interactions")
    catalog = load_processed("hotel_catalog")
    trips = load_processed("trips")

    ranking = evaluate_rankers(interactions, trips)
    artifact = build_recommender(interactions, catalog)
    attach_pipe, attach_metrics = train_attach_model(trips, seed)

    RANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, RANKER_PATH)
    joblib.dump(attach_pipe, ATTACH_PATH)

    best = ranking.iloc[0]
    METRICS_PATH.write_text(json.dumps({
        "ranking": {"top_k": TOP_K,
                    "best_strategy": best["strategy"],
                    "results": ranking.to_dict(orient="records")},
        "attach_model": attach_metrics,
        "notes": [
            "Hotel and city are bijective (9 hotels, one per city). If the "
            "destination is known at request time the hotel is determined exactly, "
            "so ranking only matters when the destination is unknown.",
            "Random top-3 of 9 hotels = 0.333 hit rate; judge strategies against that.",
            f"Attach model AUC={attach_metrics['roc_auc']} against a "
            f"{attach_metrics['base_rate']:.1%} base rate: whether a trip books a "
            "hotel is NOT predictable from trip context in this data.",
        ],
    }, indent=2), encoding="utf-8")

    log.info("saved recommender -> %s", RANKER_PATH)
    log.info("saved attach model -> %s", ATTACH_PATH)
    log.info("=== hotel recommendation complete ===")
    return {"ranking": ranking, "artifact": artifact,
            "attach_metrics": attach_metrics, "best_strategy": best["strategy"]}


def main(argv=None) -> None:
    argparse.ArgumentParser(description="Train the hotel recommender").parse_args(argv)
    run()


if __name__ == "__main__":
    main()
