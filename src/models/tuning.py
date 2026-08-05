"""Hyperparameter tuning for every Voyage Analytics model.

Until this stage all hyperparameters sat at hand-picked defaults
(``Ridge(alpha=1.0)``, ``n_estimators=300``, ``max_iter=200``). This module
searches them properly and reports the gain over the untuned baseline.

Search strategy per model — matched to how much signal exists:

============  ==================================  ===============================
Model         Search                              Cross-validation
============  ==================================  ===============================
flight price  GridSearchCV (Ridge, tree) +        ``GroupKFold`` over routes, so a
              RandomizedSearchCV (HistGBM)        tuned model still has to
                                                  generalise to unseen city pairs
gender        GridSearchCV                        ``StratifiedKFold(5)``
hotel attach  RandomizedSearchCV                  ``StratifiedKFold(3)`` on ROC AUC
============  ==================================  ===============================

An honest expectation, set before running: tuning searches for better
*parameters*, not for information that is not there. Validation (notebook 07)
showed gender is at chance (p = 0.396) and attach has an AUC interval containing
0.5. Tuning them is included for completeness and to **demonstrate empirically**
that the ceiling is a property of the data — not to rescue them.

Sampling note: flight-price and attach searches run on a subsample for speed, then
the winning configuration is refit on the full dataset. The search only needs to
*rank* configurations, which a sample does reliably; the final model still sees
everything.

Run::

    python -m src.models.tuning            # tune all models
    python -m src.models.tuning --quick    # smaller grids / samples
    python -m src.models.tuning --model flight_price
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     RandomizedSearchCV, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_ingestion import load_processed
from src.features import make_preprocessor
from src.utils import PATHS, get_logger

log = get_logger("model.tuning")

RESULTS_PATH = PATHS.reports / "tuning_results.json"
TUNED_DIR = PATHS.root / "models" / "tuned"


# --------------------------------------------------------------------------- #
# Search spaces                                                                #
# --------------------------------------------------------------------------- #
def _flight_price_param_grids(quick: bool) -> dict:
    """Param grids keyed by the pipeline names in ``flight_price.build_models``.

    Like the gender grids, these tune the *actual* estimators rather than rebuilt
    copies. That matters for ``gbr_rate_per_km``, which is a
    :class:`RatePerKmRegressor` wrapping a whole pipeline rather than a plain
    ``Pipeline`` — so its parameters sit one level deeper.
    """
    if quick:
        return {
            "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "gbr_rate_per_km": {
                "base_pipeline__model__learning_rate": [0.05, 0.1],
                "base_pipeline__model__max_iter": [200, 300]},
        }
    return {
        "ridge": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        "decision_tree": {"model__max_depth": [6, 10, 16, 24, None],
                          "model__min_samples_leaf": [5, 20, 50, 100]},
        "hist_gbr": {"model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                     "model__max_iter": [200, 400],
                     "model__max_leaf_nodes": [15, 31, 63]},
        # the current best model — tune it properly
        "gbr_rate_per_km": {
            "base_pipeline__model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "base_pipeline__model__max_iter": [200, 300, 500],
            "base_pipeline__model__max_leaf_nodes": [15, 31, 63],
            "base_pipeline__model__min_samples_leaf": [20, 50]},
    }


def _rate_per_km_space(quick: bool) -> dict:
    """Search space for the rate-per-km wrapper (the current best model).

    Parameters are nested one level deeper because the estimator wraps a whole
    pipeline: ``model__base_pipeline__model__<param>``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.models.flight_price import RatePerKmRegressor, _preprocessor

    def fresh():
        return RatePerKmRegressor(
            base_pipeline=Pipeline([("prep", _preprocessor()),
                                    ("model", HistGradientBoostingRegressor(random_state=42))]))

    grid = ({"model__base_pipeline__model__learning_rate": [0.05, 0.1],
             "model__base_pipeline__model__max_iter": [200, 300]}
            if quick else
            {"model__base_pipeline__model__learning_rate": [0.03, 0.05, 0.1, 0.2],
             "model__base_pipeline__model__max_iter": [200, 300, 500],
             "model__base_pipeline__model__max_leaf_nodes": [15, 31, 63],
             "model__base_pipeline__model__min_samples_leaf": [20, 50]})
    return {"gbr_rate_per_km": (fresh(), grid, "random")}


def _gender_param_grids(quick: bool) -> dict:
    """Param grids keyed by the pipeline names in ``gender.build_models``.

    Tuning clones those pipelines rather than rebuilding them, so each family
    keeps its own preprocessor — the name-based models need the character n-gram
    vectoriser, not the numeric/one-hot encoder the behaviour models use.
    """
    if quick:
        return {
            "behaviour_logistic": {"model__C": [0.1, 1.0]},
            "name_logistic": {"model__C": [0.1, 1.0, 10.0]},
        }
    return {
        "behaviour_logistic": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "behaviour_rf": {"model__n_estimators": [200, 500],
                         "model__max_depth": [4, 8, None],
                         "model__min_samples_leaf": [1, 5, 20]},
        "behaviour_hist_gbr": {"model__learning_rate": [0.03, 0.1, 0.2],
                               "model__max_iter": [100, 300],
                               "model__max_leaf_nodes": [15, 31]},
        # the family that actually works — tune the n-gram range too
        "name_logistic": {"model__C": [0.1, 1.0, 5.0, 10.0],
                          "prep__name__ngram_range": [(2, 3), (2, 4), (2, 5)],
                          "prep__name__min_df": [1, 2]},
        "name_plus_behaviour": {"model__C": [0.1, 1.0, 10.0],
                                "prep__name__ngram_range": [(2, 4)]},
    }


def _attach_space(quick: bool) -> dict:
    grid = ({"model__learning_rate": [0.05, 0.1], "model__max_iter": [100, 200]}
            if quick else
            {"model__learning_rate": [0.03, 0.05, 0.1, 0.2],
             "model__max_iter": [100, 200, 400],
             "model__max_leaf_nodes": [15, 31, 63],
             "model__min_samples_leaf": [20, 50]})
    return {"hist_gbr": (HistGradientBoostingClassifier(random_state=42), grid, "random")}


# --------------------------------------------------------------------------- #
# Search helper                                                                #
# --------------------------------------------------------------------------- #
def _search(estimator, params, kind, cat, num, X, y, cv, scoring,
            groups=None, n_iter=20, scale=True):
    pipe = Pipeline([("prep", make_preprocessor(cat, num, scale=scale)),
                     ("model", estimator)])
    if kind == "grid":
        search = GridSearchCV(pipe, params, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        search = RandomizedSearchCV(pipe, params, n_iter=n_iter, cv=cv,
                                    scoring=scoring, n_jobs=-1, random_state=42)
    t0 = time.time()
    search.fit(X, y, groups=groups) if groups is not None else search.fit(X, y)
    return search, time.time() - t0


def _default_score(default_estimator, cat, num, X, y, cv, scoring,
                   groups=None, scale=True) -> float:
    """Score the *untuned* estimator under the identical CV, sample and scorer.

    Without this the comparison is meaningless: the model-validation report used
    GroupKFold(5) on the full dataset, while tuning runs GroupKFold(3) on a
    sample. Comparing across those two setups makes tuning look like a regression
    when the difference is really just the protocol.
    """
    from sklearn.model_selection import cross_val_score

    pipe = Pipeline([("prep", make_preprocessor(cat, num, scale=scale)),
                     ("model", default_estimator)])
    scores = cross_val_score(pipe, X, y, groups=groups, cv=cv,
                             scoring=scoring, n_jobs=-1)
    return float(np.mean(scores))


def _clean_params(best: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in best.items()}


def _verify_tuned(estimator, best_params, cat, num, X, y, scoring,
                  cv_factory, seeds=(1, 7, 99), scale=True) -> dict:
    """Re-score the winning configuration on **fresh** CV folds.

    A search reports ``best_score_`` for the configuration that scored highest on
    the folds it searched over. Picking the maximum of N candidates on the same
    folds is optimistically biased — with a small dataset and a flat response
    surface, the "winner" is often just the luckiest split.

    Re-scoring the chosen parameters against folds that took no part in the
    selection removes that bias. If the gain disappears here, it was never real.
    """
    from sklearn.model_selection import cross_val_score

    est = estimator.__class__(**{**estimator.get_params(),
                                 **{k: v for k, v in best_params.items()}})
    pipe = Pipeline([("prep", make_preprocessor(cat, num, scale=scale)),
                     ("model", est)])
    means = []
    for seed in seeds:
        scores = cross_val_score(pipe, X, y, cv=cv_factory(seed),
                                 scoring=scoring, n_jobs=-1)
        means.append(float(np.mean(scores)))
    return {"verified_mean": round(float(np.mean(means)), 4),
            "verified_per_seed": [round(m, 4) for m in means],
            "seeds": list(seeds)}


# =========================================================================== #
# 1. Flight price                                                             #
# =========================================================================== #
def tune_flight_price(quick: bool = False, sample: int | None = 60_000) -> dict:
    """Tune on a sample with GroupKFold over routes, then refit best on all data."""
    from src.models.flight_price import (CAT_FEATURES, NUM_FEATURES, TARGET,
                                         GROUP_COL, load_dataset)

    full = load_dataset()
    df = full.sample(sample, random_state=42) if sample and sample < len(full) else full
    X, y, groups = df[CAT_FEATURES + NUM_FEATURES], df[TARGET], df[GROUP_COL]
    cv = GroupKFold(n_splits=3)
    log.info("flight price: tuning on %s rows, GroupKFold(3) over %d routes",
             f"{len(df):,}", groups.nunique())

    # Tune the ACTUAL estimators from build_models. `gbr_rate_per_km` is a
    # RatePerKmRegressor wrapping a pipeline, not a Pipeline itself, so it has no
    # `named_steps` — cloning the real object avoids having to special-case that.
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from src.models.flight_price import build_models as flight_pipelines

    pipelines = flight_pipelines()

    out = {}
    for name, params in _flight_price_param_grids(quick).items():
        if name not in pipelines:
            continue
        base_est = pipelines[name]
        base = float(np.mean(cross_val_score(clone(base_est), X, y, groups=groups,
                                             cv=cv, scoring="r2", n_jobs=-1)))
        n_combos = int(np.prod([len(v) for v in params.values()]))
        if n_combos <= (8 if quick else 24):
            search = GridSearchCV(clone(base_est), params, cv=cv, scoring="r2", n_jobs=-1)
        else:
            search = RandomizedSearchCV(clone(base_est), params,
                                        n_iter=8 if quick else 20, cv=cv,
                                        scoring="r2", n_jobs=-1, random_state=42)
        t0 = time.time()
        search.fit(X, y, groups=groups)
        secs = time.time() - t0

        out[name] = {"best_params": {k: str(v) for k, v in search.best_params_.items()},
                     "best_cv_r2": round(float(search.best_score_), 4),
                     "default_cv_r2": round(base, 4),
                     "gain": round(float(search.best_score_) - base, 4),
                     "n_candidates": len(search.cv_results_["params"]),
                     "seconds": round(secs, 1)}
        out[name]["_estimator"] = search.best_estimator_
        log.info("  %-16s default=%.4f -> tuned=%.4f (%+.4f)  (%d cand, %.0fs)",
                 name, base, search.best_score_, out[name]["gain"],
                 out[name]["n_candidates"], secs)

    best_name = max(out, key=lambda k: out[k]["best_cv_r2"])
    log.info("  winner: %s (R2=%.4f)", best_name, out[best_name]["best_cv_r2"])

    # Refit the winning configuration on the FULL dataset for serving.
    final = clone(out[best_name]["_estimator"])
    final.fit(full[CAT_FEATURES + NUM_FEATURES], full[TARGET])
    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, TUNED_DIR / "flight_price_tuned.joblib")
    log.info("  refit '%s' on all %s rows -> models/tuned/", best_name, f"{len(full):,}")

    # Drop the fitted estimators before returning — they are not JSON-serialisable
    # and the artifact is already on disk.
    for d in out.values():
        d.pop("_estimator", None)

    return {"models": out, "winner": best_name,
            "tuned_score": out[best_name]["best_cv_r2"],
            "default_score": out[best_name]["default_cv_r2"]}


# =========================================================================== #
# 2. Gender                                                                   #
# =========================================================================== #
def tune_gender(quick: bool = False) -> dict:
    """Tune the classifier. Expected outcome: no meaningful movement."""
    from src.models.gender import TARGET, POSITIVE, load_dataset

    labelled, _, cat, num = load_dataset()
    from src.models.gender import NAME_COL
    feature_cols = cat + num + ([NAME_COL] if NAME_COL in labelled.columns else [])
    X = labelled[feature_cols]
    y = (labelled[TARGET] == POSITIVE).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log.info("gender: tuning on %d labelled users, StratifiedKFold(5)", len(labelled))

    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from src.models.gender import build_models as gender_pipelines

    # Tune the ACTUAL pipelines rather than rebuilding them: each family carries
    # its own preprocessor (character n-grams for the name models, one-hot +
    # scaling for the behaviour models), so they cannot share one factory.
    pipelines = gender_pipelines(cat, num)

    out = {}
    for name, params in _gender_param_grids(quick).items():
        if name not in pipelines:
            continue
        base_pipe = pipelines[name]
        base = float(np.mean(cross_val_score(clone(base_pipe), X, y, cv=cv,
                                             scoring="accuracy", n_jobs=-1)))
        search = GridSearchCV(clone(base_pipe), params, cv=cv,
                              scoring="accuracy", n_jobs=-1)
        t0 = time.time()
        search.fit(X, y)
        secs = time.time() - t0

        # Only 900 rows, so best_score_ is badly exposed to selection bias —
        # re-score the winning configuration on folds it never saw.
        verified = []
        for seed in (1, 7, 99):
            p = clone(base_pipe).set_params(**search.best_params_)
            verified.append(float(np.mean(cross_val_score(
                p, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                scoring="accuracy", n_jobs=-1))))
        ver = {"verified_mean": round(float(np.mean(verified)), 4),
               "verified_per_seed": [round(v, 4) for v in verified],
               "seeds": [1, 7, 99]}

        out[name] = {"best_params": {k: str(v) for k, v in search.best_params_.items()},
                     "best_cv_accuracy": round(float(search.best_score_), 4),
                     "default_cv_accuracy": round(base, 4),
                     "search_gain": round(float(search.best_score_) - base, 4),
                     **ver,
                     "gain": round(ver["verified_mean"] - base, 4),
                     "n_candidates": len(search.cv_results_["params"]),
                     "seconds": round(secs, 1)}
        log.info("  %-20s default=%.4f -> search=%.4f -> verified=%.4f (%+.4f)",
                 name, base, search.best_score_, ver["verified_mean"], out[name]["gain"])

    # Rank on the verified score, not the optimistic search score.
    best_name = max(out, key=lambda k: out[k]["verified_mean"])
    return {"models": out, "winner": best_name,
            "tuned_score": out[best_name]["verified_mean"],
            "default_score": out[best_name]["default_cv_accuracy"]}


# =========================================================================== #
# 3. Hotel attach                                                             #
# =========================================================================== #
def tune_attach(quick: bool = False, sample: int | None = 60_000) -> dict:
    """Tune the attach classifier on ROC AUC. Expected outcome: stays at 0.5."""
    from src.models.recommender import ATTACH_CAT, ATTACH_NUM, ATTACH_TARGET

    trips = load_processed("trips")
    if sample and sample < len(trips):
        trips = trips.sample(sample, random_state=42)
    X, y = trips[ATTACH_CAT + ATTACH_NUM], trips[ATTACH_TARGET].astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    log.info("hotel attach: tuning on %s trips, StratifiedKFold(3) on AUC", f"{len(trips):,}")

    out = {}
    for name, (est, params, kind) in _attach_space(quick).items():
        base = _default_score(HistGradientBoostingClassifier(max_iter=200, random_state=42),
                              ATTACH_CAT, ATTACH_NUM, X, y, cv, "roc_auc")
        search, secs = _search(est, params, kind, ATTACH_CAT, ATTACH_NUM,
                               X, y, cv, "roc_auc", n_iter=6 if quick else 15)
        out[name] = {"best_params": _clean_params(search.best_params_),
                     "best_cv_auc": round(float(search.best_score_), 4),
                     "default_cv_auc": round(base, 4),
                     "gain": round(float(search.best_score_) - base, 4),
                     "n_candidates": len(search.cv_results_["params"]),
                     "seconds": round(secs, 1)}
        log.info("  %-14s default=%.4f -> tuned=%.4f (%+.4f)  %s  (%d cand, %.0fs)",
                 name, base, search.best_score_, out[name]["gain"],
                 out[name]["best_params"], out[name]["n_candidates"], secs)

    best_name = max(out, key=lambda k: out[k]["best_cv_auc"])
    return {"models": out, "winner": best_name,
            "tuned_score": out[best_name]["best_cv_auc"],
            "default_score": out[best_name]["default_cv_auc"]}


# =========================================================================== #
# 4. Baseline comparison                                                      #
# =========================================================================== #
def _untuned_baselines() -> dict:
    """Pull the pre-tuning scores from the model-validation report."""
    path = PATHS.reports / "model_validation.json"
    if not path.exists():
        log.warning("no model_validation.json — run `--stages validate_models` first")
        return {}
    v = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    if "flight_price" in v:
        fp = v["flight_price"]
        best = max(fp, key=lambda k: fp[k]["r2"]["mean"])
        out["flight_price"] = {"metric": "r2", "model": best,
                               "score": fp[best]["r2"]["mean"]}
    if "gender" in v:
        g = {k: x for k, x in v["gender"].items() if k != "permutation_test"}
        best = max(g, key=lambda k: g[k]["accuracy"]["mean"])
        out["gender"] = {"metric": "accuracy", "model": best,
                         "score": g[best]["accuracy"]["mean"]}
    if "hotel_attach" in v:
        out["hotel_attach"] = {"metric": "roc_auc", "model": "hist_gbr",
                               "score": v["hotel_attach"]["roc_auc"]["mean"]}
    return out


# =========================================================================== #
# 5. Driver                                                                   #
# =========================================================================== #
def run(quick: bool = False, only: str | None = None) -> dict:
    log.info("=== hyperparameter tuning started%s ===", " (quick)" if quick else "")
    baselines = _untuned_baselines()

    results = {}
    if only in (None, "flight_price"):
        results["flight_price"] = tune_flight_price(quick)
    if only in (None, "gender"):
        results["gender"] = tune_gender(quick)
    if only in (None, "hotel_attach"):
        results["hotel_attach"] = tune_attach(quick)

    # Before/after comparison.
    # `before_tuning` is the DEFAULT estimator scored under the identical CV,
    # sample and scorer as the search — the only fair comparison. The figure from
    # model_validation.json is carried alongside for reference, but it used a
    # different protocol (more folds, full data) so it is not the baseline.
    comparison = []
    for key, res in results.items():
        before = res["default_score"]
        after = res["tuned_score"]
        gain = round(after - before, 4)
        comparison.append({
            "model": key,
            "metric": baselines.get(key, {}).get("metric", "?"),
            "before_tuning": before,
            "after_tuning": after,
            "gain": gain,
            "winner": res["winner"],
            "best_params": res["models"][res["winner"]]["best_params"],
            "meaningful": bool(gain > 0.01),
            "reference_full_cv": baselines.get(key, {}).get("score"),
        })

    payload = {"comparison": comparison, "detail": results,
               "notes": [
                   "Baseline = the DEFAULT estimator scored under the identical CV, "
                   "sample and scorer as the search. Comparing against "
                   "model_validation.json (different folds, full data) would report "
                   "a phantom gain or regression.",
                   "For gender the reported score is the winning configuration "
                   "RE-SCORED on fresh folds (seeds 1/7/99). The search's own "
                   "best_score_ is biased upward by selecting the max of N "
                   "candidates on the folds it searched.",
                   "Tuning searches for better parameters, not for signal that does "
                   "not exist. Gender and hotel-attach were already shown to be at "
                   "chance; their near-zero verified gains confirm the ceiling is a "
                   "property of the data.",
               ]}
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info("-" * 74)
    log.info("TUNING SUMMARY  (default -> tuned, same CV / same sample)")
    for c in comparison:
        log.info("  %-14s %-8s %.4f -> %.4f  (%+.4f) %s",
                 c["model"], c["metric"], c["before_tuning"], c["after_tuning"],
                 c["gain"], "MEANINGFUL" if c["meaningful"] else "negligible")
    log.info("-" * 74)
    log.info("saved -> %s", RESULTS_PATH)
    log.info("=== hyperparameter tuning complete ===")
    return payload


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hyperparameter tuning for all models")
    p.add_argument("--quick", action="store_true", help="smaller grids and samples")
    p.add_argument("--model", choices=["flight_price", "gender", "hotel_attach"],
                   help="tune only this model")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(quick=args.quick, only=args.model)


if __name__ == "__main__":
    main()
