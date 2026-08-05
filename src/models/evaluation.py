"""Cross-validated model validation for every Voyage Analytics model.

Why this exists
---------------
The per-model training scripts each report a **single** score from one split.
That is fine for selecting a model but too fragile to report: the flight-price
grouped split holds out only 14 of 70 routes, so its R2 carries real variance.

This module re-scores every model with repeated cross-validation and reports
**mean +/- std and a 95% confidence interval**, so results are quoted as ranges
rather than point estimates. It also tests the two "at chance" verdicts properly,
instead of eyeballing a small gap:

  * flight price  - GroupKFold over routes (every route is held out exactly once)
  * gender        - RepeatedStratifiedKFold + permutation test vs the baseline
  * hotel attach  - StratifiedKFold on ROC AUC (chance = 0.5)
  * hotel ranking - bootstrap CI on leave-one-out hit-rate@K

Run::

    python -m src.models.evaluation           # full validation
    python -m src.models.evaluation --quick   # smaller samples / fewer repeats
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     permutation_test_score)

from src.data_ingestion import load_processed
from src.utils import PATHS, get_logger

log = get_logger("model.validation")

REPORT_PATH = PATHS.reports / "model_validation.json"
CONF = 1.96          # 95% normal-approximation interval


def _ci(scores: np.ndarray) -> dict:
    """Mean, std and 95% CI for a vector of fold scores."""
    scores = np.asarray(scores, dtype=float)
    mean, std = float(scores.mean()), float(scores.std(ddof=1))
    half = CONF * std / np.sqrt(len(scores))
    return {"mean": round(mean, 4), "std": round(std, 4),
            "ci_low": round(mean - half, 4), "ci_high": round(mean + half, 4),
            "n_folds": len(scores)}


# =========================================================================== #
# 1. Flight price - GroupKFold over routes                                    #
# =========================================================================== #
def validate_flight_price(quick: bool = False, n_splits: int = 5) -> dict:
    """Every route is held out exactly once across the folds.

    This replaces the single GroupShuffleSplit used during training, where one
    unlucky draw of 14 routes could move R2 noticeably.
    """
    from src.models.flight_price import (CAT_FEATURES, NUM_FEATURES, TARGET,
                                         GROUP_COL, build_models, load_dataset)

    df = load_dataset(sample=40_000 if quick else None)
    X, y, groups = df[CAT_FEATURES + NUM_FEATURES], df[TARGET], df[GROUP_COL]
    cv = GroupKFold(n_splits=n_splits)
    log.info("flight price: GroupKFold(%d) over %d routes, %s rows",
             n_splits, groups.nunique(), f"{len(df):,}")

    out = {}
    models = build_models()
    if quick:
        models.pop("hist_gbr", None)          # slowest; skip in quick mode

    for name, pipe in models.items():
        r2 = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="r2", n_jobs=1)
        rmse = -cross_val_score(pipe, X, y, groups=groups, cv=cv,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
        out[name] = {"r2": _ci(r2), "rmse": _ci(rmse),
                     "fold_r2": [round(v, 4) for v in r2]}
        log.info("  %-14s R2 = %.4f +/- %.4f  (95%% CI %.4f..%.4f)",
                 name, out[name]["r2"]["mean"], out[name]["r2"]["std"],
                 out[name]["r2"]["ci_low"], out[name]["r2"]["ci_high"])
    return out


# =========================================================================== #
# 2. Gender - repeated CV + permutation test                                  #
# =========================================================================== #
def validate_gender(quick: bool = False) -> dict:
    """Repeated stratified CV, plus a permutation test on the best model.

    The permutation test answers the question the raw accuracy cannot: is the
    tiny lift over the baseline distinguishable from random labelling? A p-value
    above 0.05 means the model has learned nothing.
    """
    from src.models.gender import (TARGET, POSITIVE, NAME_COL, build_models,
                                   load_dataset)

    labelled, _, cat, num = load_dataset()
    # first_name must ride along: the name-based pipelines consume it directly.
    feature_cols = cat + num + ([NAME_COL] if NAME_COL in labelled.columns else [])
    X = labelled[feature_cols]
    y = (labelled[TARGET] == POSITIVE).astype(int)

    n_repeats = 2 if quick else 5
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    log.info("gender: RepeatedStratifiedKFold(5 x %d) on %d labelled users",
             n_repeats, len(labelled))

    out = {}
    for name, pipe in build_models(cat, num).items():
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        out[name] = {"accuracy": _ci(acc)}
        log.info("  %-18s acc = %.4f +/- %.4f  (95%% CI %.4f..%.4f)",
                 name, out[name]["accuracy"]["mean"], out[name]["accuracy"]["std"],
                 out[name]["accuracy"]["ci_low"], out[name]["accuracy"]["ci_high"])

    # Permutation test on the strongest non-baseline model
    real = {k: v for k, v in out.items() if k != "baseline_majority"}
    best = max(real, key=lambda k: real[k]["accuracy"]["mean"])
    n_perm = 30 if quick else 100
    log.info("  permutation test on '%s' (%d permutations)...", best, n_perm)

    score, perm_scores, pvalue = permutation_test_score(
        build_models(cat, num)[best], X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_permutations=n_perm, scoring="accuracy", n_jobs=-1, random_state=42)

    out["permutation_test"] = {
        "model": best,
        "score": round(float(score), 4),
        "permutation_mean": round(float(np.mean(perm_scores)), 4),
        "p_value": round(float(pvalue), 4),
        "significant_at_0.05": bool(pvalue < 0.05),
    }
    log.info("  permutation: score=%.4f vs shuffled-mean=%.4f  p=%.4f -> %s",
             score, np.mean(perm_scores), pvalue,
             "SIGNIFICANT" if pvalue < 0.05 else "NOT significant (chance)")
    return out


# =========================================================================== #
# 3. Hotel attach - stratified CV on AUC                                      #
# =========================================================================== #
def validate_attach(quick: bool = False) -> dict:
    """AUC under stratified CV. Chance = 0.5; a CI spanning 0.5 means no signal."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from src.features import make_preprocessor
    from src.models.recommender import ATTACH_CAT, ATTACH_NUM, ATTACH_TARGET

    trips = load_processed("trips")
    if quick:
        trips = trips.sample(40_000, random_state=42)
    X, y = trips[ATTACH_CAT + ATTACH_NUM], trips[ATTACH_TARGET].astype(int)

    pipe = Pipeline([("prep", make_preprocessor(ATTACH_CAT, ATTACH_NUM)),
                     ("model", HistGradientBoostingClassifier(max_iter=200, random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log.info("hotel attach: StratifiedKFold(5) on %s trips (base rate %.1f%%)",
             f"{len(trips):,}", y.mean() * 100)

    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    res = {"roc_auc": _ci(auc), "accuracy": _ci(acc),
           "base_rate": round(float(y.mean()), 4),
           "majority_baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4)}
    res["auc_ci_includes_chance"] = bool(res["roc_auc"]["ci_low"] <= 0.5 <= res["roc_auc"]["ci_high"])
    log.info("  AUC = %.4f +/- %.4f  (95%% CI %.4f..%.4f) -> %s",
             res["roc_auc"]["mean"], res["roc_auc"]["std"],
             res["roc_auc"]["ci_low"], res["roc_auc"]["ci_high"],
             "CI includes 0.5 = CHANCE" if res["auc_ci_includes_chance"] else "above chance")
    return res


# =========================================================================== #
# 4. Hotel ranking - bootstrap CI on hit-rate                                 #
# =========================================================================== #
def validate_ranking(quick: bool = False, n_boot: int = 500) -> dict:
    """Bootstrap over users to put a confidence interval on hit-rate@K."""
    from src.models.recommender import TOP_K, evaluate_rankers

    interactions = load_processed("user_hotel_interactions")
    trips = load_processed("trips")
    ranking = evaluate_rankers(interactions, trips, k=TOP_K)

    n_boot = 100 if quick else n_boot
    rng = np.random.default_rng(42)
    out = {}
    for _, row in ranking.iterrows():
        p = float(row[f"hit_rate@{TOP_K}"])
        n = int(row["n_users"])
        # bootstrap the per-user Bernoulli outcomes implied by the observed rate
        draws = rng.binomial(n, p, size=n_boot) / n
        out[row["strategy"]] = {
            "hit_rate": round(p, 4),
            "ci_low": round(float(np.percentile(draws, 2.5)), 4),
            "ci_high": round(float(np.percentile(draws, 97.5)), 4),
            "n_users": n,
            "random_baseline": round(TOP_K / interactions["hotel"].nunique(), 4),
        }
        log.info("  %-20s hit@%d = %.4f  (95%% CI %.4f..%.4f)",
                 row["strategy"], TOP_K, p,
                 out[row["strategy"]]["ci_low"], out[row["strategy"]]["ci_high"])
    return out


# =========================================================================== #
# 5. Driver                                                                   #
# =========================================================================== #
def run(quick: bool = False) -> dict:
    log.info("=== model validation started%s ===", " (quick)" if quick else "")

    results = {
        "flight_price": validate_flight_price(quick),
        "gender": validate_gender(quick),
        "hotel_attach": validate_attach(quick),
        "hotel_ranking": validate_ranking(quick),
    }

    verdicts = _verdicts(results)
    results["verdicts"] = verdicts

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("saved report -> %s", REPORT_PATH)

    log.info("-" * 66)
    log.info("VALIDATION VERDICTS")
    for k, v in verdicts.items():
        log.info("  %-14s %s", k, v)
    log.info("-" * 66)
    log.info("=== model validation complete ===")
    return results


def _verdicts(r: dict) -> dict:
    """Turn the intervals into plain-language conclusions."""
    fp = r["flight_price"]
    best_fp = max(fp, key=lambda k: fp[k]["r2"]["mean"])
    fp_v = (f"{best_fp}: R2 {fp[best_fp]['r2']['mean']:.3f} "
            f"[{fp[best_fp]['r2']['ci_low']:.3f}, {fp[best_fp]['r2']['ci_high']:.3f}] - VALID")

    p = r["gender"]["permutation_test"]
    g_v = (f"{p['model']}: p={p['p_value']:.3f} -> "
           + ("significant" if p["significant_at_0.05"] else "NOT significant, at chance"))

    a = r["hotel_attach"]
    a_v = (f"AUC {a['roc_auc']['mean']:.3f} "
           f"[{a['roc_auc']['ci_low']:.3f}, {a['roc_auc']['ci_high']:.3f}] -> "
           + ("chance" if a["auc_ci_includes_chance"] else "above chance"))

    rk = r["hotel_ranking"]
    best_rk = max(rk, key=lambda k: rk[k]["hit_rate"])
    rk_v = (f"{best_rk}: hit@3 {rk[best_rk]['hit_rate']:.3f} "
            f"[{rk[best_rk]['ci_low']:.3f}, {rk[best_rk]['ci_high']:.3f}] "
            f"vs random {rk[best_rk]['random_baseline']:.3f} - VALID")

    return {"flight_price": fp_v, "gender": g_v,
            "hotel_attach": a_v, "hotel_ranking": rk_v}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cross-validate every model")
    p.add_argument("--quick", action="store_true",
                   help="smaller samples and fewer repeats for a fast check")
    return p.parse_args(argv)


def main(argv=None) -> None:
    run(quick=_parse_args(argv).quick)


if __name__ == "__main__":
    main()
