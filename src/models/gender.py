"""Gender classification (project objective #8).

Predicts ``users.gender`` from demographics and travel behaviour, then imputes
the ~33% of users recorded as ``none``.

Honest framing
--------------
Feature analysis (notebook 04) showed the strongest correlation between any
feature and gender is |r| = 0.06, with the two classes balanced ~50/50. So the
expected ceiling here is **near chance**, and a number like 0.75 would be a red
flag for leakage rather than a success.

Every model is therefore scored against a :class:`DummyClassifier` baseline, and
the report states the lift over that baseline rather than the raw accuracy alone.
``name`` is excluded on purpose — predicting gender from a first name is a lookup
that would leak the answer and fail on exactly the ambiguous cases that matter.

Run::

    python -m src.models.gender
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_ingestion import load_processed
from src.features import GENDER_CAT_FEATURES, GENDER_NUM_FEATURES, make_preprocessor
from src.utils import PATHS, get_logger

log = get_logger("model.gender")

TARGET = "gender"
POSITIVE = "male"
ARTIFACT_PATH = PATHS.root / "models" / "gender_model.joblib"
METRICS_PATH = PATHS.reports / "gender_metrics.json"
IMPUTED_PATH = PATHS.processed / "gender_imputed.csv"


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
NAME_COL = "first_name"


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split the feature table into the labelled train pool and the rows to impute."""
    df = load_processed("gender_features")

    share_cols = [c for c in df.columns if c.startswith(("share_class_", "share_agency_"))]
    num = [c for c in GENDER_NUM_FEATURES + share_cols if c in df.columns]
    cat = [c for c in GENDER_CAT_FEATURES if c in df.columns]

    labelled = df[df["gender_known"]].copy()
    unlabelled = df[~df["gender_known"]].copy()

    log.info("labelled=%d  to-impute=%d  features=%d cat + %d num",
             len(labelled), len(unlabelled), len(cat), len(num))
    return labelled, unlabelled, cat, num


def _name_preprocessor(cat: list[str], num: list[str], with_behaviour: bool = True):
    """Character n-grams over the first name, optionally plus behaviour features.

    ``char_wb`` n-grams capture morphology (the '-a' ending, '-o', common
    prefixes) rather than memorising whole names, which is what lets the model
    handle a name it has never seen.
    """
    blocks = [("name", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                       min_df=2, lowercase=True), NAME_COL)]
    if with_behaviour:
        blocks += [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                   ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                     ("scale", StandardScaler())]), num)]
    return ColumnTransformer(blocks, remainder="drop")


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def build_models(cat: list[str], num: list[str]) -> dict[str, Pipeline]:
    """Baseline first — every other model is judged relative to it.

    The ``behaviour_*`` models use travel features only and sit at chance. The
    ``name_*`` models add first-name character n-grams, which is where the signal
    actually is. Both families are kept so the comparison stays visible.
    """
    def pipe(est):
        return Pipeline([("prep", make_preprocessor(cat, num)), ("model", est)])

    def name_pipe(est, with_behaviour=True):
        return Pipeline([("prep", _name_preprocessor(cat, num, with_behaviour)),
                         ("model", est)])

    return {
        "baseline_majority": pipe(DummyClassifier(strategy="most_frequent")),
        # behaviour only — retained to show it carries no signal
        "behaviour_logistic": pipe(LogisticRegression(max_iter=1000, random_state=42)),
        "behaviour_rf": pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)),
        "behaviour_hist_gbr": pipe(HistGradientBoostingClassifier(
            max_iter=200, random_state=42)),
        # first-name character n-grams — the models that actually work
        "name_logistic": name_pipe(
            LogisticRegression(max_iter=2000, random_state=42), with_behaviour=False),
        "name_plus_behaviour": name_pipe(
            LogisticRegression(max_iter=2000, random_state=42), with_behaviour=True),
    }


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate(labelled: pd.DataFrame, cat: list[str], num: list[str],
             seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """5-fold stratified CV plus a held-out test split for every candidate."""
    # Keep first_name in X so the name-based pipelines can reach it.
    feature_cols = cat + num + ([NAME_COL] if NAME_COL in labelled.columns else [])
    X = labelled[feature_cols]
    y = (labelled[TARGET] == POSITIVE).astype(int)
    names = labelled[NAME_COL] if NAME_COL in labelled.columns else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows, fitted = [], {}
    for name, pipe in build_models(cat, num).items():
        cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        # Grouped by first name: the test fold contains only names absent from
        # training. 583 of 900 users share a name with someone else, so standard
        # CV lets the model see a test name during training and reads high.
        grouped_acc = None
        if names is not None:
            g = cross_val_score(pipe, X, y, groups=names, cv=GroupKFold(5),
                                scoring="accuracy", n_jobs=-1)
            grouped_acc = round(float(g.mean()), 4)

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        try:
            proba = pipe.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
        except Exception:                       # baseline has no meaningful proba
            auc = float("nan")

        rows.append({
            "model": name,
            "cv_accuracy": round(cv_acc.mean(), 4),
            "cv_std": round(cv_acc.std(), 4),
            "cv_accuracy_unseen_names": grouped_acc,
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_f1": round(f1_score(y_te, pred, zero_division=0), 4),
            "test_roc_auc": round(auc, 4) if auc == auc else None,
        })
        fitted[name] = pipe
        log.info("%-20s cv_acc=%.4f  unseen_names=%s  auc=%s",
                 name, cv_acc.mean(),
                 f"{grouped_acc:.4f}" if grouped_acc is not None else "n/a",
                 rows[-1]["test_roc_auc"])

    results = pd.DataFrame(rows)
    baseline = results.loc[results["model"] == "baseline_majority", "cv_accuracy"].iloc[0]
    results["lift_over_baseline"] = (results["cv_accuracy"] - baseline).round(4)
    return results, {"fitted": fitted, "baseline": baseline,
                     "X_te": X_te, "y_te": y_te}


# --------------------------------------------------------------------------- #
# Train / impute / persist                                                     #
# --------------------------------------------------------------------------- #
def impute_unlabelled(pipe: Pipeline, unlabelled: pd.DataFrame,
                      cat: list[str], num: list[str]) -> pd.DataFrame:
    """Predict gender for the users recorded as ``none``."""
    if unlabelled.empty:
        return unlabelled
    feature_cols = cat + num + ([NAME_COL] if NAME_COL in unlabelled.columns else [])
    X = unlabelled[feature_cols]
    pred = pipe.predict(X)
    out = unlabelled[["code"]].copy()
    out["gender_predicted"] = np.where(pred == 1, POSITIVE, "female")
    try:
        out["confidence"] = pipe.predict_proba(X).max(axis=1).round(4)
    except Exception:
        out["confidence"] = np.nan
    return out


def run(seed: int = 42) -> dict:
    """Full run: load -> evaluate -> select -> refit -> impute -> save."""
    log.info("=== gender classification started ===")
    labelled, unlabelled, cat, num = load_dataset()

    results, extra = evaluate(labelled, cat, num, seed)

    # Select the best non-baseline model on cross-validated accuracy.
    real = results[results["model"] != "baseline_majority"]
    best = real.loc[real["cv_accuracy"].idxmax(), "model"]
    best_acc = real.loc[real["cv_accuracy"].idxmax(), "cv_accuracy"]
    lift = best_acc - extra["baseline"]
    log.info("best=%s cv_acc=%.4f | baseline=%.4f | lift=%+.4f",
             best, best_acc, extra["baseline"], lift)

    # Refit on all labelled data, then impute the 'none' rows.
    pipe = build_models(cat, num)[best]
    feature_cols = cat + num + ([NAME_COL] if NAME_COL in labelled.columns else [])
    X = labelled[feature_cols]
    y = (labelled[TARGET] == POSITIVE).astype(int)
    pipe.fit(X, y)

    imputed = impute_unlabelled(pipe, unlabelled, cat, num)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACT_PATH)
    if not imputed.empty:
        imputed.to_csv(IMPUTED_PATH, index=False)

    # Judge on the unseen-names figure where available: standard CV lets a name
    # seen in training reappear in the test fold, which reads optimistically.
    best_row = real.loc[real["cv_accuracy"].idxmax()]
    unseen = best_row.get("cv_accuracy_unseen_names")
    if lift < 0.05:
        verdict = "near chance - gender is not predictable from this data"
    elif unseen is not None:
        verdict = (f"predictive from the first name: {best_acc:.4f} standard CV, "
                   f"{unseen:.4f} on names never seen in training "
                   f"(behaviour features alone remain at chance)")
    else:
        verdict = "meaningful lift over baseline"
    METRICS_PATH.write_text(json.dumps({
        "best_model": best,
        "baseline_accuracy": round(extra["baseline"], 4),
        "best_cv_accuracy": round(float(best_acc), 4),
        "best_cv_accuracy_unseen_names": unseen,
        "lift_over_baseline": round(float(lift), 4),
        "verdict": verdict,
        "n_labelled": len(labelled),
        "n_imputed": len(imputed),
        "features": {"categorical": cat, "numeric": num},
        "results": results.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    log.info("saved model   -> %s", ARTIFACT_PATH)
    log.info("saved metrics -> %s", METRICS_PATH)
    log.info("verdict: %s", verdict)
    log.info("=== gender classification complete ===")

    return {"results": results, "best_model": best, "pipeline": pipe,
            "imputed": imputed, "baseline": extra["baseline"],
            "lift": lift, "verdict": verdict}


def main(argv=None) -> None:
    argparse.ArgumentParser(description="Train the gender classifier").parse_args(argv)
    run()


if __name__ == "__main__":
    main()
