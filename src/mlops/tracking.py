"""MLflow experiment tracking and model registry (project objective #7).

Wraps every model's training run so that parameters, metrics, artifacts and the
serialized pipeline land in one tracking store, and promotes the models that
survived validation into the **Model Registry**.

Design
------
Training is executed *inside* an MLflow run rather than logged afterwards, so the
recorded parameters are guaranteed to be the ones that produced the metrics.
Cross-validated figures from ``reports/model_validation.json`` are attached to the
same run, so a single place answers "how good is this model, really".

Only models that passed validation are registered. Gender and hotel-attach are
logged for the record — with their verdicts — but **not** promoted, because
registering a chance-level model invites someone downstream to deploy it.

Backend: local file store at ``mlflow/mlruns`` — no server required. Launch the UI
with::

    mlflow ui --backend-store-uri mlflow/mlruns

Run::

    python -m src.mlops                  # track everything, register valid models
    python -m src.mlops --no-register    # log runs only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.utils import PATHS, get_logger

log = get_logger("mlops")

TRACKING_DIR = PATHS.root / "mlflow" / "mlruns"
EXPERIMENT = "voyage-analytics"

# Registry names for the models that passed validation.
# Gender was added once first-name character n-grams took it from chance (0.50)
# to 0.88 standard CV / 0.74 on names never seen in training. Hotel-attach is
# still excluded — it remains at AUC 0.50 under every configuration tried.
REGISTERED_NAMES = {
    "flight_price": "voyage-flight-price",
    "hotel_recommender": "voyage-hotel-recommender",
    "gender": "voyage-gender-classifier",
}


# --------------------------------------------------------------------------- #
# Setup                                                                        #
# --------------------------------------------------------------------------- #
def setup_mlflow(experiment: str = EXPERIMENT) -> str:
    """Point MLflow at the local file store and select the experiment."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.resolve().as_uri())
    mlflow.set_experiment(experiment)
    log.info("tracking uri: %s", mlflow.get_tracking_uri())
    log.info("experiment  : %s", experiment)
    return mlflow.get_tracking_uri()


def _validation_metrics() -> dict:
    """Cross-validated results, if `validate_models` has been run."""
    path = PATHS.reports / "model_validation.json"
    if not path.exists():
        log.warning("no model_validation.json — run `python main.py --stages validate_models`")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _log_artifact_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _flight_price_example(n: int = 2) -> pd.DataFrame:
    """A real row from the training table, so the signature dtypes are exact."""
    from src.data_ingestion import load_processed
    from src.models.flight_price import CAT_FEATURES, NUM_FEATURES

    df = load_processed("flight_price_model")
    # cast categoricals to plain strings: MLflow signatures do not model pandas
    # 'category' dtype, and JSON payloads arrive as strings anyway.
    ex = df[CAT_FEATURES + NUM_FEATURES].head(n).copy()
    for c in CAT_FEATURES:
        ex[c] = ex[c].astype(str)
    return ex


def _gender_example(n: int = 2) -> pd.DataFrame:
    """A real labelled row for the gender classifier's signature.

    Must carry ``first_name`` as well: the winning pipeline feeds it to a
    character n-gram vectoriser, so a signature without it would reject every
    real request.
    """
    from src.models.gender import NAME_COL, load_dataset

    labelled, _, cat, num = load_dataset()
    cols = cat + num + ([NAME_COL] if NAME_COL in labelled.columns else [])
    ex = labelled[cols].head(n).copy()
    for c in cat + ([NAME_COL] if NAME_COL in ex.columns else []):
        ex[c] = ex[c].astype(str)
    return ex


# --------------------------------------------------------------------------- #
# Per-model tracking                                                           #
# --------------------------------------------------------------------------- #
def track_flight_price(validation: dict) -> dict:
    """Train the flight-price regressor inside an MLflow run and log everything."""
    from src.models import flight_price

    with mlflow.start_run(run_name="flight_price_regression") as run:
        result = flight_price.run()
        res: pd.DataFrame = result["results"]
        best = result["best_model"]

        mlflow.set_tags({
            "objective": "1 - flight price prediction",
            "problem_type": "regression",
            "validation": "GroupShuffleSplit + GroupKFold over routes",
            "status": "VALID",
        })
        mlflow.log_params({
            "model": best,
            "features_categorical": ",".join(flight_price.CAT_FEATURES),
            "features_numeric": ",".join(flight_price.NUM_FEATURES),
            "target": flight_price.TARGET,
            "group_column": flight_price.GROUP_COL,
            "excluded_features": "time (r=0.99999 with distance), route (group key)",
        })

        # single-split metrics for every candidate, both splits
        for _, r in res.iterrows():
            prefix = f"{r['split']}_{r['model']}"
            mlflow.log_metrics({f"{prefix}_r2": r["r2"], f"{prefix}_rmse": r["rmse"],
                                f"{prefix}_mae": r["mae"], f"{prefix}_mape": r["mape_pct"]})

        # headline metrics = the honest grouped split for the selected model
        sel = res[(res["split"] == "grouped") & (res["model"] == best)].iloc[0]
        mlflow.log_metrics({"r2": sel["r2"], "rmse": sel["rmse"],
                            "mae": sel["mae"], "mape_pct": sel["mape_pct"],
                            "mse": round(sel["rmse"] ** 2, 2)})

        # cross-validated figures
        cv = validation.get("flight_price", {}).get(best)
        if cv:
            mlflow.log_metrics({
                "cv_r2_mean": cv["r2"]["mean"], "cv_r2_std": cv["r2"]["std"],
                "cv_r2_ci_low": cv["r2"]["ci_low"], "cv_r2_ci_high": cv["r2"]["ci_high"],
                "cv_rmse_mean": cv["rmse"]["mean"],
            })

        # Log with an input example so MLflow infers a signature — the serving
        # layer then validates incoming payloads against it instead of failing
        # deep inside the pipeline on a wrong dtype or missing column.
        example = _flight_price_example()
        mlflow.sklearn.log_model(result["pipeline"], artifact_path="model",
                                 input_example=example,
                                 signature=infer_signature(example,
                                                           result["pipeline"].predict(example)))
        _log_artifact_if_exists(PATHS.reports / "flight_price_metrics.json", "reports")
        log.info("logged flight_price run %s", run.info.run_id)
        return {"run_id": run.info.run_id, "model": best, "registrable": True}


def track_gender(validation: dict) -> dict:
    """Log the gender classifier together with its 'not significant' verdict."""
    from src.models import gender

    with mlflow.start_run(run_name="gender_classification") as run:
        result = gender.run()
        res: pd.DataFrame = result["results"]

        perm = validation.get("gender", {}).get("permutation_test", {})
        best_name = str(result["best_model"])
        # Status reflects the measured outcome rather than a fixed assumption:
        # behaviour-only models sit at chance, the name-based ones do not.
        is_valid = result["lift"] >= 0.05
        mlflow.set_tags({
            "objective": "8 - gender classification",
            "problem_type": "binary classification",
            "validation": ("RepeatedStratifiedKFold + permutation test + "
                           "GroupKFold over first name (unseen names)"),
            "status": "VALID" if is_valid else "AT_CHANCE",
            "signal_source": ("first-name character n-grams"
                              if best_name.startswith("name")
                              else "travel behaviour"),
            "verdict": result["verdict"],
        })
        mlflow.log_params({
            "model": result["best_model"],
            "target": gender.TARGET,
            "positive_class": gender.POSITIVE,
            "excluded_features": "name (would leak the label)",
        })
        for _, r in res.iterrows():
            m = {f"{r['model']}_cv_accuracy": r["cv_accuracy"],
                 f"{r['model']}_test_accuracy": r["test_accuracy"]}
            if r.get("cv_accuracy_unseen_names") is not None:
                m[f"{r['model']}_cv_accuracy_unseen_names"] = r["cv_accuracy_unseen_names"]
            mlflow.log_metrics(m)

        best_row = res.loc[res["model"] == result["best_model"]].iloc[0]
        metrics = {
            "accuracy": float(best_row["cv_accuracy"]),
            "baseline_accuracy": result["baseline"],
            "lift_over_baseline": result["lift"],
        }
        # The honest headline: accuracy on names the model has never seen.
        if best_row.get("cv_accuracy_unseen_names") is not None:
            metrics["accuracy_unseen_names"] = float(best_row["cv_accuracy_unseen_names"])
        mlflow.log_metrics(metrics)
        if perm:
            mlflow.log_metrics({"permutation_p_value": perm["p_value"],
                                "permutation_score": perm["score"]})

        example = _gender_example()
        mlflow.sklearn.log_model(result["pipeline"], artifact_path="model",
                                 input_example=example,
                                 signature=infer_signature(example,
                                                           result["pipeline"].predict(example)))
        _log_artifact_if_exists(PATHS.reports / "gender_metrics.json", "reports")
        log.info("logged gender run %s (%s)", run.info.run_id,
                 "registrable" if is_valid else "not registered: at chance")
        return {"run_id": run.info.run_id, "model": result["best_model"],
                "registrable": is_valid}


def track_recommender(validation: dict) -> dict:
    """Log the hotel ranker (valid) and the attach classifier (chance)."""
    from src.models import recommender

    with mlflow.start_run(run_name="hotel_recommender") as run:
        result = recommender.run()
        ranking: pd.DataFrame = result["ranking"]
        attach = result["attach_metrics"]
        rank_cv = validation.get("hotel_ranking", {})
        attach_cv = validation.get("hotel_attach", {})

        mlflow.set_tags({
            "objective": "9 - hotel recommendation",
            "problem_type": "ranking (collaborative filtering) + binary classification",
            "validation": "leave-one-out + bootstrap CI",
            "status": "RANKING_VALID / ATTACH_AT_CHANCE",
        })
        mlflow.log_params({
            "best_strategy": result["best_strategy"],
            "top_k": recommender.TOP_K,
            "attach_model": "HistGradientBoostingClassifier",
            "n_hotels": len(result["artifact"]["popularity_order"]),
        })
        for _, r in ranking.iterrows():
            mlflow.log_metric(f"hit_rate_{r['strategy']}", r[f"hit_rate@{recommender.TOP_K}"])

        best_strategy = result["best_strategy"]
        mlflow.log_metrics({
            "hit_rate_at_k": float(ranking.iloc[0][f"hit_rate@{recommender.TOP_K}"]),
            "attach_roc_auc": attach["roc_auc"],
            "attach_accuracy": attach["accuracy"],
            "attach_baseline_accuracy": attach["majority_baseline_accuracy"],
        })
        if best_strategy in rank_cv:
            mlflow.log_metrics({"hit_rate_ci_low": rank_cv[best_strategy]["ci_low"],
                                "hit_rate_ci_high": rank_cv[best_strategy]["ci_high"],
                                "random_baseline": rank_cv[best_strategy]["random_baseline"]})
        if attach_cv:
            mlflow.log_metrics({"attach_cv_auc_mean": attach_cv["roc_auc"]["mean"],
                                "attach_cv_auc_ci_low": attach_cv["roc_auc"]["ci_low"],
                                "attach_cv_auc_ci_high": attach_cv["roc_auc"]["ci_high"]})

        _log_artifact_if_exists(PATHS.root / "models" / "hotel_recommender.joblib", "model")
        _log_artifact_if_exists(PATHS.root / "models" / "hotel_attach_model.joblib", "model")
        _log_artifact_if_exists(PATHS.reports / "recommender_metrics.json", "reports")
        log.info("logged recommender run %s", run.info.run_id)
        return {"run_id": run.info.run_id, "model": best_strategy, "registrable": True}


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
def register_model(run_id: str, key: str, artifact_path: str = "model") -> str | None:
    """Register a run's model under its registry name and return the version."""
    name = REGISTERED_NAMES.get(key)
    if name is None:
        return None
    uri = f"runs:/{run_id}/{artifact_path}"
    try:
        mv = mlflow.register_model(model_uri=uri, name=name)
        log.info("registered %s version %s", name, mv.version)
        return mv.version
    except Exception as exc:                      # noqa: BLE001 - registry is optional
        log.warning("could not register %s: %s", name, exc)
        return None


def list_runs(experiment: str = EXPERIMENT) -> pd.DataFrame:
    """Return this experiment's runs as a tidy frame for notebooks/reports."""
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return pd.DataFrame()
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    keep = [c for c in df.columns
            if c.startswith(("metrics.", "tags.", "params.")) or c in
            ("run_id", "start_time", "status")]
    return df[keep]


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run(register: bool = True) -> dict:
    log.info("=== MLflow tracking started ===")
    setup_mlflow()
    validation = _validation_metrics()

    tracked = {
        "flight_price": track_flight_price(validation),
        "gender": track_gender(validation),
        "hotel_recommender": track_recommender(validation),
    }

    registered = {}
    if register:
        for key, info in tracked.items():
            if info["registrable"]:
                v = register_model(info["run_id"], key)
                if v:
                    registered[REGISTERED_NAMES[key]] = v
        skipped = [k for k, i in tracked.items() if not i["registrable"]]
        if skipped:
            log.info("not registered (failed validation): %s", ", ".join(skipped))

    log.info("tracked %d runs, registered %d models", len(tracked), len(registered))
    log.info("view with: mlflow ui --backend-store-uri mlflow/mlruns")
    log.info("=== MLflow tracking complete ===")
    return {"tracked": tracked, "registered": registered,
            "tracking_uri": mlflow.get_tracking_uri()}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track all models with MLflow")
    p.add_argument("--no-register", action="store_true",
                   help="log runs without promoting models to the registry")
    return p.parse_args(argv)


def main(argv=None) -> None:
    run(register=not _parse_args(argv).no_register)


if __name__ == "__main__":
    main()
