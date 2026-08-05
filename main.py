"""Voyage Analytics — end-to-end pipeline entry point.

Runs every stage in `src/` sequentially, in dependency order::

    1. validate      gate data/raw against schema, bounds and referential rules
    2. ingest        raw CSVs -> clean (data/interim) -> entity tables (data/processed)
    3. features      entity tables -> model-ready feature sets (data/processed)
    4. audit         re-validate everything the pipeline wrote
    5. train_price      flight-price regression    (objective #1)
    6. train_gender     gender classification      (objective #8)
    7. train_recsys     hotel recommender + attach (objective #9)
    8. validate_models  cross-validate all models, with confidence intervals
    9. tune             hyperparameter search vs the untuned baseline
   10. mlflow           track runs + register validated models (objective #7)

Each stage consumes what the previous one wrote, so the order is not optional.

Usage
-----
::

    python main.py                          # run the whole pipeline
    python main.py --quick                  # fast smoke run (samples during training)
    python main.py --stages ingest features # run a subset, in the given order
    python main.py --skip validate audit    # run everything except those stages
    python main.py --parquet                # write parquet instead of CSV
    python main.py --list                   # show available stages and exit

Exit code is 0 only if every requested stage succeeded.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

# Make `src` importable no matter where the script is invoked from.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from src.utils import PATHS, get_logger  # noqa: E402

log = get_logger("pipeline")


# --------------------------------------------------------------------------- #
# Stage definitions                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    """One pipeline step: a name, a description and the function that runs it."""

    name: str
    description: str
    run: Callable[["Options"], object]


@dataclass
class Options:
    """Runtime switches shared by every stage."""

    parquet: bool = False
    quick: bool = False
    validate: bool = True


@dataclass
class StageResult:
    name: str
    status: str                       # ok | failed | skipped
    seconds: float = 0.0
    detail: str = ""
    error: str = ""


# --- stage bodies ---------------------------------------------------------- #
def _run_validate(opts: Options):
    from src.validation import run_validation
    reports = run_validation(scope="raw", strict=True)
    return reports["raw"].summary()


def _run_audit(opts: Options):
    from src.validation import run_validation
    reports = run_validation(scope="processed", strict=True)
    return reports["processed"].summary()


def _run_ingest(opts: Options):
    from src.data_ingestion import run_pipeline
    result = run_pipeline(parquet=opts.parquet)
    return f"{len(result['written'])} files"


def _run_features(opts: Options):
    from src.features import build_all
    result = build_all(parquet=opts.parquet)
    return f"{len(result['written'])} feature sets"


def _run_train_flight_price(opts: Options):
    from src.models import flight_price
    result = flight_price.run(sample=40_000 if opts.quick else None)
    res = result["results"]
    grouped = res[res["split"] == "grouped"]
    best_row = grouped.loc[grouped["rmse"].idxmin()]
    return (f"best={result['best_model']} "
            f"(held-out routes R2={best_row['r2']:.4f}, RMSE={best_row['rmse']:.2f})")


def _run_train_gender(opts: Options):
    from src.models import gender
    result = gender.run()
    return (f"best={result['best_model']} "
            f"(cv_acc vs baseline {result['baseline']:.3f}, "
            f"lift={result['lift']:+.4f}) — {result['verdict']}")


def _run_tune(opts: Options):
    from src.models import tuning
    result = tuning.run(quick=opts.quick)
    parts = []
    for c in result["comparison"]:
        gain = c["gain"] if c["gain"] is not None else 0.0
        parts.append(f"{c['model']} {gain:+.4f}"
                     + ("" if c["meaningful"] else " (negligible)"))
    return " | ".join(parts)


def _run_mlflow(opts: Options):
    from src.mlops import run as track_all
    result = track_all(register=True)
    return (f"{len(result['tracked'])} runs tracked, "
            f"{len(result['registered'])} model(s) registered")


def _run_validate_models(opts: Options):
    from src.models import evaluation
    result = evaluation.run(quick=opts.quick)
    v = result["verdicts"]
    return " | ".join(f"{k}: {val.split(' - ')[0].split(' -> ')[0]}" for k, val in v.items())


def _run_train_recommender(opts: Options):
    from src.models import recommender
    result = recommender.run()
    a = result["attach_metrics"]
    return (f"ranking best={result['best_strategy']} | "
            f"attach auc={a['roc_auc']:.4f} acc={a['accuracy']:.4f}")


STAGES: tuple[Stage, ...] = (
    Stage("validate", "Gate the raw CSVs on schema, bounds and referential rules", _run_validate),
    Stage("ingest", "Clean the raw CSVs and build the entity tables", _run_ingest),
    Stage("features", "Build model-ready feature sets from the entity tables", _run_features),
    Stage("audit", "Re-validate every table the pipeline wrote", _run_audit),
    Stage("train_price", "Train the flight-price regressor (objective #1)", _run_train_flight_price),
    Stage("train_gender", "Train the gender classifier (objective #8)", _run_train_gender),
    Stage("train_recsys", "Train the hotel recommender + attach model (objective #9)",
          _run_train_recommender),
    Stage("validate_models", "Cross-validate every model with confidence intervals",
          _run_validate_models),
    Stage("tune", "Hyperparameter search for every model, vs the untuned baseline",
          _run_tune),
    Stage("mlflow", "Track all runs in MLflow and register validated models (objective #7)",
          _run_mlflow),
)

STAGE_MAP = {s.name: s for s in STAGES}


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run_pipeline(stage_names: list[str], opts: Options) -> list[StageResult]:
    """Execute the named stages in order, stopping at the first failure.

    A later stage depends on the artifacts of an earlier one, so continuing past
    a failure would only produce misleading downstream results.
    """
    results: list[StageResult] = []
    started = time.time()

    log.info("=" * 68)
    log.info("VOYAGE ANALYTICS PIPELINE  |  stages: %s", " -> ".join(stage_names))
    log.info("project root: %s", PATHS.root)
    log.info("output format: %s%s", "parquet" if opts.parquet else "csv",
             "  (quick mode)" if opts.quick else "")
    log.info("=" * 68)

    failed = False
    for i, name in enumerate(stage_names, 1):
        stage = STAGE_MAP[name]

        if failed:
            results.append(StageResult(name, "skipped", detail="upstream stage failed"))
            log.warning("[%d/%d] %-10s SKIPPED (upstream failure)", i, len(stage_names), name)
            continue

        log.info("")
        log.info("[%d/%d] %s | %s", i, len(stage_names), name.upper(), stage.description)
        log.info("-" * 68)

        t0 = time.time()
        try:
            detail = stage.run(opts) or ""
            elapsed = time.time() - t0
            results.append(StageResult(name, "ok", elapsed, str(detail)))
            log.info("[%d/%d] %-10s OK in %.1fs | %s", i, len(stage_names), name, elapsed, detail)
        except Exception as exc:                      # noqa: BLE001 - report and stop
            elapsed = time.time() - t0
            results.append(StageResult(name, "failed", elapsed, error=f"{type(exc).__name__}: {exc}"))
            log.error("[%d/%d] %-10s FAILED after %.1fs", i, len(stage_names), name, elapsed)
            log.error("%s", traceback.format_exc())
            failed = True

    _summary(results, time.time() - started)
    return results


def _summary(results: list[StageResult], total: float) -> None:
    log.info("")
    log.info("=" * 68)
    log.info("PIPELINE SUMMARY")
    log.info("-" * 68)
    for r in results:
        mark = {"ok": "OK  ", "failed": "FAIL", "skipped": "SKIP"}[r.status]
        line = f"  {mark}  {r.name:<10} {r.seconds:>6.1f}s"
        if r.detail:
            line += f"  {r.detail}"
        if r.error:
            line += f"  {r.error}"
        log.info(line)
    log.info("-" * 68)

    n_ok = sum(r.status == "ok" for r in results)
    log.info("%d/%d stages succeeded in %.1fs total", n_ok, len(results), total)
    if n_ok == len(results):
        log.info("artifacts: %s | %s | %s",
                 PATHS.interim.relative_to(PATHS.root),
                 PATHS.processed.relative_to(PATHS.root),
                 "models/")
    log.info("=" * 68)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the Voyage Analytics pipeline end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(s.name for s in STAGES),
    )
    p.add_argument("--stages", nargs="+", metavar="STAGE",
                   help="run only these stages, in the order given")
    p.add_argument("--skip", nargs="+", metavar="STAGE", default=[],
                   help="run everything except these stages")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke run (training uses a 40k-row sample)")
    p.add_argument("--parquet", action="store_true",
                   help="write parquet instead of the default CSV")
    p.add_argument("--list", action="store_true",
                   help="list available stages and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.list:
        print("Available stages (dependency order):\n")
        for s in STAGES:
            print(f"  {s.name:<10} {s.description}")
        return 0

    selected = args.stages or [s.name for s in STAGES]

    unknown = [n for n in list(selected) + list(args.skip) if n not in STAGE_MAP]
    if unknown:
        print(f"error: unknown stage(s): {unknown}", file=sys.stderr)
        print(f"choose from: {[s.name for s in STAGES]}", file=sys.stderr)
        return 2

    selected = [n for n in selected if n not in args.skip]
    if not selected:
        print("error: no stages left to run", file=sys.stderr)
        return 2

    opts = Options(parquet=args.parquet, quick=args.quick)
    results = run_pipeline(selected, opts)

    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
