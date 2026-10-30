import json
import sqlite3
import statistics
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from govec_bench.results_file import ResultsFile, RunTag, StartedAs
from govec_bench.store import import_paths

RESULTS_DIR = Path("results")

# The index over every results file (store.py); rebuilt with `task results:import`.
RESULTS_DB = RESULTS_DIR / "bench.db"


@dataclass(frozen=True, slots=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p99_ms: float


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    return LatencyStats(
        mean_ms=statistics.mean(latencies_ms),
        p50_ms=statistics.median(latencies_ms),
        p99_ms=statistics.quantiles(latencies_ms, n=100)[98],
    )


@dataclass(frozen=True, slots=True)
class RecallStats:
    mean: float
    min: float
    max: float


def compute_recall_stats(recalls: list[float]) -> RecallStats:
    return RecallStats(
        mean=statistics.mean(recalls),
        min=min(recalls),
        max=max(recalls),
    )


def write_results(
    benchmark: str,
    dataset: str,
    results: dict[str, object],
    setup: Mapping[str, StartedAs] | None = None,
    *,
    experiment: RunTag | None = None,
    steps: list[str] | None = None,
    directory: Path = RESULTS_DIR,
    name: str | None = None,
) -> Path:
    # setup: what each database ran as (image, GoVec commit, settings), from common.setup_of().
    # experiment: which experiment, session, round and arm this run was, when bench run made it.
    # steps: the pipeline steps that ran, when a run chose some (--steps, an experiment's steps).
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    payload: ResultsFile = {
        "benchmark": benchmark,
        "timestamp": timestamp.isoformat(),
        "dataset": dataset,
        "setup": dict(setup or {}),
        "results": results,
    }
    if experiment is not None:
        payload["experiment"] = experiment
    if steps is not None:
        payload["steps"] = steps

    path = directory / f"{name or benchmark + '_' + timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2))

    # The file is the record; failing to index it loses nothing, `task results:import` retries.
    try:
        import_paths(RESULTS_DB, [path])
    except sqlite3.Error as exc:
        warnings.warn(f"wrote {path} but couldn't add it to {RESULTS_DB}: {exc}", stacklevel=2)

    return path
