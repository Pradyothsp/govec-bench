"""Every results file as rows in one SQLite database, so runs can be compared by query.

The JSON files stay the record: each benchmark writes one, the published ones are committed, and
older sessions live in govec-notes. The database is an index over them, rebuilt by importing the
files again, so deleting it loses nothing. A run is keyed by a hash of its file, so the same file
imported from two places (results/ and a copy in govec-notes) is one run.

Tables:
  runs     one per results file: benchmark, dataset, when, where it was imported from, label,
           and for a run bench run made, its experiment, session, round, arm and position
  entries  one per database in a run: what it ran as (image, GoVec commit, settings), and the
           error if it failed partway
  metrics  one per number a database recorded, under a dotted name: queries.k_10.latency.mean_ms
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

from govec_bench.results_file import ResultsFile

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    benchmark TEXT NOT NULL,
    dataset TEXT,
    timestamp TEXT,
    source TEXT NOT NULL,
    label TEXT NOT NULL,
    experiment TEXT,
    session TEXT,
    round INTEGER,
    arm TEXT,
    position INTEGER
);
CREATE TABLE IF NOT EXISTS entries (
    run_id TEXT NOT NULL REFERENCES runs(id),
    database TEXT NOT NULL,
    service TEXT,
    image TEXT,
    image_id TEXT,
    govec_commit TEXT,
    env TEXT,
    error TEXT,
    PRIMARY KEY (run_id, database)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(id),
    database TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, database, metric)
);
CREATE INDEX IF NOT EXISTS metrics_by_name ON metrics (metric, database);
"""

# Columns added to runs after its first version, so an index built before them still opens.
_ADDED_RUN_COLUMNS = {"experiment": "TEXT", "session": "TEXT", "round": "INTEGER", "arm": "TEXT", "position": "INTEGER"}


class Imported(NamedTuple):
    path: Path
    run_id: str
    new: bool


def _list_key(item: dict[str, object]) -> str | None:
    # Lists the benchmarks write, keyed so a metric name says which item it is.
    if "ef" in item and "k" in item:  # the ef sweep's points
        return f"ef_{item['ef']}.k_{item['k']}"
    if "t_s" in item:  # the memory benchmark's RAM samples, by seconds after the load
        return f"t_{item['t_s']}s"

    return None


def flatten(value: object, prefix: str = "") -> Iterator[tuple[str, float]]:
    # Every number, under the dotted path to it. Strings and booleans aren't measurements; a list
    # is kept only if its items say what they are (see _list_key).
    if isinstance(value, bool):
        return

    if isinstance(value, int | float):
        yield prefix, float(value)
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
        return

    if isinstance(value, list):
        for item in value:
            key = _list_key(item) if isinstance(item, dict) else None
            if key is not None:
                yield from flatten({k: v for k, v in item.items() if k not in {"ef", "k", "t_s"}}, f"{prefix}.{key}")


def _memory_added(memory: Mapping[str, object], prefix: str) -> Iterator[tuple[str, float]]:
    # RAM added by the load, at the last sample: what the published tables compare. The samples
    # hold totals; the empty container's footprint is the baseline.
    baseline, series = memory.get("baseline"), memory.get("ram_series")
    if not isinstance(baseline, dict) or not isinstance(series, list) or not series:
        return

    last = series[-1]
    for sample_key, baseline_key, name in (
        ("ram_bytes", "ram_bytes", "ram_added_bytes"),
        ("process_bytes", "ram_process_bytes", "process_added_bytes"),
        ("file_cache_bytes", "ram_file_cache_bytes", "file_cache_added_bytes"),
    ):
        if sample_key in last and baseline_key in baseline:
            yield f"{prefix}{name}", float(last[sample_key] - baseline[baseline_key])


def metrics_of(result: Mapping[str, object]) -> dict[str, float]:
    metrics = dict(flatten(result))

    memory = result.get("memory")
    if isinstance(memory, dict):  # the pipeline nests it
        metrics.update(_memory_added(memory, "memory."))
    elif "ram_series" in result:  # the memory benchmark's own file
        metrics.update(_memory_added(result, ""))

    return metrics


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)

    present = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
    for column, kind in _ADDED_RUN_COLUMNS.items():
        if column not in present:
            db.execute(f"ALTER TABLE runs ADD COLUMN {column} {kind}")

    return db


def import_file(db: sqlite3.Connection, path: Path) -> Imported:
    content = path.read_bytes()
    run_id = hashlib.sha256(content).hexdigest()[:16]
    if db.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone():
        return Imported(path, run_id, new=False)

    payload: ResultsFile = json.loads(content)
    setup = payload.get("setup") or {}
    experiment = payload.get("experiment")
    tag = (
        (experiment["name"], experiment["session"], experiment["round"], experiment["arm"], experiment["position"])
        if experiment
        else (None,) * 5
    )

    with db:
        db.execute(
            "INSERT INTO runs (id, benchmark, dataset, timestamp, source, label, experiment, session, round, arm, position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                payload["benchmark"],
                payload.get("dataset"),
                payload.get("timestamp"),
                str(path),
                path.stem,
                *tag,
            ),
        )

        for database, result in payload["results"].items():
            # The ef sweep writes series under labels (govec@efc200); each names its database.
            service = result.get("service") if isinstance(result, dict) else None
            ran_as = setup.get(database) or (setup.get(service) if isinstance(service, str) else None)
            error = result.get("error") if isinstance(result, dict) else None
            image = (
                (
                    ran_as["service"],
                    ran_as["image"],
                    ran_as["image_id"],
                    ran_as["govec_commit"],
                    json.dumps(ran_as["env"]),
                )
                if ran_as
                else (None,) * 5
            )
            db.execute(
                "INSERT INTO entries (run_id, database, service, image, image_id, govec_commit, env, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    database,
                    *image,
                    error,
                ),
            )

            if isinstance(result, dict):
                db.executemany(
                    "INSERT INTO metrics (run_id, database, metric, value) VALUES (?, ?, ?, ?)",
                    [(run_id, database, metric, value) for metric, value in metrics_of(result).items()],
                )

    return Imported(path, run_id, new=True)


def results_files(paths: Iterable[Path]) -> Iterator[Path]:
    # A results file is any JSON with "benchmark" and "results": folders are searched for them.
    for path in paths:
        candidates = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_bytes())
            except OSError, ValueError:
                continue
            if isinstance(payload, dict) and "benchmark" in payload and isinstance(payload.get("results"), dict):
                yield candidate


def import_paths(db_path: Path, paths: Iterable[Path]) -> list[Imported]:
    with closing(connect(db_path)) as db:
        return [import_file(db, path) for path in results_files(paths)]
