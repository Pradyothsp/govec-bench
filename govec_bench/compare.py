"""What an experiment session found, metric by metric: `bench compare <session>`.

Each arm's median and range over its rounds, the change against the control arm (the one that ran
first in round 1, unless --control names another), and a verdict that only says "better" when
every run of the arm beat every run of the control: with three to five runs a session, medians
alone moved by tens of percent between nights (README, run-to-run variation).

--baseline puts the session next to an earlier one, or next to the published runs. An arm that
ran the same image and settings in both is marked: its change is what moved between the sessions,
not the code, and every other change against the baseline includes as much.
"""

import sqlite3
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal, NamedTuple

Unit = Literal["ms", "bytes", "recall", ""]

# Which way is better; "neither" for a size that's neither good nor bad by itself (file cache).
Better = Literal["lower", "higher", "neither"]

PUBLISHED = "published"

# A change smaller than this is "level", even when every run is on one side: a disk size that
# differs by a few bytes between runs isn't a finding.
LEVEL = 0.01


class Metric(NamedTuple):
    name: str  # as stored: a dotted path into the results file
    label: str
    unit: Unit
    better: Better
    caution: str = ""


_DOCKER_STATS = (
    "docker stats leaves out inactive file pages, so memory-mapped data can move it without the memory "
    "changing: read process and file cache"
)

# The rows the published tables and the session notes compare.
HEADLINE: tuple[Metric, ...] = (
    Metric("queries_grpc.k_10.latency.mean_ms", "Query over gRPC, k=10", "ms", "lower"),
    Metric("queries_grpc.k_50.latency.mean_ms", "  k=50", "ms", "lower"),
    Metric("queries_grpc.k_10.latency.p99_ms", "  k=10 p99", "ms", "lower"),
    Metric("queries.k_10.latency.mean_ms", "Query over REST, k=10", "ms", "lower"),
    Metric("queries.k_10.latency.p99_ms", "  k=10 p99", "ms", "lower"),
    Metric("batch_insert.per_vector_ms", "Batch insert, per vector", "ms", "lower"),
    Metric("single_insert.per_vector_ms", "Single insert, per vector", "ms", "lower"),
    Metric("cold_start_loaded.p50_ms", "Restart with data, p50", "ms", "lower"),
    Metric("cold_start_empty.p50_ms", "Restart empty, p50", "ms", "lower"),
    Metric("memory.ram_added_bytes", "RAM added, docker stats", "bytes", "lower", _DOCKER_STATS),
    Metric("memory.process_added_bytes", "  process", "bytes", "lower"),
    Metric("memory.file_cache_added_bytes", "  file cache", "bytes", "neither"),
    Metric("queries.k_10.recall.mean", "Recall@10", "recall", "higher"),
    Metric("queries.k_50.recall.mean", "Recall@50", "recall", "higher"),
    Metric("memory.delta_disk_allocated_bytes", "Disk allocated", "bytes", "lower"),
)

# Stored, but not measurements of the database: where it ran in the order, the empty container.
_NOT_COMPARED = ("run_position", "memory.baseline.*", "memory.ram_series.*")


def metric_of(name: str) -> Metric:
    # A metric outside HEADLINE, labelled with its stored name and read from its suffix.
    for metric in HEADLINE:
        if metric.name == name:
            return metric

    if "recall" in name:
        return Metric(name, name, "recall", "higher")
    if "_bytes" in name:  # also delta_ram_bytes_peak, delta_ram_bytes_last_sample
        return Metric(name, name, "bytes", "lower", _DOCKER_STATS if "ram_bytes" in name else "")
    if name.endswith(("_ms", "_s")):
        return Metric(name, name, "ms" if name.endswith("_ms") else "", "lower")
    return Metric(name, name, "", "neither")


class SessionKey(NamedTuple):
    experiment: str
    session: str


@dataclass(slots=True)
class ArmData:
    name: str
    database: str
    runs: int = 0
    failed: int = 0
    # (image ID, settings as JSON) of each run: one entry if every run ran the same thing.
    setups: set[tuple[str | None, str | None]] = field(default_factory=set)
    values: dict[str, list[float]] = field(default_factory=dict)  # by metric name, one per run


class Summary(NamedTuple):
    median: float
    low: float
    high: float
    n: int


def summarize(values: Sequence[float]) -> Summary:
    return Summary(statistics.median(values), min(values), max(values), len(values))


def change(control: Summary, other: Summary) -> float | None:
    if control.median == 0:
        return None
    return other.median / control.median - 1


def verdict(control: Summary, other: Summary, better: Better) -> str:
    moved = change(control, other)
    if moved is not None and abs(moved) < LEVEL:
        return "level"
    if control.n < 2 or other.n < 2:  # noqa: PLR2004 -- a range needs two runs
        return "one run"

    if other.low > control.high:
        direction = "higher"
    elif other.high < control.low:
        direction = "lower"
    else:
        return "overlaps"

    if better == "neither":
        return f"every run {direction}"
    return f"every run {'better' if direction == better else 'worse'}"


# --- Reading the index -------------------------------------------------------------------


def resolve_session(db: sqlite3.Connection, text: str) -> SessionKey:
    # An experiment's name (its latest session), experiment/session, or a session's folder.
    path = Path(text)
    if path.is_dir() and (path / "session.json").exists():
        return SessionKey(path.parent.name, path.name)

    experiment, _, session = text.rstrip("/").partition("/")
    if session:
        found = db.execute(
            "SELECT 1 FROM runs WHERE experiment = ? AND session = ? LIMIT 1", (experiment, session)
        ).fetchone()
    else:
        found = db.execute("SELECT max(session) FROM runs WHERE experiment = ?", (experiment,)).fetchone()
        session = found[0] if found else None

    if not found or not session:
        known = [row[0] for row in db.execute("SELECT DISTINCT experiment FROM runs WHERE experiment IS NOT NULL")]
        msg = f"no session {text!r} in the index; experiments with sessions: {sorted(known)}"
        raise SystemExit(msg)

    return SessionKey(experiment, session)


def _collect(rows: Iterable[sqlite3.Row], key: str) -> dict[str, ArmData]:
    # rows: one per run and database, ordered as the arms should be listed.
    arms: dict[str, ArmData] = {}
    for row in rows:
        arm = arms.setdefault(row[key], ArmData(name=row[key], database=row["database"]))
        arm.runs += 1
        arm.failed += row["error"] is not None
        arm.setups.add((row["image_id"], row["env"]))
    return arms


def _add_metrics(
    db: sqlite3.Connection, arms: dict[str, ArmData], key: str, where: str, params: Sequence[object]
) -> None:
    query = (
        f"SELECT {key} AS name, m.metric, m.value FROM runs r "  # noqa: S608 -- key and where are fixed by callers
        f"JOIN metrics m ON m.run_id = r.id {where} ORDER BY r.timestamp"
    )
    for name, metric, value in db.execute(query, params):
        if name in arms:
            arms[name].values.setdefault(metric, []).append(value)


def load_session(db: sqlite3.Connection, key: SessionKey) -> tuple[str, dict[str, ArmData]]:
    # The session's dataset, and its arms in the order they ran in round 1.
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT r.arm, r.dataset, e.database, e.image_id, e.env, e.error FROM runs r "
        "JOIN entries e ON e.run_id = r.id WHERE r.experiment = ? AND r.session = ? ORDER BY r.round, r.position",
        key,
    ).fetchall()
    if not rows:
        msg = f"session {key.experiment}/{key.session} has no runs in the index"
        raise SystemExit(msg)

    arms = _collect(rows, "arm")
    _add_metrics(db, arms, "r.arm", "WHERE r.experiment = ? AND r.session = ?", key)
    return str(rows[0]["dataset"]), arms


def load_published(db: sqlite3.Connection, dataset: str) -> dict[str, ArmData]:
    # The README's runs on the same dataset, by database: what an arm of that database compares to.
    db.row_factory = sqlite3.Row
    where = "WHERE r.benchmark = 'pipeline' AND r.dataset = ? AND r.source LIKE '%published%'"
    rows = db.execute(
        "SELECT e.database, e.image_id, e.env, e.error FROM runs r "  # noqa: S608 -- fixed text
        f"JOIN entries e ON e.run_id = r.id {where} ORDER BY r.timestamp",
        (dataset,),
    ).fetchall()
    arms = _collect(rows, "database")
    _add_metrics(db, arms, "m.database", where, (dataset,))
    return arms


# --- The report --------------------------------------------------------------------------


def _number(value: float) -> str:
    size = abs(value)
    if size >= 100:  # noqa: PLR2004
        return f"{value:,.0f}"
    if size >= 10:  # noqa: PLR2004
        return f"{value:.1f}"
    return f"{value:.2f}"


def formatted(value: float, unit: Unit) -> str:
    if unit == "recall":
        return f"{value:.3f}"
    if unit == "bytes":
        return _number(value / 1000**2)
    return _number(value)


def summary_text(summary: Summary | None, unit: Unit) -> str:
    if summary is None:
        return "-"
    if summary.n == 1:
        return formatted(summary.median, unit)
    return f"{formatted(summary.median, unit)} ({formatted(summary.low, unit)}–{formatted(summary.high, unit)})"  # noqa: RUF001 -- a range


def _label(metric: Metric) -> str:
    unit = {"ms": " (ms)", "bytes": " (MB)"}.get(metric.unit, "")
    return metric.label + unit


def select_metrics(arms: Iterable[ArmData], patterns: Sequence[str], *, every: bool) -> list[Metric]:
    if not patterns and not every:
        return list(HEADLINE)

    # Headline metrics in their order (a sub-row stays under its row), then the rest by name.
    stored = {name for arm in arms for name in arm.values}
    headline = [metric.name for metric in HEADLINE]
    names = [
        name
        for name in [*(name for name in headline if name in stored), *sorted(stored - set(headline))]
        if not any(fnmatch(name, skip) for skip in _NOT_COMPARED)
        and (every or any(fnmatch(name, pattern) for pattern in patterns))
    ]
    return [metric_of(name) for name in names]


def same_setup(arm: ArmData, other: ArmData) -> bool:
    # Only when it's known: files from before results recorded what ran have no image ID.
    (setup,) = arm.setups if len(arm.setups) == 1 else (None,)
    return setup is not None and setup[0] is not None and arm.setups == other.setups


def _summary_of(arm: ArmData | None, metric: Metric) -> Summary | None:
    values = arm.values.get(metric.name) if arm else None
    return summarize(values) if values else None


def _change_text(control: Summary | None, other: Summary | None, metric: Metric) -> str:
    if control is None or other is None:
        return ""
    moved = change(control, other)
    moved_text = f"{moved:+.1%}" if moved is not None else ""
    return f"{moved_text:>7}  {verdict(control, other, metric.better)}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in [header, *rows]
    ]


def arms_line(arms: dict[str, ArmData]) -> str:
    parts = []
    for arm in arms.values():
        failed = f", {arm.failed} failed partway" if arm.failed else ""
        varied = ", setup varied between runs" if len(arm.setups) > 1 else ""
        parts.append(f"{arm.name} ({arm.database}, {arm.runs} runs{failed}{varied})")
    return "arms: " + "; ".join(parts)


def session_report(arms: dict[str, ArmData], control: str, metrics: Sequence[Metric]) -> list[str]:
    others = [name for name in arms if name != control]
    header = ["", *arms, *(f"{name} vs {control}" for name in others)]
    rows = []
    cautions = []
    for metric in metrics:
        summaries = {name: _summary_of(arm, metric) for name, arm in arms.items()}
        if not any(summaries.values()):
            continue
        label = _label(metric) + (" !" if metric.caution else "")
        rows.append(
            [
                label,
                *(summary_text(summary, metric.unit) for summary in summaries.values()),
                *(_change_text(summaries[control], summaries[name], metric) for name in others),
            ]
        )
        if metric.caution:
            cautions.append(f"! {metric.label.strip()}: {metric.caution}")
    return [*_table(header, rows), *cautions]


def baseline_report(
    arms: dict[str, ArmData], baseline: dict[str, ArmData], *, by_database: bool, metrics: Sequence[Metric]
) -> list[str]:
    lines = []
    for arm in arms.values():
        match = baseline.get(arm.database if by_database else arm.name)
        if match is None:
            lines.append(f"{arm.name}: nothing to compare in the baseline")
            continue

        same = same_setup(arm, match)
        note = " (same image and settings: this change is the sessions, not the code)" if same else ""
        rows = []
        for metric in metrics:
            before, after = _summary_of(match, metric), _summary_of(arm, metric)
            if before is None and after is None:
                continue
            rows.append(
                [
                    _label(metric),
                    summary_text(before, metric.unit),
                    summary_text(after, metric.unit),
                    _change_text(before, after, metric),
                ]
            )
        lines.extend(
            [f"{arm.name}{note}", *_table(["", f"baseline ({match.runs})", f"session ({arm.runs})", ""], rows), ""]
        )
    return lines


def compare(
    db: sqlite3.Connection,
    session: str,
    *,
    control: str | None = None,
    baseline: str | None = None,
    patterns: Sequence[str] = (),
    every: bool = False,
) -> list[str]:
    key = resolve_session(db, session)
    dataset, arms = load_session(db, key)
    control = control or next(iter(arms))
    if control not in arms:
        msg = f"no arm {control!r} in {key.experiment}/{key.session}; arms: {list(arms)}"
        raise SystemExit(msg)

    metrics = select_metrics(arms.values(), patterns, every=every)
    lines = [f"{key.experiment}/{key.session}: {dataset}, control {control}", arms_line(arms), ""]
    lines.extend(session_report(arms, control, metrics))

    if baseline:
        by_database = baseline == PUBLISHED
        if by_database:
            reference = load_published(db, dataset)
            name = f"the published runs ({dataset})"
        else:
            base_key = resolve_session(db, baseline)
            _, reference = load_session(db, base_key)
            name = f"{base_key.experiment}/{base_key.session}"
        lines.extend(["", f"Against {name}:", ""])
        lines.extend(baseline_report(arms, reference, by_database=by_database, metrics=metrics))

    return lines
