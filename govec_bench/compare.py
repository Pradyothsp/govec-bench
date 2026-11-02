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

from rich.text import Text

from govec_bench.theme import palette

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


class Setup(NamedTuple):
    # What one run ran as; None where its file didn't record it.
    image_id: str | None
    env: str | None  # settings as JSON
    memory_limit: int | None


@dataclass(slots=True)
class ArmData:
    name: str
    database: str
    runs: int = 0
    failed: int = 0
    setups: set[Setup] = field(default_factory=set)  # one entry if every run ran the same thing
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
        arm.setups.add(Setup(row["image_id"], row["env"], row["memory_limit"]))
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
        "SELECT r.arm, r.dataset, e.database, e.image_id, e.env, e.memory_limit, e.error FROM runs r "
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
        "SELECT e.database, e.image_id, e.env, e.memory_limit, e.error FROM runs r "  # noqa: S608 -- fixed text
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
    # Only when it's known: files from before results recorded what ran have no image ID or
    # memory limit.
    (setup,) = arm.setups if len(arm.setups) == 1 else (None,)
    known = setup is not None and setup.image_id is not None and setup.memory_limit is not None
    return known and arm.setups == other.setups


def _summary_of(arm: ArmData | None, metric: Metric) -> Summary | None:
    values = arm.values.get(metric.name) if arm else None
    return summarize(values) if values else None


class Change(NamedTuple):
    moved: float | None  # the median's relative change; None against a median of 0
    verdict: str


def change_of(control: Summary | None, other: Summary | None, metric: Metric) -> Change | None:
    if control is None or other is None:
        return None
    return Change(change(control, other), verdict(control, other, metric.better))


class Row(NamedTuple):
    # One metric of a comparison: a summary per column, and a change for each compared column.
    metric: Metric
    summaries: dict[str, Summary | None]  # by arm, or "baseline" and "session"
    changes: dict[str, Change | None]  # by the arm compared with the control, or "session"


def session_rows(arms: dict[str, ArmData], control: str, metrics: Sequence[Metric]) -> list[Row]:
    rows = []
    for metric in metrics:
        summaries = {name: _summary_of(arm, metric) for name, arm in arms.items()}
        if any(summaries.values()):
            changes = {
                name: change_of(summaries[control], summary, metric)
                for name, summary in summaries.items()
                if name != control
            }
            rows.append(Row(metric, summaries, changes))
    return rows


class BaselineMatch(NamedTuple):
    arm: str
    baseline: ArmData | None  # None when the baseline has nothing for this arm
    same_setup: bool  # same image and settings in both: the change is the sessions, not the code
    rows: list[Row]


def baseline_matches(
    arms: dict[str, ArmData], reference: dict[str, ArmData], *, by_database: bool, metrics: Sequence[Metric]
) -> list[BaselineMatch]:
    matches = []
    for arm in arms.values():
        match = reference.get(arm.database if by_database else arm.name)
        rows = []
        for metric in metrics if match else ():
            before, after = _summary_of(match, metric), _summary_of(arm, metric)
            if before is not None or after is not None:
                rows.append(
                    Row(metric, {"baseline": before, "session": after}, {"session": change_of(before, after, metric)})
                )
        matches.append(BaselineMatch(arm.name, match, match is not None and same_setup(arm, match), rows))
    return matches


def with_baseline(rows: Sequence[Row], matches: Sequence[BaselineMatch]) -> list[Row]:
    # The session's rows with a change per arm against the baseline, after the session's own.
    against = {(match.arm, row.metric.name): row.changes["session"] for match in matches for row in match.rows}
    return [
        row._replace(
            changes={
                **row.changes,
                **{baseline_column(match.arm): against.get((match.arm, row.metric.name)) for match in matches},
            }
        )
        for row in rows
    ]


def baseline_column(arm: str) -> str:
    return f"{arm} vs baseline"


class Baseline(NamedTuple):
    name: str
    arms: dict[str, ArmData]
    by_database: bool  # the published runs are matched to arms by database, a session by arm name


def load_baseline(db: sqlite3.Connection, baseline: str, dataset: str) -> Baseline:
    if baseline == PUBLISHED:
        return Baseline(f"the published runs ({dataset})", load_published(db, dataset), by_database=True)

    key = resolve_session(db, baseline)
    _, arms = load_session(db, key)
    return Baseline(f"{key.experiment}/{key.session}", arms, by_database=False)


# --- Text --------------------------------------------------------------------------------


def change_text(item: Change | None, *, width: int = 7) -> Text:
    # width: the change right-aligned to it, so a text table's verdicts line up; a DataTable's
    # columns line up by themselves.
    if item is None:
        return Text("")
    moved = f"{item.moved:+.1%}" if item.moved is not None else ""
    text = f"{moved:>{width}}  {item.verdict}" if width else f"{moved}  {item.verdict}".lstrip()
    return Text(text, style=palette().verdict(item.verdict))


def label_of(metric: Metric) -> str:
    unit = {"ms": " (ms)", "bytes": " (MB)"}.get(metric.unit, "")
    return metric.label + unit


def label_text(metric: Metric) -> Text:
    # A metric's name, a sub-row ("  k=50") dimmed, with ! where it needs a caution.
    text = Text(label_of(metric), style="dim" if metric.label != metric.label.lstrip() else "")
    if metric.caution:
        text.append(" !", style=f"bold {palette().caution}")
    return text


def column_text(name: str, arms: Sequence[str]) -> Text:
    # An arm's column in its colour, and a change column ("new vs old") in its arm's.
    arm = name.split(" vs ", 1)[0].removesuffix(" (baseline)")
    return Text(name, style=f"bold {palette().arm(arms, arm)}" if arm in arms else "bold")


def _table(header: Sequence[Text], rows: Sequence[Sequence[Text]]) -> list[Text]:
    widths = [max(row[column].cell_len for row in [header, *rows]) for column in range(len(header))]
    lines = []
    for row in [header, *rows]:
        line = Text("  ").join(cell + " " * (width - cell.cell_len) for cell, width in zip(row, widths, strict=True))
        line.rstrip()
        lines.append(line)
    return lines


def arms_line(arms: dict[str, ArmData]) -> Text:
    names, line = list(arms), Text("arms: ")
    for index, arm in enumerate(arms.values()):
        if index:
            line.append("; ")
        line.append(arm.name, style=f"bold {palette().arm(names, arm.name)}")
        failed = f", {arm.failed} failed partway" if arm.failed else ""
        varied = ", setup varied between runs" if len(arm.setups) > 1 else ""
        limits = {setup.memory_limit for setup in arm.setups if setup.memory_limit is not None}
        memory = f", {limits.pop() / 1024**2:g} MiB memory" if len(limits) == 1 else ""
        line.append(f" ({arm.database}{memory}, {arm.runs} runs")
        line.append(failed, style=palette().bad)
        line.append(varied, style=palette().caution)
        line.append(")")
    return line


def _text_rows(rows: Sequence[Row]) -> list[list[Text]]:
    return [
        [
            label_text(row.metric),
            *(Text(summary_text(summary, row.metric.unit)) for summary in row.summaries.values()),
            *(change_text(item) for item in row.changes.values()),
        ]
        for row in rows
    ]


def _cautions(rows: Sequence[Row]) -> list[Text]:
    return [
        Text(f"! {row.metric.label.strip()}: {row.metric.caution}", style=palette().caution)
        for row in rows
        if row.metric.caution
    ]


def session_report(arms: dict[str, ArmData], control: str, metrics: Sequence[Metric]) -> list[Text]:
    rows = session_rows(arms, control, metrics)
    names = list(arms)
    header = [
        Text(""),
        *(
            column_text(name, names)
            for name in [*names, *(f"{name} vs {control}" for name in names if name != control)]
        ),
    ]
    return [*_table(header, _text_rows(rows)), *_cautions(rows)]


SAME_SETUP_NOTE = "same image and settings: this change is the sessions, not the code"


def baseline_report(arms: dict[str, ArmData], baseline: Baseline, metrics: Sequence[Metric]) -> list[Text]:
    lines: list[Text] = []
    names = list(arms)
    for match in baseline_matches(arms, baseline.arms, by_database=baseline.by_database, metrics=metrics):
        if match.baseline is None:
            lines.append(Text(f"{match.arm}: nothing to compare in the baseline", style="dim"))
            continue

        title = Text(match.arm, style=f"bold {palette().arm(names, match.arm)}")
        if match.same_setup:
            title.append(f" ({SAME_SETUP_NOTE})", style=palette().info)
        header = [
            Text(""),
            Text(f"baseline ({match.baseline.runs})", style="bold"),
            Text(f"session ({arms[match.arm].runs})", style="bold"),
            Text(""),
        ]
        lines.extend([title, *_table(header, _text_rows(match.rows)), Text("")])
    return lines


def compare(
    db: sqlite3.Connection,
    session: str,
    *,
    control: str | None = None,
    baseline: str | None = None,
    patterns: Sequence[str] = (),
    every: bool = False,
) -> list[Text]:
    key = resolve_session(db, session)
    dataset, arms = load_session(db, key)
    control = control or next(iter(arms))
    if control not in arms:
        msg = f"no arm {control!r} in {key.experiment}/{key.session}; arms: {list(arms)}"
        raise SystemExit(msg)

    metrics = select_metrics(arms.values(), patterns, every=every)
    title = Text.assemble(
        (f"{key.experiment}/{key.session}", "bold"),
        f": {dataset}, control ",
        (control, f"bold {palette().arm(list(arms), control)}"),
    )
    lines = [title, arms_line(arms), Text("")]
    lines.extend(session_report(arms, control, metrics))

    if baseline:
        reference = load_baseline(db, baseline, dataset)
        lines.extend([Text(""), Text(f"Against {reference.name}:", style="bold"), Text("")])
        lines.extend(baseline_report(arms, reference, metrics))

    return lines


# --- For browsing: every session, one metric's runs, its history ------------------------


class SessionInfo(NamedTuple):
    key: SessionKey
    dataset: str
    runs: int
    failed: int
    started: str  # the first run's timestamp


def list_sessions(db: sqlite3.Connection) -> list[SessionInfo]:
    # Newest first.
    rows = db.execute(
        "SELECT r.experiment, r.session, r.dataset, count(DISTINCT r.id), count(e.error), min(r.timestamp) "
        "FROM runs r JOIN entries e ON e.run_id = r.id WHERE r.experiment IS NOT NULL "
        "GROUP BY r.experiment, r.session ORDER BY min(r.timestamp) DESC"
    )
    return [
        SessionInfo(SessionKey(experiment, session), dataset, runs, failed, started or "")
        for experiment, session, dataset, runs, failed, started in rows
    ]


class RunValue(NamedTuple):
    arm: str
    round: int
    position: int
    value: float
    label: str  # the results file's name


def run_values(db: sqlite3.Connection, key: SessionKey, metric: str) -> list[RunValue]:
    rows = db.execute(
        "SELECT r.arm, r.round, r.position, m.value, r.label FROM runs r JOIN metrics m ON m.run_id = r.id "
        "WHERE r.experiment = ? AND r.session = ? AND m.metric = ? ORDER BY r.round, r.position",
        (*key, metric),
    )
    return [RunValue(*row) for row in rows]


class HistoryPoint(NamedTuple):
    session: str
    arm: str
    summary: Summary


def history(db: sqlite3.Connection, experiment: str, metric: str) -> list[HistoryPoint]:
    # Each arm's median and range in every session of an experiment, oldest first.
    rows = db.execute(
        "SELECT r.session, r.arm, m.value FROM runs r JOIN metrics m ON m.run_id = r.id "
        "WHERE r.experiment = ? AND m.metric = ? ORDER BY r.session, r.round, r.position",
        (experiment, metric),
    )
    values: dict[tuple[str, str], list[float]] = {}
    for session, arm, value in rows:
        values.setdefault((session, arm), []).append(value)
    return [HistoryPoint(session, arm, summarize(points)) for (session, arm), points in values.items()]
