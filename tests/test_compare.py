import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from govec_bench.compare import (
    HEADLINE,
    ArmData,
    Better,
    Summary,
    compare,
    metric_of,
    resolve_session,
    select_metrics,
    verdict,
)
from govec_bench.results_file import ResultsFile, StartedAs
from govec_bench.store import connect, import_file

HEAD_SETUP: StartedAs = {
    "service": "govec",
    "image": "govec-bench/govec:450295fd1adf",
    "image_id": "sha256:4ee6",
    "govec_commit": "450295fd1adf",
    "env": {"GOVEC_ENABLE_MMAP": "true"},
}
OLD_SETUP: StartedAs = {
    **HEAD_SETUP,
    "image": "ghcr.io/pradyothsp/govec:0.2.1",
    "image_id": "sha256:3e7f",
    "govec_commit": None,
}


def _run_file(directory: Path, *, session: str, round_number: int, arm: str, position: int, restart_ms: float) -> Path:
    old = arm == "old"
    payload: ResultsFile = {
        "benchmark": "pipeline",
        "timestamp": f"2026-10-30T0{round_number}:0{position}:00+00:00",
        "dataset": "dbpedia100k",
        "setup": {"govec-mmap": OLD_SETUP if old else HEAD_SETUP},
        "results": {"govec-mmap": {"run_position": position, "cold_start_loaded": {"p50_ms": restart_ms}}},
        "experiment": {"name": "ab", "session": session, "round": round_number, "arm": arm, "position": position},
    }
    path = directory / f"{session}-r{round_number}-{arm}.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    # Two sessions of an old-against-new A/B: in the second the old arm, unchanged, is 10% slower.
    connection = connect(tmp_path / "bench.db")
    for session, slower in (("s1", 1.0), ("s2", 1.1)):
        for round_number, order in ((1, ("old", "new")), (2, ("new", "old")), (3, ("old", "new"))):
            for position, arm in enumerate(order):
                restart = (490 + round_number) * slower if arm == "old" else 550 + round_number
                import_file(
                    connection,
                    _run_file(
                        tmp_path,
                        session=session,
                        round_number=round_number,
                        arm=arm,
                        position=position,
                        restart_ms=restart,
                    ),
                )
    yield connection
    connection.close()


@pytest.mark.parametrize(
    ("control", "other", "better", "expected"),
    [
        (Summary(490, 489, 525, 3), Summary(553, 545, 566, 3), "lower", "every run worse"),
        (Summary(846, 831, 847, 3), Summary(226, 215, 231, 3), "lower", "every run better"),
        (Summary(3.75, 3.72, 4.03, 3), Summary(3.65, 3.59, 3.73, 3), "lower", "overlaps"),
        (Summary(0.96, 0.95, 0.96, 3), Summary(0.98, 0.97, 0.99, 3), "higher", "every run better"),
        (Summary(646, 646, 646, 3), Summary(632, 632, 632, 3), "neither", "every run lower"),
        (Summary(632.0, 632.0, 632.0, 3), Summary(632.1, 632.1, 632.1, 3), "lower", "level"),
        (Summary(490, 490, 490, 1), Summary(553, 553, 553, 1), "lower", "one run"),
    ],
)
def test_verdict__ranges__better_only_when_every_run_is(
    control: Summary, other: Summary, better: Better, expected: str
) -> None:
    # Arrange
    direction = better

    # Act
    result = verdict(control, other, direction)

    # Assert
    assert result == expected


def test_metric_of__names_outside_the_headline__direction_and_unit_from_the_suffix() -> None:
    # Arrange
    names = ["queries.k_5.recall.min", "batch_insert.total_s", "memory.delta_ram_bytes_peak"]

    # Act
    metrics = [metric_of(name) for name in names]

    # Assert
    assert [(metric.unit, metric.better) for metric in metrics] == [
        ("recall", "higher"),
        ("", "lower"),
        ("bytes", "lower"),
    ]
    assert metrics[2].caution  # a docker stats figure


def test_select_metrics__patterns__headline_order_then_by_name_without_bookkeeping() -> None:
    # Arrange
    names = {metric.name for metric in HEADLINE} | {"run_position", "memory.baseline.ram_bytes", "a.extra_ms"}
    arm = ArmData(name="a", database="govec", values={name: [1.0] for name in names})

    # Act
    metrics = select_metrics([arm], ["memory.*", "a.*"], every=False)

    # Assert
    assert [metric.name for metric in metrics] == [
        "memory.ram_added_bytes",
        "memory.process_added_bytes",
        "memory.file_cache_added_bytes",
        "memory.delta_disk_allocated_bytes",
        "a.extra_ms",
    ]


def test_resolve_session__experiment_name__its_latest_session(db: sqlite3.Connection) -> None:
    # Arrange
    name = "ab"

    # Act
    key = resolve_session(db, name)

    # Assert
    assert key == ("ab", "s2")


def test_resolve_session__unknown__names_the_experiments_there_are(db: sqlite3.Connection) -> None:
    # Arrange
    name = "nope"

    # Act
    with pytest.raises(SystemExit) as raised:
        resolve_session(db, name)

    # Assert
    assert "['ab']" in str(raised.value)


def test_compare__session__control_is_round_ones_first_arm_and_the_change_has_a_verdict(db: sqlite3.Connection) -> None:
    # Arrange
    session = "ab/s1"

    # Act
    lines = compare(db, session, patterns=["cold_start_loaded.p50_ms"])

    # Assert
    assert lines[0] == "ab/s1: dbpedia100k, control old"
    assert lines[1] == "arms: old (govec-mmap, 3 runs); new (govec-mmap, 3 runs)"
    row = next(line for line in lines if line.startswith("Restart with data"))
    assert row.split()[-5:] == ["(551–553)", "+12.2%", "every", "run", "worse"]  # noqa: RUF001 -- a range


def test_compare__baseline_session__an_arm_with_the_same_setup_is_marked_as_drift(db: sqlite3.Connection) -> None:
    # Arrange
    session, baseline = "ab/s2", "ab/s1"

    # Act
    lines = compare(db, session, baseline=baseline, patterns=["cold_start_loaded.p50_ms"])

    # Assert
    assert "Against ab/s1:" in lines
    assert "old (same image and settings: this change is the sessions, not the code)" in lines
    assert "new (same image and settings: this change is the sessions, not the code)" in lines
    old_row = lines[lines.index("old (same image and settings: this change is the sessions, not the code)") + 2]
    assert "+10.0%" in old_row
