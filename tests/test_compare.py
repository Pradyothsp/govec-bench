import sqlite3

import pytest

from govec_bench.compare import (
    HEADLINE,
    ArmData,
    Better,
    Setup,
    Summary,
    compare,
    metric_of,
    resolve_session,
    same_setup,
    select_metrics,
    verdict,
)


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
    assert lines[1] == "arms: old (govec-mmap, 2048 MiB memory, 3 runs); new (govec-mmap, 2048 MiB memory, 3 runs)"
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


@pytest.mark.parametrize(
    ("limit", "other_limit", "expected"),
    [
        (2 * 1024**3, 2 * 1024**3, True),
        (2 * 1024**3, 450 * 1024**2, False),  # a capped arm against the same code uncapped
        (None, None, False),  # files from before the limit was recorded: not known to match
    ],
)
def test_same_setup__memory_limit__part_of_what_must_match(
    *, limit: int | None, other_limit: int | None, expected: bool
) -> None:
    # Arrange
    arm = ArmData("old", "govec-mmap", setups={Setup("sha256:3e7f", "{}", limit)})
    other = ArmData("old", "govec-mmap", setups={Setup("sha256:3e7f", "{}", other_limit)})

    # Act
    same = same_setup(arm, other)

    # Assert
    assert same is expected
