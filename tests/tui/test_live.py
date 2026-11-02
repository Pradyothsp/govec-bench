import asyncio
import json
import os
import sqlite3
from pathlib import Path

import pytest
from textual.widgets import DataTable, RichLog, Static

from govec_bench.events import Event, EventLog
from govec_bench.tui import live
from govec_bench.tui.app import BenchApp
from govec_bench.tui.live import (
    ContainerStats,
    LiveScreen,
    LiveState,
    duration_text,
    fold,
    parse_docker_stats,
    time_left,
)


def _events() -> list[Event]:
    return [
        Event("round", round=1, t=100),
        Event("arm", round=1, arm="old", position=0, t=100),
        Event("step", round=1, arm="old", text="batch insert", t=110),
        Event("result", round=1, arm="old", status="done", file="r1-old.json", t=400),
        Event("arm", round=1, arm="new", position=1, t=400),
        Event("stage", round=1, arm="new", text="loaded index", t=401),
        Event("step", round=1, arm="new", text="recall and query latency", t=500),
        Event("warning", text="the Mac's disk has 30 GB free", t=510),
    ]


def test_fold__events__where_the_session_is_and_what_finished() -> None:
    # Arrange
    state = LiveState()

    # Act
    for event in _events():
        fold(state, event)

    # Assert
    assert (state.round, state.arm, state.stage, state.step) == (1, "new", "loaded index", "recall and query latency")
    assert state.statuses == {(1, "old"): "done"}
    assert state.durations == [300]
    assert state.warnings == ["the Mac's disk has 30 GB free"]
    assert state.new_results


def test_time_left__one_run_finished__mean_run_times_the_rest_less_the_current_one() -> None:
    # Arrange
    state = LiveState()
    for event in _events():
        fold(state, event)

    # Act
    left = time_left(state, total_runs=4, now=550)

    # Assert
    assert left == 300 * 3 - 150


def test_parse_docker_stats__json_lines__name_cpu_memory_and_disk() -> None:
    # Arrange
    output = (
        '{"Name":"govec-bench-govec","CPUPerc":"155.07%","MemUsage":"257.7MiB / 450MiB","BlockIO":"399GB / 840MB"}\n'
    )

    # Act
    stats = parse_docker_stats(output)

    # Assert
    assert stats == [ContainerStats("govec-bench-govec", "155.07%", "257.7MiB / 450MiB", "399GB / 840MB")]


def test_duration_text__seconds_minutes_hours() -> None:
    # Arrange
    seconds = [42, 754, 7_380]

    # Act
    texts = [duration_text(value) for value in seconds]

    # Assert
    assert texts == ["42 s", "12 m 34 s", "2 h 03 m"]


def test_live_screen__session_folder__shows_where_it_is_and_the_plan(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(live, "DOCKER", None)  # no docker stats from a test
    directory = tmp_path / "sessions" / "ab" / "s1"
    directory.mkdir(parents=True)
    (directory / "session.json").write_text(json.dumps({"plan": [["old", "new"], ["new", "old"]], "pid": os.getpid()}))
    log = EventLog(directory)
    for event in _events():
        log.emit(event)
    (directory / "run.log").write_text(
        "round 1 of 2\n  old:\n    batch insert...\n"
        "batch inserts:  64%|######4   | 640/1000 [03:09<18:30,  3.08s/batch]\n"
    )

    async def drive() -> tuple[str, str, list[str], int, str, int]:
        app = BenchApp(db, live=directory)
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LiveScreen)
            now = str(screen.query_one("#now", Static).render())
            session = str(screen.query_one("#session", Static).render())
            plan = screen.query_one("#plan", DataTable)
            cells = [str(cell) for cell in plan.get_row_at(0)]
            results = screen.query_one("#results", DataTable).row_count
            progress = str(screen.query_one("#progress-text", Static).render())
            logged = len(screen.query_one("#log", RichLog).lines)
        return now, session, cells, results, progress, logged

    # Act
    now, session, cells, results, progress, logged = asyncio.run(drive())

    # Assert
    assert "Round 1 of 2" in now
    assert " new " in now
    assert "loaded index → recall and query latency" in now
    assert "⚠ the Mac's disk has 30 GB free" in now
    assert "1 of 4 runs" in session
    assert cells == ["1", "old ✓", "new ▶"]
    assert results == 1  # session s1's restart metric, read from the index
    assert progress == "640 of 1,000  ·  03:09 in  ·  18:30 left  ·  3.08 s/batch"
    assert logged == 3  # the bar shows as a bar, not in the log
