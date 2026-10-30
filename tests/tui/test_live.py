import asyncio
import json
import os
import sqlite3
from pathlib import Path

import pytest
from textual.widgets import DataTable, Static

from govec_bench.events import Event, EventLog
from govec_bench.tui import live
from govec_bench.tui.app import BenchApp
from govec_bench.tui.live import LiveScreen, LiveState, duration_text, fold, last_segments, time_left


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


def test_last_segments__progress_bar_redraws__keeps_what_the_terminal_would_show() -> None:
    # Arrange
    text = "batch insert...\nbatch:  10%\rbatch:  50%\rbatch: 100%\n  -> done\npartial"

    # Act
    lines = last_segments(text)

    # Assert
    assert lines == ["batch insert...", "batch: 100%", "  -> done", "partial"]


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
    (directory / "run.log").write_text("round 1 of 2\n  old:\n")

    async def drive() -> tuple[str, list[str], int]:
        app = BenchApp(db, live=directory)
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LiveScreen)
            status = str(screen.query_one("#status", Static).render())
            plan = screen.query_one("#plan", DataTable)
            cells = [str(cell) for cell in plan.get_row_at(0)]
            results = screen.query_one("#results", DataTable).row_count
        return status, cells, results

    # Act
    status, cells, results = asyncio.run(drive())

    # Assert
    assert "round 1 of 2 · new · loaded index · recall and query latency" in status
    assert "1 of 4 runs done" in status
    assert "last warning: the Mac's disk has 30 GB free" in status
    assert cells == ["1", "old ✓", "new ▶"]
    assert results == 1  # session s1's restart metric, read from the index
