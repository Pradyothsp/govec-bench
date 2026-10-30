import asyncio
import sqlite3

from textual.widgets import DataTable, Static

from govec_bench.compare import SessionKey
from govec_bench.tui.app import BenchApp
from govec_bench.tui.browse import BrowseScreen, RunsScreen


def test_browse_screen__newest_session__its_comparison_and_drill_down_to_runs(db: sqlite3.Connection) -> None:
    # Arrange
    app = BenchApp(db)

    async def drive() -> tuple[str, list[str], int]:
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, BrowseScreen)
            title = str(screen.query_one("#title", Static).render())
            table = screen.query_one("#rows", DataTable)
            row = [str(cell) for cell in table.get_row_at(0)]

            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RunsScreen)
            runs = app.screen.query_one(DataTable).row_count
        return title, row, runs

    # Act
    title, row, runs = asyncio.run(drive())

    # Assert
    assert title.startswith("ab/s2: dbpedia100k, control old")
    assert row[0] == "Restart with data, p50 (ms)"
    assert row[-1] == "+2.0%  every run worse"  # new, 552 ms, against old slowed to 541
    assert runs == 6


def test_browse_screen__baseline_session__a_column_per_arm_and_the_same_setup_noted(db: sqlite3.Connection) -> None:
    # Arrange
    app = BenchApp(db, session=SessionKey("ab", "s2"), baseline="ab/s1")

    async def drive() -> tuple[str, list[str]]:
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, BrowseScreen)
            title = str(screen.query_one("#title", Static).render())
            row = [str(cell) for cell in screen.query_one("#rows", DataTable).get_row_at(0)]
        return title, row

    # Act
    title, row = asyncio.run(drive())

    # Assert
    assert "old: same image and settings: this change is the sessions, not the code" in title
    assert row[-2] == "+10.0%  every run worse"  # old against itself in s1: the sessions' drift
