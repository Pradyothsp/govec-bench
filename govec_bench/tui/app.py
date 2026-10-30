"""The bench terminal app: `bench ui` to browse, and what `bench run` opens while a session runs."""

import sqlite3
import subprocess
import sys
from pathlib import Path

from textual.app import App

from govec_bench.compare import SessionKey
from govec_bench.tui.browse import BrowseScreen
from govec_bench.tui.live import LiveScreen


class BenchApp(App[None]):
    TITLE = "govec-bench"

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        live: Path | None = None,
        session: SessionKey | None = None,
        baseline: str | None = None,
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        super().__init__()
        self.db, self.live, self.session, self.baseline, self.process = db, live, session, baseline, process

    def on_mount(self) -> None:
        if self.live is not None:
            self.push_screen(LiveScreen(self.db, self.live, baseline=self.baseline, process=self.process))
        else:
            self.push_screen(BrowseScreen(self.db, session=self.session, baseline=self.baseline))


def start_runner(experiment: Path, directory: Path, rounds: int | None) -> subprocess.Popen[bytes]:
    # The run in a process of its own, in a session of its own: Ctrl-C or a closed terminal reaches
    # the view, not the night. Its output goes to run.log; what it prints before that (an import
    # error) goes to runner.err.
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "govec_bench.cli", "run", str(experiment), "--plain", "--into", str(directory)]
    if rounds:
        command += ["--rounds", str(rounds)]
    with (directory / "runner.err").open("wb") as errors:
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors, start_new_session=True)  # noqa: S603 -- our own CLI
