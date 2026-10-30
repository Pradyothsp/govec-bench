"""Live mode: what a session is doing while it runs, from the files it writes.

It reads the session's folder, not the runner: events.jsonl for the round, arm and step,
session.json for the plan, run.log for the output, and bench.db for each run's numbers as they
land. So it can attach to a session already running (`bench ui` finds it), and quitting it
leaves the run going.
`bench ui` opens it on the newest session still running, without being told which. Docker's container stats are read every 10 s, a light call next to what
the benchmark itself does.
"""

import json
import os
import shutil
import signal
import sqlite3
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, override

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Log, Static

from govec_bench.benchmarks.common import DOCKER, DOCKER_DESKTOP_DISK_DIR
from govec_bench.benchmarks.experiment import runner_alive
from govec_bench.compare import (
    SAME_SETUP_NOTE,
    SessionKey,
    baseline_column,
    baseline_matches,
    load_baseline,
    load_session,
    select_metrics,
    session_rows,
    with_baseline,
)
from govec_bench.events import EVENTS_FILE, Event, read_events
from govec_bench.tui.browse import BrowseScreen
from govec_bench.tui.tables import fill_rows

POLL_S = 1.0
DOCKER_STATS_S = 10.0
STOP_CONFIRM_S = 5.0

_MARKS = {"done": "✓", "failed": "✗", "skipped": "-"}


@dataclass(slots=True)
class LiveState:
    # What the events so far say; fold() applies each one.
    round: int | None = None
    arm: str | None = None
    stage: str | None = None
    step: str | None = None
    step_started: float | None = None
    arm_started: float | None = None
    first: float | None = None
    durations: list[float] = field(default_factory=list)  # each finished arm run, in seconds
    statuses: dict[tuple[int, str], str] = field(default_factory=dict)  # by (round, arm)
    warnings: list[str] = field(default_factory=list)
    ended: str | None = None
    new_results: bool = False  # a run ended since the view last read bench.db


def fold(state: LiveState, event: Event) -> None:
    state.first = state.first or event.t
    if event.kind == "round":
        state.round = event.round
    elif event.kind == "arm":
        state.round, state.arm, state.arm_started = event.round, event.arm, event.t
        state.stage = state.step = None
    elif event.kind == "stage":
        state.stage = event.text
    elif event.kind == "step":
        state.step, state.step_started = event.text, event.t
    elif event.kind == "result" and event.round is not None and event.arm is not None:
        state.statuses[event.round, event.arm] = event.status or "done"
        if state.arm_started is not None and event.status != "skipped":
            state.durations.append(event.t - state.arm_started)
        state.arm = state.stage = state.step = None
        state.new_results = True
    elif event.kind == "warning" and event.text:
        state.warnings.append(event.text)
    elif event.kind == "end":
        state.ended = event.status


def time_left(state: LiveState, total_runs: int, now: float) -> float | None:
    # The mean finished run times the runs still to go, less how far into the current one.
    if not state.durations or state.ended:
        return None
    remaining = total_runs - len(state.statuses)
    into_current = now - state.arm_started if state.arm is not None and state.arm_started is not None else 0
    return max(statistics.mean(state.durations) * remaining - into_current, 0)


def duration_text(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} m"
    return f"{minutes} m {secs:02d} s" if minutes else f"{secs} s"


def last_segments(text: str) -> list[str]:
    # run.log's lines, each as a terminal would leave it: a progress bar redraws itself with \r.
    return [line.rsplit("\r", 1)[-1] for line in text.split("\n")]


class LiveScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "stop", "Stop the run"),
        Binding("b", "browse", "Browse"),
        Binding("q", "quit", "Quit (the run goes on)"),
    ]

    DEFAULT_CSS = """
    #status { height: auto; padding: 0 1; }
    #plan { width: 40; height: 1fr; }
    #results { height: 1fr; }
    #log { height: 12; }
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        directory: Path,
        *,
        baseline: str | None = None,
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        super().__init__()
        self.db, self.directory, self.baseline_name, self.process = db, directory, baseline, process
        self.key = SessionKey(directory.parent.name, directory.name)
        self.state = LiveState()
        self.event_offset = self.log_offset = 0
        self.log_partial = ""
        self.plan: list[list[str]] = []
        self.pid: int | None = None
        self.containers = ""
        self.stop_asked: float | None = None

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status")
        with Horizontal():
            yield DataTable(id="plan", cursor_type="none")
            with Vertical():
                yield DataTable(id="results", cursor_type="row")
        yield Log(id="log", max_lines=2000)
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"{self.key.experiment}/{self.key.session}"
        self.poll()
        self.set_interval(POLL_S, self.poll)
        if DOCKER:
            self.set_interval(DOCKER_STATS_S, self.read_docker_stats)
            self.read_docker_stats()

    # --- Reading the session's files ----------------------------------------------------

    def poll(self) -> None:
        self._read_session()
        events, self.event_offset = read_events(self.directory / EVENTS_FILE, self.event_offset)
        warnings_before = len(self.state.warnings)
        for event in events:
            fold(self.state, event)
        for warning in self.state.warnings[warnings_before:]:
            self.notify(warning, severity="warning", timeout=10)

        self._read_log()
        if self.state.new_results:
            self.state.new_results = False
            self.show_results()
        self.show_plan()
        self.show_status()

    def _read_session(self) -> None:
        try:
            record = json.loads((self.directory / "session.json").read_text())
        except OSError, ValueError:
            return
        self.plan, self.pid = record.get("plan", []), record.get("pid")

    def _read_log(self) -> None:
        try:
            with (self.directory / "run.log").open("rb") as file:
                file.seek(self.log_offset)
                data = file.read()
        except FileNotFoundError:
            return
        self.log_offset += len(data)
        lines = last_segments(self.log_partial + data.decode(errors="replace"))
        self.log_partial = lines.pop()  # no newline yet: may still grow
        if lines:
            self.query_one(Log).write_lines(lines)

    def read_docker_stats(self) -> None:
        self.run_worker(self._docker_stats, thread=True, exclusive=True, group="docker-stats")

    def _docker_stats(self) -> None:
        if DOCKER is None:
            return
        result = subprocess.run(  # noqa: S603 -- fixed args
            [DOCKER, "stats", "--no-stream", "--format", "{{.Name}} CPU {{.CPUPerc}}, memory {{.MemUsage}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = "; ".join(result.stdout.split("\n")[:-1]) if result.returncode == 0 else "docker stats failed"
        self.app.call_from_thread(self._set_containers, text or "no containers running")

    def _set_containers(self, text: str) -> None:
        self.containers = text
        self.show_status()

    # --- What's shown -------------------------------------------------------------------

    def runner_alive(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return runner_alive(self.pid)

    def _runner_error(self) -> str:
        # What the runner printed outside run.log: a crash before or after it logs.
        try:
            lines = (self.directory / "runner.err").read_text().strip().splitlines()
        except OSError:
            return ""
        return lines[-1] if lines else ""

    def show_status(self) -> None:
        state, now = self.state, time.time()
        total = sum(len(order) for order in self.plan)
        lines = []

        if state.ended:
            lines.append(f"session {state.ended}: {len(state.statuses)} of {total} runs")
        elif not self.runner_alive() and (state.first is not None or self.process is not None):
            lines.append(f"the runner stopped before the session ended: {self._runner_error() or 'see run.log'}")
        else:
            where = f"round {state.round or '-'} of {len(self.plan)}"
            if state.arm:
                where += f" · {state.arm}"
            if state.stage:
                where += f" · {state.stage}"
            if state.step:
                elapsed = f" ({duration_text(now - state.step_started)})" if state.step_started else ""
                where += f" · {state.step}{elapsed}"
            lines.append(where)

        timing = [f"{len(state.statuses)} of {total} runs done"]
        if state.first is not None:
            timing.append(f"{duration_text(now - state.first)} in")
        left = time_left(state, total, now)
        if left is not None:
            timing.append(f"about {duration_text(left)} left")
        lines.append(" · ".join(timing))

        disk = DOCKER_DESKTOP_DISK_DIR if DOCKER_DESKTOP_DISK_DIR.exists() else Path("/")
        resources = [f"Mac disk {shutil.disk_usage(disk).free / 1000**3:.0f} GB free"]
        if self.containers:
            resources.append(self.containers)
        lines.append(" · ".join(resources))
        if state.warnings:
            lines.append(f"last warning: {state.warnings[-1]}")

        self.query_one("#status", Static).update("\n".join(lines))

    def show_plan(self) -> None:
        table = self.query_one("#plan", DataTable)
        width = max((len(order) for order in self.plan), default=0)
        table.clear(columns=True)
        table.add_columns("round", *(f"{position + 1}" for position in range(width)))
        for number, order in enumerate(self.plan, start=1):
            cells = []
            for arm in order:
                status = self.state.statuses.get((number, arm))
                running = self.state.round == number and self.state.arm == arm
                cells.append(f"{arm} {_MARKS.get(status or '', '▶' if running else '·')}")
            table.add_row(number, *cells)

    def show_results(self) -> None:
        try:
            dataset, arms = load_session(self.db, self.key)
        except SystemExit:  # nothing indexed yet
            return
        metrics = select_metrics(arms.values(), (), every=False)
        control = next(iter(arms))
        rows = session_rows(arms, control, metrics)
        header = [*arms, *(f"{name} vs {control}" for name in arms if name != control)]

        if self.baseline_name:
            baseline = load_baseline(self.db, self.baseline_name, dataset)
            matches = baseline_matches(arms, baseline.arms, by_database=baseline.by_database, metrics=metrics)
            rows = with_baseline(rows, matches)
            header.extend(baseline_column(match.arm) for match in matches)
            for match in matches:
                if match.same_setup:
                    self.notify(f"{match.arm}: {SAME_SETUP_NOTE}", timeout=10)

        fill_rows(self.query_one("#results", DataTable), header, rows)

    # --- Keys ---------------------------------------------------------------------------

    def action_stop(self) -> None:
        if not self.runner_alive():
            self.notify("the run isn't going")
            return
        now = time.time()
        if self.stop_asked is None or now - self.stop_asked > STOP_CONFIRM_S:
            self.stop_asked = now
            self.notify("press s again to stop the run; what ran so far is kept", timeout=STOP_CONFIRM_S)
            return
        # As Ctrl-C would: the runner marks the session interrupted and takes its container down.
        pid = self.process.pid if self.process is not None else self.pid
        if pid is not None:
            os.kill(pid, signal.SIGINT)
        self.notify("stopping after the runner cleans up")

    def action_browse(self) -> None:
        self.app.push_screen(BrowseScreen(self.db, session=self.key, baseline=self.baseline_name))

    def action_quit(self) -> None:
        message = "The run goes on. Watch it again with: task bench:ui"
        self.app.exit(message=message if self.runner_alive() else None)
