"""Live mode: what a session is doing while it runs, from the files it writes.

It reads the session's folder, not the runner: events.jsonl for the round, arm and step,
session.json for the plan, run.log for the output, and bench.db for each run's numbers as they
land. So it can attach to a session already running (`bench ui` finds it), and quitting it
leaves the run going. `bench ui` opens it on the newest session still running, without being
told which. Docker's container stats are read every 10 s, a light call next to what the benchmark
itself does.

Top, three cards: where the session is, how far through it is, and the machine. Under them, the
step's progress bar. Middle, the plan (every round's arms and how each ended) and the results so
far, one row per metric. Bottom, run.log in colour, without the progress bars.
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
from typing import ClassVar, NamedTuple, override

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, ProgressBar, RichLog, Static

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
from govec_bench.tui.runlog import Progress, last_segments, parse_progress, styled_line
from govec_bench.tui.tables import fill_rows
from govec_bench.tui.theme import palette

POLL_S = 1.0
DOCKER_STATS_S = 10.0
STOP_CONFIRM_S = 5.0


def _mark(status: str) -> tuple[str, str]:
    # How a run in the plan ended, or that it's running or still to come: its mark and style.
    colors = palette()
    marks = {
        "done": ("✓", f"bold {colors.good}"),
        "failed": ("✗", f"bold {colors.bad}"),
        "skipped": ("-", "dim"),
        "running": ("▶", f"bold {colors.caution}"),
    }
    return marks.get(status, ("·", "dim"))


class ContainerStats(NamedTuple):
    name: str
    cpu: str
    memory: str
    block_io: str  # read / written since the container started


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


def parse_docker_stats(output: str) -> list[ContainerStats]:
    stats = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        stats.append(ContainerStats(item["Name"], item["CPUPerc"], item["MemUsage"], item["BlockIO"]))
    return stats


def _row(label: str, value: str | Text, style: str = "") -> Text:
    return Text.assemble((f"{label:<10}", "dim"), value if isinstance(value, Text) else Text(value, style=style))


class LiveScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "stop", "Stop the run"),
        Binding("b", "browse", "Browse"),
        Binding("q", "quit", "Quit (the run goes on)"),
    ]

    DEFAULT_CSS = """
    LiveScreen .card {
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 0 1;
    }
    #cards { height: 6; }
    #cards .card { width: 1fr; height: 100%; }
    #progress { height: 4; }
    #progress ProgressBar { width: 100%; }
    #progress Bar { width: 1fr; }
    #progress-text { color: $text-muted; }
    #middle { height: 2fr; }
    #plan { width: 34; }
    #results-box { width: 1fr; }
    #results { height: 1fr; }
    #results-empty { color: $text-muted; padding: 1 2; }
    #log { height: 1fr; min-height: 6; }
    LiveScreen .warning-card { border: round $warning; }
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
        self.progress: Progress | None = None
        self.plan: list[list[str]] = []
        self.pid: int | None = None
        self.containers: list[ContainerStats] | None = None
        self.stop_asked: float | None = None

    @override
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="cards"):
            yield Static(id="now", classes="card")
            yield Static(id="session", classes="card")
            yield Static(id="machine", classes="card")
        with Vertical(id="progress", classes="card"):
            yield ProgressBar(show_eta=False)
            yield Static(id="progress-text")
        with Horizontal(id="middle"):
            yield DataTable(id="plan", cursor_type="none", classes="card")
            with Vertical(id="results-box", classes="card"):
                yield DataTable(id="results", cursor_type="row")
                yield Static("Numbers land here as each run finishes.", id="results-empty")
        yield RichLog(id="log", max_lines=2000, wrap=True, classes="card")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.key.experiment
        self.sub_title = f"session {self.key.session}, live"
        titles = {
            "#now": "Now",
            "#session": "Session",
            "#machine": "Machine",
            "#progress": "Progress",
            "#plan": "Plan",
            "#results-box": "Results so far: median (range) per arm, and each against the control",
            "#log": "Log",
        }
        for selector, title in titles.items():
            self.query_one(selector).border_title = title
        self.query_one("#results", DataTable).display = False
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
        log, arms = self.query_one(RichLog), self.arms()
        for line in lines:
            progress = parse_progress(line)
            if progress is not None:
                self.progress = progress
            elif line.strip():
                # A line after a bar means its step moved on; the bar shows until the next one.
                if self.progress is not None and self.progress.done < self.progress.total:
                    self.progress = None
                log.write(styled_line(line, arms))
        # On a terminal the bar redraws in place, so the newest one is the line still growing.
        progress = parse_progress(self.log_partial)
        if progress is not None:
            self.progress = progress
        self.show_progress()

    def read_docker_stats(self) -> None:
        self.run_worker(self._docker_stats, thread=True, exclusive=True, group="docker-stats")

    def _docker_stats(self) -> None:
        if DOCKER is None:
            return
        result = subprocess.run(  # noqa: S603 -- fixed args
            [DOCKER, "stats", "--no-stream", "--format", "{{json .}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        stats = parse_docker_stats(result.stdout) if result.returncode == 0 else None
        self.app.call_from_thread(self._set_containers, stats)

    def _set_containers(self, stats: list[ContainerStats] | None) -> None:
        self.containers = stats
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

    def arms(self) -> list[str]:
        # The arms in the order round 1 runs them: what colours them everywhere.
        return list(self.plan[0]) if self.plan else []

    def show_status(self) -> None:
        state, now = self.state, time.time()
        total = sum(len(order) for order in self.plan)
        self._show_now(state, now)

        done = len(state.statuses)
        failed = sum(status == "failed" for status in state.statuses.values())
        width = 30
        filled = round(width * done / total) if total else 0
        runs = Text.assemble((f"{done} of {total} runs", "bold"))
        if failed:
            runs.append(f"  ·  {failed} failed", style=f"bold {palette().bad}")
        session = [runs, Text.assemble(("━" * filled, palette().good), ("━" * (width - filled), palette().track))]
        if state.first is not None:
            session.append(_row("elapsed", duration_text(now - state.first)))
        left = time_left(state, total, now)
        if left is not None:
            session.append(_row("left", f"about {duration_text(left)}"))
        self.query_one("#session", Static).update(Text("\n").join(session))

        disk = DOCKER_DESKTOP_DISK_DIR if DOCKER_DESKTOP_DISK_DIR.exists() else Path("/")
        machine = [_row("Mac disk", f"{shutil.disk_usage(disk).free / 1000**3:.0f} GB free")]
        if DOCKER is None:
            machine.append(_row("docker", "not found", "dim"))
        elif self.containers is None:
            machine.append(_row("docker", "reading stats…", "dim"))
        elif not self.containers:
            machine.append(_row("docker", "no container running", "dim"))
        for container in self.containers or []:
            name = container.name.removeprefix("govec-bench-")
            machine.append(_row(name, f"CPU {container.cpu}  ·  memory {container.memory}"))
            machine.append(_row("", f"disk read / written {container.block_io}", "dim"))
        self.query_one("#machine", Static).update(Text("\n").join(machine))

    def _show_now(self, state: LiveState, now: float) -> None:
        card, colors = self.query_one("#now", Static), palette()
        lines: list[Text] = []
        if state.ended:
            style = f"bold {colors.good}" if state.ended == "done" else f"bold {colors.caution}"
            lines.append(Text(f"Session {state.ended}", style=style))
        elif not self.runner_alive() and (state.first is not None or self.process is not None):
            lines.append(Text("The runner stopped before the session ended", style=f"bold {colors.bad}"))
            lines.append(Text(self._runner_error() or "see the log", style=colors.bad))
        else:
            where = Text.assemble(("Round ", "bold"), (f"{state.round or '-'} of {len(self.plan)}", "bold"))
            if state.arm:
                where.append("   ")
                where.append(
                    f" {state.arm} ", style=f"bold {colors.background} on {colors.arm(self.arms(), state.arm)}"
                )
            lines.append(where)
            if state.stage or state.step:
                lines.append(_row("step", " → ".join(part for part in (state.stage, state.step) if part)))
            if state.step and state.step_started:
                lines.append(_row("for", duration_text(now - state.step_started)))
        if state.warnings:
            lines.append(Text(f"⚠ {state.warnings[-1]}", style=colors.caution, overflow="ellipsis", no_wrap=True))
        card.set_class(bool(state.warnings), "warning-card")
        card.update(Text("\n").join(lines))

    def show_progress(self) -> None:
        box = self.query_one("#progress", Vertical)
        bar, text = self.query_one(ProgressBar), self.query_one("#progress-text", Static)
        progress = self.progress
        if progress is None:
            box.border_title = "Progress"
            bar.update(total=None, progress=0)
            text.update("this step has no progress bar")
            return
        box.border_title = progress.desc
        bar.update(total=progress.total, progress=progress.done)
        left = "" if progress.left == "?" else f"  ·  {progress.left} left"
        rate = f"  ·  {progress.rate}" if progress.rate else ""
        text.update(f"{progress.done:,} of {progress.total:,}  ·  {progress.elapsed} in{left}{rate}")

    def show_plan(self) -> None:
        table = self.query_one("#plan", DataTable)
        width = max((len(order) for order in self.plan), default=0)
        arms = self.arms()
        table.clear(columns=True)
        table.add_columns("round", *(("first", "then")[min(position, 1)] for position in range(width)))
        for number, order in enumerate(self.plan, start=1):
            cells = []
            for arm in order:
                status = self.state.statuses.get((number, arm))
                running = self.state.round == number and self.state.arm == arm
                mark, style = _mark(status or ("running" if running else "waiting"))
                cell = Text(arm, style=palette().arm(arms, arm) if status or running else "dim")
                cell.append(f" {mark}", style=style)
                cells.append(cell)
            table.add_row(Text(str(number), style="bold" if self.state.round == number else "dim"), *cells)

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

        table = self.query_one("#results", DataTable)
        fill_rows(table, header, rows, arms=self.arms())
        table.display = bool(rows)
        self.query_one("#results-empty").display = not rows

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
