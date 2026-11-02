"""Browse mode: every session in bench.db, compared as `bench compare` does, with its runs plotted.

Left, the sessions, newest first. Right, the highlighted session's comparison: each arm's median
and range per metric, the change against the control and its verdict, and against a baseline
picked above it (none, the published runs, or another session). Below, the highlighted metric
as dots, one per run, or with h its median in every session of the experiment. Enter shows the
metric's runs one by one.
"""

import sqlite3
from typing import ClassVar, override

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, OptionList, Select, Static
from textual.widgets.option_list import Option
from textual_plotext import PlotextPlot

from govec_bench.compare import (
    PUBLISHED,
    SAME_SETUP_NOTE,
    ArmData,
    Baseline,
    Metric,
    Row,
    SessionInfo,
    SessionKey,
    baseline_column,
    baseline_matches,
    formatted,
    history,
    list_sessions,
    load_baseline,
    load_session,
    metric_of,
    run_values,
    select_metrics,
    session_rows,
    with_baseline,
)
from govec_bench.tui.tables import dot_plot, fill_rows, line_plot, plural, started_text
from govec_bench.tui.theme import palette

NO_BASELINE = "none"


def session_id(key: SessionKey) -> str:
    return f"{key.experiment}/{key.session}"


def session_option(info: SessionInfo) -> Text:
    colors, text = palette(), Text(info.key.experiment, style="bold")
    text.append(f"\n{started_text(info.key.session)}  ·  {info.dataset}", style=colors.muted)
    text.append(f"\n{plural(info.runs, 'run')}", style=colors.muted)
    if info.failed:
        text.append(f", {info.failed} failed", style=colors.bad)
    return text


def arms_text(arms: dict[str, ArmData]) -> Text:
    # Each arm as a chip in its colour, with its database, memory limit and runs.
    names, text, colors = list(arms), Text(), palette()
    for arm in arms.values():
        if text:
            text.append("   ")
        text.append(f" {arm.name} ", style=f"bold {colors.background} on {colors.arm(names, arm.name)}")
        limits = {setup.memory_limit for setup in arm.setups if setup.memory_limit is not None}
        memory = f", {limits.pop() / 1024**2:g} MiB" if len(limits) == 1 else ""
        text.append(f" {arm.database}{memory}, {plural(arm.runs, 'run')}", style="dim")
        if arm.failed:
            text.append(f", {arm.failed} failed", style=colors.bad)
        if len(arm.setups) > 1:
            text.append(", setup varied", style=colors.caution)
    return text


class RunsScreen(ModalScreen[None]):
    # One metric's runs in a session, one row each, over the browse view.
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape,q,enter", "dismiss", "Back")]

    DEFAULT_CSS = """
    RunsScreen { align: center middle; }
    #runs-card {
        width: 90;
        height: auto;
        max-height: 80%;
        border: round $accent;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    #runs { height: auto; max-height: 100%; }
    """

    def __init__(self, db: sqlite3.Connection, key: SessionKey, metric: Metric, arms: list[str]) -> None:
        super().__init__()
        self.db, self.key, self.metric, self.arms = db, key, metric, arms

    @override
    def compose(self) -> ComposeResult:
        with Vertical(id="runs-card"):
            yield DataTable(id="runs", cursor_type="row")

    def on_mount(self) -> None:
        card = self.query_one("#runs-card")
        card.border_title = self.metric.label.strip()
        card.border_subtitle = f"{self.key.experiment}  ·  {started_text(self.key.session)}  ·  esc to go back"

        colors, table = palette(), self.query_one(DataTable)
        table.add_columns("round", "arm", "ran", Text("value", justify="right"), "results file")
        for run in run_values(self.db, self.key, self.metric.name):
            table.add_row(
                run.round,
                Text(run.arm, style=f"bold {colors.arm(self.arms, run.arm)}"),
                "first" if run.position == 0 else "then",
                Text(formatted(run.value, self.metric.unit), style="bold", justify="right"),
                Text(run.label, style="dim"),
            )


class BrowseScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("h", "toggle_history", "Runs / history"),
        Binding("a", "toggle_all", "Headline / all metrics"),
        Binding("q", "app.quit", "Quit"),
    ]

    DEFAULT_CSS = """
    BrowseScreen .card {
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
        border-title-style: bold;
    }
    BrowseScreen .card:focus-within { border: round $accent; }
    #sessions { width: 46; padding: 0 1; }
    #sessions > .option-list--option-highlighted { background: $boost; text-style: none; }
    #sessions:focus > .option-list--option-highlighted { background: $accent 30%; }
    #title { height: auto; padding: 0 1; }
    #baseline-bar { height: 3; }
    #baseline-label { width: auto; padding: 1 1 0 1; color: $text-muted; }
    #baseline { width: 56; }
    #rows { height: 1fr; }
    #plot { height: 1fr; }
    """

    def __init__(
        self, db: sqlite3.Connection, *, session: SessionKey | None = None, baseline: str | None = None
    ) -> None:
        super().__init__()
        self.db = db
        self.sessions = list_sessions(db)
        self.key = session or (self.sessions[0].key if self.sessions else None)
        self.baseline_name = baseline or NO_BASELINE
        self.show_history = False
        self.every = False
        self.arms: dict[str, ArmData] = {}
        self.baseline: Baseline | None = None
        self.metrics: list[Metric] = []

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield OptionList(
                *(
                    item
                    for index, info in enumerate(self.sessions)
                    for item in ([None] if index else []) + [Option(session_option(info), id=session_id(info.key))]
                ),
                id="sessions",
                classes="card",
            )
            with Vertical():
                yield Static(id="title", classes="card")
                with Horizontal(id="baseline-bar"):
                    yield Static("Compare against", id="baseline-label")
                    yield Select(self._baseline_options(), allow_blank=False, value=self.baseline_name, id="baseline")
                yield DataTable(id="rows", cursor_type="row", classes="card")
                yield PlotextPlot(id="plot", classes="card")
        yield Footer()

    def _baseline_options(self) -> list[tuple[str, str]]:
        sessions = [(session_id(info.key), session_id(info.key)) for info in self.sessions]
        return [("no baseline", NO_BASELINE), ("the published runs", PUBLISHED), *sessions]

    def on_mount(self) -> None:
        self.sub_title = "browse"
        self.query_one("#sessions").border_title = "Sessions"
        self.query_one(
            "#rows"
        ).border_title = "Comparison: median (range) per arm; ↑↓ picks the plotted metric, enter its runs"
        self.query_one("#plot").border_title = "Runs"
        if self.key is None:
            self.query_one("#title", Static).update(
                "No experiment sessions in the index yet: run one with `task bench:run`, or `task results:import`."
            )
            return
        options = self.query_one(OptionList)
        ids = [session_id(info.key) for info in self.sessions]
        if session_id(self.key) in ids:
            options.highlighted = ids.index(session_id(self.key))
        self.show_session()

    # --- What's shown -------------------------------------------------------------------

    def show_session(self) -> None:
        if self.key is None:
            return
        dataset, self.arms = load_session(self.db, self.key)
        self.baseline = (
            None if self.baseline_name == NO_BASELINE else load_baseline(self.db, self.baseline_name, dataset)
        )
        self.metrics = select_metrics(self.arms.values(), (), every=self.every)

        control = next(iter(self.arms))
        rows = session_rows(self.arms, control, self.metrics)
        header = [*self.arms, *(f"{name} vs {control}" for name in self.arms if name != control)]
        title_card = self.query_one("#title", Static)
        title_card.border_title = f"{self.key.experiment}  ·  {started_text(self.key.session)}"
        title = [
            Text.assemble(
                (dataset, "bold"),
                ("  ·  control ", "dim"),
                (control, f"bold {palette().arm(list(self.arms), control)}"),
            ),
            arms_text(self.arms),
        ]

        if self.baseline is not None:
            rows, header, notes = self._with_baseline(rows, header, self.baseline)
            title.extend(Text(note, style=palette().info) for note in notes)
        cautions = {row.metric.label.strip(): row.metric.caution for row in rows if row.metric.caution}
        title.extend(Text(f"! {label}: {caution}", style=palette().caution) for label, caution in cautions.items())

        title_card.update(Text("\n").join(title))
        fill_rows(self.query_one("#rows", DataTable), header, rows, arms=list(self.arms))
        self.show_plot()

    def _with_baseline(
        self, rows: list[Row], header: list[str], baseline: Baseline
    ) -> tuple[list[Row], list[str], list[str]]:
        matches = baseline_matches(self.arms, baseline.arms, by_database=baseline.by_database, metrics=self.metrics)
        notes = [f"against {baseline.name}"] + [
            f"{match.arm}: {SAME_SETUP_NOTE}" if match.same_setup else f"{match.arm}: nothing in the baseline"
            for match in matches
            if match.same_setup or match.baseline is None
        ]
        columns = [*header, *(baseline_column(match.arm) for match in matches)]
        return with_baseline(rows, matches), columns, notes

    def highlighted_metric(self) -> Metric | None:
        table = self.query_one("#rows", DataTable)
        if not table.row_count:
            return None
        name = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return metric_of(name) if name else None

    def show_plot(self) -> None:
        metric = self.highlighted_metric()
        if metric is None or self.key is None:
            return
        plot = self.query_one(PlotextPlot)
        plot.border_title = (
            f"{metric.label.strip()}: {'median per session' if self.show_history else 'one dot per run'}"
        )

        if self.show_history:
            series: dict[str, tuple[list[str], list[float]]] = {}
            for point in history(self.db, self.key.experiment, metric.name):
                sessions, medians = series.setdefault(point.arm, ([], []))
                sessions.append(point.session)
                medians.append(point.summary.median)
            line_plot(plot, metric, series)
            return

        groups: dict[str, list[float]] = {name: arm.values.get(metric.name, []) for name, arm in self.arms.items()}
        if self.baseline is not None:
            for name, arm in self.arms.items():
                match = self.baseline.arms.get(arm.database if self.baseline.by_database else name)
                if match is not None:
                    groups[f"{name} (baseline)"] = match.values.get(metric.name, [])
        dot_plot(plot, metric, groups)

    # --- Events -------------------------------------------------------------------------

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        experiment, _, session = (event.option.id or "").partition("/")
        if session and SessionKey(experiment, session) != self.key:
            self.key = SessionKey(experiment, session)
            self.show_session()

    def on_select_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and event.value != self.baseline_name:
            self.baseline_name = event.value
            self.show_session()

    def on_data_table_row_highlighted(self, _event: DataTable.RowHighlighted) -> None:
        self.show_plot()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        metric = self.highlighted_metric()
        if event.data_table.id == "rows" and metric is not None and self.key is not None:
            self.app.push_screen(RunsScreen(self.db, self.key, metric, list(self.arms)))

    def action_toggle_history(self) -> None:
        self.show_history = not self.show_history
        self.show_plot()

    def action_toggle_all(self) -> None:
        self.every = not self.every
        self.show_session()
