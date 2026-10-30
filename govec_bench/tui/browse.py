"""Browse mode: every session in bench.db, compared as `bench compare` does, with its runs plotted.

Left, the sessions, newest first. Right, the highlighted session's comparison: each arm's median
and range per metric, the change against the control and its verdict, and against a baseline
picked above it (none, the published runs, or another session). Below, the highlighted metric
as dots, one per run, or with h its median in every session of the experiment. Enter shows the
metric's runs one by one.
"""

import sqlite3
from typing import ClassVar, override

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
    SessionKey,
    arms_line,
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
from govec_bench.tui.tables import dot_plot, fill_rows, line_plot

NO_BASELINE = "none"


def session_id(key: SessionKey) -> str:
    return f"{key.experiment}/{key.session}"


class RunsScreen(ModalScreen[None]):
    # One metric's runs in a session, one row each.
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape,q,enter", "dismiss", "Back")]

    def __init__(self, db: sqlite3.Connection, key: SessionKey, metric: Metric) -> None:
        super().__init__()
        self.db, self.key, self.metric = db, key, metric

    @override
    def compose(self) -> ComposeResult:
        yield Static(f"{self.metric.label.strip()} in {self.key.experiment}/{self.key.session}", id="runs-title")
        yield DataTable(id="runs", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("round", "arm", "position", "value", "file")
        for run in run_values(self.db, self.key, self.metric.name):
            table.add_row(run.round, run.arm, run.position, formatted(run.value, self.metric.unit), run.label)


class BrowseScreen(Screen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("h", "toggle_history", "Runs / history"),
        Binding("a", "toggle_all", "Headline / all metrics"),
        Binding("q", "app.quit", "Quit"),
    ]

    DEFAULT_CSS = """
    #sessions { width: 42; }
    #title { height: auto; padding: 0 1; }
    #baseline { width: 60; }
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
                    Option(
                        f"{info.key.experiment}/{info.key.session}\n  {info.dataset}, {info.runs} runs",
                        id=session_id(info.key),
                    )
                    for info in self.sessions
                ),
                id="sessions",
            )
            with Vertical():
                yield Static(id="title")
                yield Select(self._baseline_options(), allow_blank=False, value=self.baseline_name, id="baseline")
                yield DataTable(id="rows", cursor_type="row")
                yield PlotextPlot(id="plot")
        yield Footer()

    def _baseline_options(self) -> list[tuple[str, str]]:
        sessions = [(session_id(info.key), session_id(info.key)) for info in self.sessions]
        return [("no baseline", NO_BASELINE), ("the published runs", PUBLISHED), *sessions]

    def on_mount(self) -> None:
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
        title = [f"{session_id(self.key)}: {dataset}, control {control}", arms_line(self.arms)]

        if self.baseline is not None:
            rows, header, notes = self._with_baseline(rows, header, self.baseline)
            title.extend(notes)
        cautions = {row.metric.label.strip(): row.metric.caution for row in rows if row.metric.caution}
        title.extend(f"! {label}: {caution}" for label, caution in cautions.items())

        self.query_one("#title", Static).update("\n".join(title))
        fill_rows(self.query_one("#rows", DataTable), header, rows)
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
            self.app.push_screen(RunsScreen(self.db, self.key, metric))

    def action_toggle_history(self) -> None:
        self.show_history = not self.show_history
        self.show_plot()

    def action_toggle_all(self) -> None:
        self.every = not self.every
        self.show_session()
