"""What the live and browse views share: a comparison's rows in a DataTable, and dot plots."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from rich.text import Text
from textual.widgets import DataTable
from textual_plotext import PlotextPlot

from govec_bench.compare import Change, Metric, Row, label_of, summary_text
from govec_bench.tui.theme import palette, rgb


def _verdict_style(verdict: str) -> str:
    colors = palette()
    for word, style in (
        ("better", colors.good),
        ("worse", colors.bad),
        ("higher", colors.caution),
        ("lower", colors.caution),
        ("level", "dim"),
        ("overlaps", "dim"),
    ):
        if word in verdict:
            return style
    return ""


def started_text(session: str) -> str:
    # A session's stamp (20261031T013303Z, UTC) as local time: Oct 30 21:33.
    try:
        started = datetime.strptime(session, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).astimezone()
    except ValueError:
        return session
    return f"{started:%b} {started.day} {started:%H:%M}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def change_cell(item: Change | None) -> Text:
    if item is None:
        return Text("")
    moved = f"{item.moved:+.1%}  " if item.moved is not None else ""
    return Text(f"{moved}{item.verdict}", style=_verdict_style(item.verdict))


def _header_cell(name: str, arms: Sequence[str]) -> Text:
    # An arm's column in its colour; a change column ("new vs old") in the first arm's.
    arm = name.split(" vs ", 1)[0].removesuffix(" (baseline)")
    return Text(name, style=f"bold {palette().arm(arms, arm)}" if arm in arms else "bold")


def fill_rows(
    table: DataTable[str | Text], header: Sequence[str], rows: Sequence[Row], *, arms: Sequence[str] = ()
) -> None:
    # Keyed by metric name, so the highlighted row can be kept across a refresh.
    highlighted = table.cursor_row
    table.clear(columns=True)
    table.add_columns(Text("metric", style="bold"), *(_header_cell(name, arms) for name in header))
    for row in rows:
        label = Text(label_of(row.metric), style="" if row.metric.label == row.metric.label.lstrip() else "dim")
        if row.metric.caution:
            label.append(" !", style=f"bold {palette().caution}")
        table.add_row(
            label,
            *(summary_text(summary, row.metric.unit) for summary in row.summaries.values()),
            *(change_cell(item) for item in row.changes.values()),
            key=row.metric.name,
        )
    if rows:
        table.move_cursor(row=min(highlighted, len(rows) - 1))


def _themed(plot: PlotextPlot) -> None:
    plt, colors = plot.plt, palette()
    plt.canvas_color(rgb(colors.background))
    plt.axes_color(rgb(colors.background))
    plt.ticks_color(rgb(colors.faint))


def _scaled(values: Sequence[float], metric: Metric) -> list[float]:
    return [value / 1000**2 for value in values] if metric.unit == "bytes" else list(values)


def dot_plot(plot: PlotextPlot, metric: Metric, groups: Mapping[str, Sequence[float]]) -> None:
    # One column of dots per group (an arm, or an arm's baseline), one dot per run, so the
    # spread shows next to the median the table gives.
    plt = plot.plt
    plt.clear_data()
    plt.clear_figure()
    names = [name for name, values in groups.items() if values]
    arms = [name for name in groups if not name.endswith(" (baseline)")]
    for index, name in enumerate(names):
        values = _scaled(groups[name], metric)
        color = rgb(palette().arm(arms, name.removesuffix(" (baseline)")))
        plt.scatter([index + 1] * len(values), values, marker="●" if name in arms else "○", color=color, label=name)
    plt.xticks(list(range(1, len(names) + 1)), names)
    plt.xlim(0.5, len(names) + 0.5)
    _themed(plot)
    plot.refresh()


def line_plot(plot: PlotextPlot, metric: Metric, series: dict[str, tuple[list[str], list[float]]]) -> None:
    # One line per arm across sessions: (session names, medians).
    plt = plot.plt
    plt.clear_data()
    plt.clear_figure()
    sessions = sorted({session for names, _ in series.values() for session in names})
    position = {session: index + 1 for index, session in enumerate(sessions)}
    for arm, (names, medians) in series.items():
        color = rgb(palette().arm(list(series), arm))
        plt.plot([position[name] for name in names], _scaled(medians, metric), marker="●", color=color, label=arm)
    plt.xticks(list(position.values()), [started_text(session) for session in sessions])
    _themed(plot)
    plot.refresh()
