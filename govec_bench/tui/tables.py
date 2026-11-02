"""What the live and browse views share: a comparison's rows in a DataTable, and dot plots."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from rich.text import Text
from textual.widgets import DataTable
from textual_plotext import PlotextPlot

from govec_bench.compare import Metric, Row, change_text, column_text, label_text, summary_text
from govec_bench.theme import palette, rgb


def started_text(session: str) -> str:
    # A session's stamp (20261031T013303Z, UTC) as local time: Oct 30 21:33.
    try:
        started = datetime.strptime(session, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).astimezone()
    except ValueError:
        return session
    return f"{started:%b} {started.day} {started:%H:%M}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def fill_rows(
    table: DataTable[str | Text], header: Sequence[str], rows: Sequence[Row], *, arms: Sequence[str] = ()
) -> None:
    # Keyed by metric name, so the highlighted row can be kept across a refresh.
    highlighted = table.cursor_row
    table.clear(columns=True)
    table.add_columns(Text("metric", style="bold"), *(column_text(name, arms) for name in header))
    for row in rows:
        table.add_row(
            label_text(row.metric),
            *(summary_text(summary, row.metric.unit) for summary in row.summaries.values()),
            *(change_text(item, width=0) for item in row.changes.values()),
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
