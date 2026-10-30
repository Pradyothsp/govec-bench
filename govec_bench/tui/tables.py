"""What the live and browse views share: a comparison's rows in a DataTable, and dot plots."""

from collections.abc import Mapping, Sequence

from rich.text import Text
from textual.widgets import DataTable
from textual_plotext import PlotextPlot

from govec_bench.compare import Change, Metric, Row, label_of, summary_text

_VERDICT_STYLES = {"better": "green", "worse": "red", "level": "dim", "overlaps": "dim"}


def change_cell(item: Change | None) -> Text:
    if item is None:
        return Text("")
    moved = f"{item.moved:+.1%}  " if item.moved is not None else ""
    style = next((style for word, style in _VERDICT_STYLES.items() if word in item.verdict), "")
    return Text(f"{moved}{item.verdict}", style=style)


def fill_rows(table: DataTable[str | Text], header: Sequence[str], rows: Sequence[Row]) -> None:
    # Keyed by metric name, so the highlighted row can be kept across a refresh.
    highlighted = table.cursor_row
    table.clear(columns=True)
    table.add_columns("", *header)
    for row in rows:
        label = label_of(row.metric) + (" !" if row.metric.caution else "")
        table.add_row(
            label,
            *(summary_text(summary, row.metric.unit) for summary in row.summaries.values()),
            *(change_cell(item) for item in row.changes.values()),
            key=row.metric.name,
        )
    if rows:
        table.move_cursor(row=min(highlighted, len(rows) - 1))


def _scaled(values: Sequence[float], metric: Metric) -> list[float]:
    return [value / 1000**2 for value in values] if metric.unit == "bytes" else list(values)


def dot_plot(plot: PlotextPlot, metric: Metric, groups: Mapping[str, Sequence[float]]) -> None:
    # One column of dots per group (an arm, or an arm's baseline), one dot per run, so the
    # spread shows next to the median the table gives.
    plt = plot.plt
    plt.clear_data()
    plt.clear_figure()
    names = [name for name, values in groups.items() if values]
    for index, name in enumerate(names):
        values = _scaled(groups[name], metric)
        plt.scatter([index + 1] * len(values), values, marker="dot", label=name)
    plt.xticks(list(range(1, len(names) + 1)), names)
    plt.xlim(0.5, len(names) + 0.5)
    plt.title(label_of(metric))
    plot.refresh()


def line_plot(plot: PlotextPlot, metric: Metric, series: dict[str, tuple[list[str], list[float]]]) -> None:
    # One line per arm across sessions: (session names, medians).
    plt = plot.plt
    plt.clear_data()
    plt.clear_figure()
    sessions = sorted({session for names, _ in series.values() for session in names})
    position = {session: index + 1 for index, session in enumerate(sessions)}
    for arm, (names, medians) in series.items():
        plt.plot([position[name] for name in names], _scaled(medians, metric), marker="dot", label=arm)
    plt.xticks(list(position.values()), [session[:8] for session in sessions])
    plt.title(f"{label_of(metric)}, median per session")
    plot.refresh()
