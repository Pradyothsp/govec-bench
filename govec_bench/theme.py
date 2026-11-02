"""The colours of bench's terminal output, by what they mean, so a theme is one entry in THEMES.

The terminal app (tui/) and the reports bench compare and bench runs print both colour from
here: the log, the cards, verdicts, plan marks, plots and report tables. Textual styles its own
widgets (borders, tables, the footer) from a Textual theme; a palette names the one it was
picked to sit on. use() switches the palette; everything reads palette() as it draws.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    textual: str  # the Textual theme for the widgets' own colours
    background: str  # the screen's: plot canvases, and text on an arm's chip
    text: str
    muted: str  # secondary text: a session's date and runs, an undated log line
    faint: str  # plot ticks and axes
    track: str  # the empty part of a bar
    band: str  # behind a round's header in the log
    good: str  # done, better
    bad: str  # failed, worse
    caution: str  # running, warnings, higher or lower with no better
    info: str  # notes about a baseline
    stage: str  # a pipeline stage in the log
    result: str  # a measured result in the log
    arms: tuple[str, ...]  # each arm's colour, in the order the plan names them

    def arm(self, arms: Sequence[str], name: str) -> str:
        # The same arm the same colour in every chip, column, plot and log line.
        return self.arms[list(arms).index(name) % len(self.arms)] if name in arms else self.text

    def verdict(self, verdict: str) -> str:
        # A comparison's verdict ("every run better", "overlaps") by what it says.
        for word, style in (
            ("better", self.good),
            ("worse", self.bad),
            ("higher", self.caution),
            ("lower", self.caution),
            ("level", "dim"),
            ("overlaps", "dim"),
        ):
            if word in verdict:
                return style
        return ""


def rgb(color: str) -> tuple[int, int, int]:
    # For plotext, which takes colours as RGB tuples.
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


THEMES = {
    "tokyo-night": Palette(
        textual="tokyo-night",
        background="#1a1b26",
        text="#c0caf5",
        muted="#a9b1d6",
        faint="#565f89",
        track="#3b4261",
        band="#292e42",
        good="#9ece6a",
        bad="#f7768e",
        caution="#e0af68",
        info="#7dcfff",
        stage="#bb9af7",
        result="#73daca",
        arms=("#7aa2f7", "#e0af68", "#bb9af7", "#73daca", "#f7768e", "#9ece6a"),
    ),
}

DEFAULT_THEME = "tokyo-night"

_active = THEMES[DEFAULT_THEME]


def palette() -> Palette:
    return _active


def use(name: str) -> Palette:
    global _active  # noqa: PLW0603 -- one palette for the whole app, picked at start
    if name not in THEMES:
        msg = f"no theme {name!r}; the themes are {', '.join(THEMES)}"
        raise ValueError(msg)
    _active = THEMES[name]
    return _active
