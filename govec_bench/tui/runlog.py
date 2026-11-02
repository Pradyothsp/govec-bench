"""Making sense of a session's run.log in the live view: progress bars apart, lines in colour.

run.log is what the runner prints: dated session lines, the pipeline's stages, steps and their
results, and tqdm's progress bars. A bar redraws itself with \\r on a terminal and prints a line
per update when it isn't one, so the view takes bar lines out of the log and shows the newest as
a bar of its own.
"""

import re
from typing import NamedTuple

from rich.text import Text

from govec_bench.tui.theme import palette

# tqdm's default layout: "desc:  64%|######4   | 640/1000 [03:09<18:30,  3.08s/batch]"
_PROGRESS = re.compile(
    r"^(?P<desc>[^:]+):\s+(?P<percent>\d+)%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)"
    r"\s*\[(?P<elapsed>[\d:]+)<(?P<left>[\d:?]+)(?:,\s*(?P<rate>[^\]]*))?\]"
)
_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} (?P<time>\d{2}:\d{2}:\d{2})\] ")
_FAILURE = re.compile(r"\b(failed|error|traceback|timed out|interrupted)\b", re.IGNORECASE)


class Progress(NamedTuple):
    desc: str
    done: int
    total: int
    elapsed: str
    left: str  # "?" until tqdm has a rate
    rate: str


def last_segments(text: str) -> list[str]:
    # run.log's lines, each as a terminal would leave it: a progress bar redraws itself with \r.
    return [line.rsplit("\r", 1)[-1] for line in text.split("\n")]


def parse_progress(line: str) -> Progress | None:
    found = _PROGRESS.match(line.strip())
    if found is None:
        return None
    return Progress(
        desc=found["desc"].strip(),
        done=int(found["done"]),
        total=int(found["total"]),
        elapsed=found["elapsed"],
        left=found["left"],
        rate=re.sub(r"(\d)([a-z?])", r"\1 \2", (found["rate"] or "").strip()),  # 3.08s/batch: 3.08 s/batch
    )


def styled_line(line: str, arms: list[str]) -> Text:
    # One run.log line coloured by what it is: the session's own dated lines, the pipeline's
    # stages ("  [loaded index]"), steps ("    batch insert..."), results ("      -> ..."), and
    # anything that failed.
    colors, text = palette(), Text()
    stamp = _STAMP.match(line)
    if stamp:
        text.append(f"{stamp['time']} ", style="dim")
        line = line[stamp.end() :]

    body = line.strip()
    if _FAILURE.search(body) and not body.startswith("->"):
        text.append(line, style=f"bold {colors.bad}")
    elif body.startswith("round "):
        text.append(line, style=f"bold {colors.text} on {colors.band}")
    elif body.endswith(":") and body[:-1] in arms:
        text.append(line, style=f"bold {colors.arm(arms, body[:-1])}")
    elif " done -> " in body:
        name = body.split(" done -> ", 1)[0]
        text.append(line, style=colors.arm(arms, name))
        text.append("  ✓", style=f"bold {colors.good}")
    elif body.startswith("[") and body.endswith("]"):
        text.append(line, style=colors.stage)
    elif body.startswith("->"):
        text.append(line, style=colors.result)
    elif body.endswith("..."):
        text.append(line, style="bold")
    else:
        text.append(line, style=colors.muted if stamp else "")
    return text
