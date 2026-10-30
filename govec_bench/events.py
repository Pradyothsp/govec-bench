"""What a running session is doing, as one JSON line per event in its events.jsonl.

bench run writes them as it goes; the live view (tui/live.py) tails the file, so the run and the
view are separate processes: closing the view, or a terminal dropping, doesn't stop a night, and
a view can attach to a session already running. session.json says what each run produced; this
says what is happening now: the round, the arm, the step, warnings.
"""

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

EVENTS_FILE = "events.jsonl"

EventKind = Literal[
    "round",  # a round starts: round
    "arm",  # an arm's run starts: round, arm, position
    "stage",  # a pipeline stage starts, in a fresh container: arm, text (the stage)
    "step",  # a step of the pipeline starts: arm, text (the step)
    "result",  # an arm's run ended: round, arm, status, file
    "warning",  # something the person watching should know: text
    "end",  # the session ended: status
]


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    round: int | None = None
    arm: str | None = None
    position: int | None = None
    text: str | None = None
    status: str | None = None
    file: str | None = None
    t: float = field(default_factory=time.time)  # when it happened


class EventLog:
    def __init__(self, directory: Path) -> None:
        self.path = directory / EVENTS_FILE

    def emit(self, event: Event) -> None:
        line = json.dumps({key: value for key, value in asdict(event).items() if value is not None})
        # One write per line, flushed at once: a reader never sees half a line it can't parse later.
        with self.path.open("a") as file:
            file.write(line + "\n")


_FIELDS = {item.name for item in fields(Event)}


def read_events(path: Path, offset: int = 0) -> tuple[list[Event], int]:
    # The events after offset, and the offset to read from next. A line still being written (no
    # newline yet) is left for the next read.
    try:
        with path.open("rb") as file:
            file.seek(offset)
            data = file.read()
    except FileNotFoundError:
        return [], offset

    complete = data[: data.rfind(b"\n") + 1]
    events = [
        Event(**{key: value for key, value in json.loads(line).items() if key in _FIELDS})
        for line in complete.splitlines()
        if line.strip()
    ]
    return events, offset + len(complete)
