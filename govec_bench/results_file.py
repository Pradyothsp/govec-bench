"""The shape of a results file: what results.write_results() writes and store.py reads."""

from typing import NotRequired, TypedDict


class StartedAs(TypedDict):
    # What a database ran as, read from its started container (common.running_alone()).
    service: str
    image: str
    image_id: str
    govec_commit: str | None  # None for an image used as it is
    env: dict[str, str]  # the database's own settings, by GOVEC_ variable


class RunTag(TypedDict):
    # Which run of an experiment a results file is, when bench run wrote it.
    name: str
    session: str
    round: int
    arm: str
    position: int


class ResultsFile(TypedDict):
    benchmark: str
    timestamp: str
    dataset: str
    setup: dict[str, StartedAs]  # by database name
    # By database (or ef sweep series) name: whatever that benchmark measured, as JSON.
    results: dict[str, object]
    experiment: NotRequired[RunTag]
