import json
import os
import random
from contextlib import closing
from pathlib import Path

import pytest

from govec_bench.benchmarks import experiment as experiment_module
from govec_bench.benchmarks.experiment import (
    Session,
    find_session_folder,
    parse_experiment,
    plan,
    round_order,
    run_rounds,
)
from govec_bench.benchmarks.pipeline import Workload
from govec_bench.events import read_events
from govec_bench.store import connect

AB = """
description: 0.2.1 against HEAD
dataset: dbpedia
size: large
rounds: 2
arms:
  - name: old
    database: govec-mmap
    govec: ghcr.io/pradyothsp/govec:0.2.1
  - name: new
    database: govec-mmap
    govec: HEAD
    set: {hnsw_m: 32, enable_wal: false}
"""


def test_parse_experiment__arms__database_with_the_arms_image_and_settings() -> None:
    # Arrange
    text = AB

    # Act
    experiment = parse_experiment("ab", text)

    # Assert
    old, new = experiment.arms
    assert (experiment.dataset, experiment.size, experiment.rounds, experiment.order) == (
        "dbpedia",
        "large",
        2,
        "alternate",
    )
    assert old.database.image_ref == "ghcr.io/pradyothsp/govec:0.2.1"
    assert old.database.env == {"GOVEC_ENABLE_MMAP": "true"}
    assert new.database.image_ref == "HEAD"
    assert new.database.env == {"GOVEC_ENABLE_MMAP": "true", "GOVEC_HNSW_M": "32", "GOVEC_ENABLE_WAL": "false"}
    assert new.database.name == "govec-mmap"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("arms: [{name: a, database: qdrant, govec: HEAD}]", "govec and set change GoVec"),
        ("arms: [{name: a, database: nope}]", "database 'nope'"),
        ("arms: [{name: a, database: govec}, {name: a, database: qdrant}]", "must be unique"),
        ("arms: [{name: a, database: govec}]\norder: random", "order 'random'"),
        ("arms: [{name: a, database: govec}]\nrounds: 0", "positive number"),
        ("arms: [{name: a, database: govec}]\nround: 3", "unknown keys ['round']"),
        ("dataset: sift", "needs a list of arms"),
    ],
)
def test_parse_experiment__invalid__says_what_is_wrong(text: str, message: str) -> None:
    # Arrange
    name = "bad"

    # Act
    with pytest.raises((ValueError, TypeError)) as raised:
        parse_experiment(name, text)

    # Assert
    assert message in str(raised.value)


def test_plan__alternate__reverses_every_other_round() -> None:
    # Arrange
    experiment = parse_experiment("ab", AB)

    # Act
    rounds = plan(experiment, 3, random.Random(0))  # noqa: S311

    # Assert
    assert rounds == [["old", "new"], ["new", "old"], ["old", "new"]]


def test_round_order__shuffle__every_arm_once() -> None:
    # Arrange
    names = ["a", "b", "c", "d"]

    # Act
    order = round_order(names, 2, "shuffle", random.Random(1))  # noqa: S311

    # Assert
    assert sorted(order) == names


def test_run_rounds__each_arm_each_round__a_results_file_and_the_session_says_what_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_module, "wait_for_disk", lambda _warn: True)
    measured: list[tuple[str | None, int]] = []

    def fake_measure(database, workload, position, progress):  # noqa: ARG001
        measured.append((database.image_ref, position))
        progress("step", "batch insert")
        if database.image_ref == "HEAD" and len(measured) > 2:
            return {"run_position": position, "error": "TimeoutError: slow"}
        return {"run_position": position, "batch_insert": {"per_vector_ms": 0.5}}

    monkeypatch.setattr(experiment_module, "measure_database", fake_measure)
    experiment = parse_experiment("ab", AB)
    rounds = plan(experiment, 2, random.Random(0))  # noqa: S311
    directory = tmp_path / "session"
    directory.mkdir()
    session = Session(experiment, directory, rounds)
    workload = Workload(base=None, queries=[], groundtruth=[], single_insert_items=None, dimensions=1536)  # ty: ignore[invalid-argument-type]

    # Act
    run_rounds(session, rounds, workload, "dbpedia100k")
    state = json.loads((directory / "session.json").read_text())
    events, _ = read_events(directory / "events.jsonl")
    with closing(connect(tmp_path / "results" / "bench.db")) as db:
        indexed = db.execute(
            "SELECT experiment, session, round, arm, position FROM runs ORDER BY round, position"
        ).fetchall()

    # Assert
    assert measured == [
        ("ghcr.io/pradyothsp/govec:0.2.1", 0),
        ("HEAD", 1),
        ("HEAD", 0),
        ("ghcr.io/pradyothsp/govec:0.2.1", 1),
    ]
    assert sorted(path.name for path in directory.glob("r*.json")) == [
        "r1-new.json",
        "r1-old.json",
        "r2-new.json",
        "r2-old.json",
    ]
    assert [(run["round"], run["arm"], run["status"]) for run in state["runs"]] == [
        (1, "old", "done"),
        (1, "new", "done"),
        (2, "new", "failed"),
        (2, "old", "done"),
    ]
    assert indexed == [
        ("ab", "session", 1, "old", 0),
        ("ab", "session", 1, "new", 1),
        ("ab", "session", 2, "new", 0),
        ("ab", "session", 2, "old", 1),
    ]
    assert [(event.kind, event.arm, event.text or event.status) for event in events[:4]] == [
        ("round", None, None),
        ("arm", "old", None),
        ("step", "old", "batch insert"),
        ("result", "old", "done"),
    ]
    assert events[-1].file == "r2-old.json"


def test_logged_to__echo_off__output_in_run_log_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    path = tmp_path / "run.log"

    # Act
    with experiment_module.logged_to(path, echo=False):
        print("batch insert...")  # noqa: T201 -- what the runner prints

    # Assert
    assert path.read_text() == "batch insert...\n"
    assert capsys.readouterr().out == ""


def _session_folder(sessions: Path, name: str, status: str, pid: int | None) -> Path:
    folder = sessions / name
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps({"status": status, "pid": pid}))
    return folder


def test_find_session_folder__nothing_named__the_newest_still_running(tmp_path: Path) -> None:
    # Arrange
    _session_folder(tmp_path, "ab/20261029T010000Z", "running", os.getpid())
    newest = _session_folder(tmp_path, "hint/20261030T010000Z", "running", os.getpid())
    _session_folder(tmp_path, "hint/20261030T020000Z", "done", os.getpid())
    _session_folder(tmp_path, "hint/20261030T030000Z", "running", None)  # its runner died

    # Act
    folder = find_session_folder(None, tmp_path)

    # Assert
    assert folder == newest


def test_find_session_folder__experiment_name__its_newest_session(tmp_path: Path) -> None:
    # Arrange
    _session_folder(tmp_path, "ab/20261029T010000Z", "done", None)
    newest = _session_folder(tmp_path, "ab/20261030T010000Z", "done", None)

    # Act
    folders = [find_session_folder(text, tmp_path) for text in ("ab", "ab/20261030T010000Z", str(newest))]

    # Assert
    assert folders == [newest, newest, newest]


def test_find_session_folder__nothing_running__says_so(tmp_path: Path) -> None:
    # Arrange
    _session_folder(tmp_path, "ab/20261029T010000Z", "done", None)

    # Act
    with pytest.raises(SystemExit) as raised:
        find_session_folder(None, tmp_path)

    # Assert
    assert "no session is running" in str(raised.value)
