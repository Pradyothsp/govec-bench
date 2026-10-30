import json
import random
from contextlib import closing
from pathlib import Path

import pytest

from govec_bench.benchmarks import experiment as experiment_module
from govec_bench.benchmarks.experiment import Session, parse_experiment, plan, round_order, run_rounds
from govec_bench.benchmarks.pipeline import Workload
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
    monkeypatch.setattr(experiment_module, "wait_for_disk", lambda: True)
    measured: list[tuple[str | None, int]] = []

    def fake_measure(database, workload, position):  # noqa: ARG001
        measured.append((database.image_ref, position))
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
