import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from govec_bench.results_file import ResultsFile, StartedAs
from govec_bench.store import connect, import_file

HEAD_SETUP: StartedAs = {
    "service": "govec",
    "image": "govec-bench/govec:450295fd1adf",
    "image_id": "sha256:4ee6",
    "govec_commit": "450295fd1adf",
    "env": {"GOVEC_ENABLE_MMAP": "true"},
}
OLD_SETUP: StartedAs = {
    **HEAD_SETUP,
    "image": "ghcr.io/pradyothsp/govec:0.2.1",
    "image_id": "sha256:3e7f",
    "govec_commit": None,
}


def _run_file(directory: Path, *, session: str, round_number: int, arm: str, position: int, restart_ms: float) -> Path:
    old = arm == "old"
    payload: ResultsFile = {
        "benchmark": "pipeline",
        "timestamp": f"2026-10-{29 + (session == 's2')}T0{round_number}:0{position}:00+00:00",
        "dataset": "dbpedia100k",
        "setup": {"govec-mmap": OLD_SETUP if old else HEAD_SETUP},
        "results": {"govec-mmap": {"run_position": position, "cold_start_loaded": {"p50_ms": restart_ms}}},
        "experiment": {"name": "ab", "session": session, "round": round_number, "arm": arm, "position": position},
    }
    path = directory / f"{session}-r{round_number}-{arm}.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    # Two sessions of an old-against-new A/B: in the second the old arm, unchanged, is 10% slower.
    connection = connect(tmp_path / "bench.db")
    for session, slower in (("s1", 1.0), ("s2", 1.1)):
        for round_number, order in ((1, ("old", "new")), (2, ("new", "old")), (3, ("old", "new"))):
            for position, arm in enumerate(order):
                restart = (490 + round_number) * slower if arm == "old" else 550 + round_number
                import_file(
                    connection,
                    _run_file(
                        tmp_path,
                        session=session,
                        round_number=round_number,
                        arm=arm,
                        position=position,
                        restart_ms=restart,
                    ),
                )
    yield connection
    connection.close()
