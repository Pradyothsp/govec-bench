import json
from contextlib import closing
from pathlib import Path

from govec_bench.store import connect, flatten, import_file, import_paths, metrics_of, results_files


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _pipeline_file(path: Path) -> Path:
    return _write(
        path,
        {
            "benchmark": "pipeline",
            "timestamp": "2026-10-30T04:00:00+00:00",
            "dataset": "dbpedia100k",
            "setup": {
                "govec-mmap": {
                    "service": "govec",
                    "image": "govec-bench/govec:450295fd1adf",
                    "image_id": "sha256:4ee6",
                    "govec_commit": "450295fd1adf",
                    "env": {"GOVEC_ENABLE_MMAP": "true"},
                }
            },
            "results": {
                "govec-mmap": {"run_position": 0, "queries": {"k_10": {"latency": {"mean_ms": 6.2}}}},
                "qdrant": {"run_position": 1, "error": "ReadTimeout: timed out"},
            },
        },
    )


def test_flatten__nested_numbers__dotted_names_without_strings_or_booleans() -> None:
    # Arrange
    result = {"queries": {"k_10": {"latency": {"mean_ms": 6.2}}}, "service": "govec", "ok": True, "n": 3}

    # Act
    metrics = dict(flatten(result))

    # Assert
    assert metrics == {"queries.k_10.latency.mean_ms": 6.2, "n": 3.0}


def test_flatten__sweep_points_and_ram_samples__keyed_by_what_they_are() -> None:
    # Arrange
    result = {
        "points": [{"ef": 100, "k": 10, "recall": {"mean": 0.99}}],
        "ram_series": [{"t_s": 60, "ram_bytes": 5}],
        "unkeyed": [1, 2, 3],
    }

    # Act
    metrics = dict(flatten(result))

    # Assert
    assert metrics == {"points.ef_100.k_10.recall.mean": 0.99, "ram_series.t_60s.ram_bytes": 5.0}


def test_metrics_of__pipeline_memory__adds_what_the_load_added_at_the_last_sample() -> None:
    # Arrange
    result = {
        "memory": {
            "baseline": {"ram_bytes": 50, "ram_process_bytes": 20, "ram_file_cache_bytes": 10},
            "ram_series": [
                {"t_s": 0, "ram_bytes": 900, "process_bytes": 200, "file_cache_bytes": 600},
                {"t_s": 60, "ram_bytes": 300, "process_bytes": 170, "file_cache_bytes": 640},
            ],
        }
    }

    # Act
    metrics = metrics_of(result)

    # Assert
    assert metrics["memory.ram_added_bytes"] == 250
    assert metrics["memory.process_added_bytes"] == 150
    assert metrics["memory.file_cache_added_bytes"] == 630


def test_import_file__pipeline__records_the_run_what_each_database_ran_as_and_its_metrics(tmp_path: Path) -> None:
    # Arrange
    path = _pipeline_file(tmp_path / "pipeline_1.json")

    # Act
    with closing(connect(tmp_path / "bench.db")) as db:
        imported = import_file(db, path)
        run = db.execute("SELECT benchmark, dataset, label FROM runs WHERE id = ?", (imported.run_id,)).fetchone()
        entries = db.execute(
            "SELECT database, govec_commit, env, error FROM entries WHERE run_id = ? ORDER BY database",
            (imported.run_id,),
        ).fetchall()
        value = db.execute(
            "SELECT value FROM metrics WHERE database = 'govec-mmap' AND metric = 'queries.k_10.latency.mean_ms'"
        ).fetchone()

    # Assert
    assert imported.new
    assert run == ("pipeline", "dbpedia100k", "pipeline_1")
    assert entries == [
        ("govec-mmap", "450295fd1adf", '{"GOVEC_ENABLE_MMAP": "true"}', None),
        ("qdrant", None, None, "ReadTimeout: timed out"),
    ]
    assert value == (6.2,)


def test_import_paths__same_file_twice_from_two_places__one_run(tmp_path: Path) -> None:
    # Arrange
    first = _pipeline_file(tmp_path / "pipeline_1.json")
    copy = tmp_path / "notes" / "r1.json"
    copy.parent.mkdir()
    copy.write_bytes(first.read_bytes())
    db_path = tmp_path / "bench.db"

    # Act
    imported = import_paths(db_path, [first, tmp_path / "notes"])
    again = import_paths(db_path, [tmp_path])
    with closing(connect(db_path)) as db:
        runs = db.execute("SELECT count(*) FROM runs").fetchone()

    # Assert
    assert [item.new for item in imported] == [True, False]
    assert not any(item.new for item in again)
    assert runs == (1,)


def test_import_file__sweep_series_under_a_label__finds_its_database_setup(tmp_path: Path) -> None:
    # Arrange
    path = _write(
        tmp_path / "ef_sweep_1.json",
        {
            "benchmark": "ef_sweep",
            "dataset": "sift10k",
            "setup": {
                "govec": {
                    "service": "govec",
                    "image": "ghcr.io/pradyothsp/govec:0.2.1",
                    "image_id": "sha256:3e7f",
                    "govec_commit": None,
                    "env": {},
                }
            },
            "results": {"govec@efc200": {"service": "govec", "ef_construction": 200, "points": []}},
        },
    )

    # Act
    with closing(connect(tmp_path / "bench.db")) as db:
        import_file(db, path)
        image = db.execute("SELECT image FROM entries WHERE database = 'govec@efc200'").fetchone()

    # Assert
    assert image == ("ghcr.io/pradyothsp/govec:0.2.1",)


def test_results_files__folder__only_files_with_benchmark_and_results(tmp_path: Path) -> None:
    # Arrange
    wanted = _write(tmp_path / "a.json", {"benchmark": "recall", "results": {}})
    _write(tmp_path / "config.json", {"services": {}})
    (tmp_path / "broken.json").write_text("{not json")

    # Act
    found = list(results_files([tmp_path]))

    # Assert
    assert found == [wanted]
