import json
from dataclasses import asdict

import pytest

from govec_bench.adapters.registry import DATABASES
from govec_bench.benchmarks.common import render_override
from govec_bench.benchmarks.ef_sweep import (
    DEFAULT_SERVICES,
    EF_CONSTRUCTION,
    EF_VALUES,
    GOVEC_DEFAULT_EF_CONSTRUCTION,
    IN_PLACE,
    K_VALUES,
    REBUILD_PER_EF,
    SeriesPlan,
    SweepFailure,
    SweepPoint,
    SweepSeries,
    format_table,
    govec_env,
    parse_sweep_args,
    plan_series,
)
from govec_bench.datasets.registry import DATASETS
from govec_bench.results import LatencyStats, RecallStats


def test_parse_sweep_args__no_flags__uses_default_ef_and_k() -> None:
    # Arrange
    argv: list[str] = []

    # Act
    args = parse_sweep_args(argv)

    # Assert
    assert args.ef_values == EF_VALUES
    assert args.k_values == K_VALUES
    assert args.bench.dataset == DATASETS["sift"]
    assert list(args.bench.databases) == list(DEFAULT_SERVICES)


def test_parse_sweep_args__db_govec_scalar__sweeps_it_though_not_a_default() -> None:
    # Arrange
    argv = ["--db", "govec-scalar"]

    # Act
    args = parse_sweep_args(argv)

    # Assert
    assert "govec-scalar" not in DEFAULT_SERVICES
    assert list(args.bench.databases) == ["govec-scalar"]


def test_parse_sweep_args__repeated_flags__sorted_without_duplicates() -> None:
    # Arrange
    argv = ["--ef", "200", "--ef", "50", "--ef", "200", "--k", "10", "--dataset", "dbpedia", "--db", "qdrant"]

    # Act
    args = parse_sweep_args(argv)

    # Assert
    assert args.ef_values == (50, 200)
    assert args.k_values == (10,)
    assert args.bench.dataset == DATASETS["dbpedia"]
    assert set(args.bench.databases) == {"qdrant"}


def test_parse_sweep_args__ef_zero__exits_with_usage_error() -> None:
    # Arrange
    argv = ["--ef", "0"]

    # Act
    with pytest.raises(SystemExit) as exc_info:
        parse_sweep_args(argv)

    # Assert
    assert exc_info.value.code == 2


def test_sweep__every_registered_database__has_a_way_to_set_ef() -> None:
    # Arrange
    swept = set(IN_PLACE) | set(REBUILD_PER_EF)

    # Act
    missing = set(DATABASES) - swept

    # Assert
    assert missing == set()


def test_plan_series__govec__adds_a_line_at_its_default_ef_construction() -> None:
    # Arrange
    services = ["govec", "qdrant"]

    # Act
    plans = plan_series(services)

    # Assert
    assert plans == [
        SeriesPlan(label="govec", service="govec", ef_construction=EF_CONSTRUCTION),
        SeriesPlan(label="govec@efc200", service="govec", ef_construction=GOVEC_DEFAULT_EF_CONSTRUCTION),
        SeriesPlan(label="qdrant", service="qdrant", ef_construction=EF_CONSTRUCTION),
    ]


def test_render_override__govec_env__sets_environment_on_that_service_only() -> None:
    # Arrange
    environment = govec_env(ef_search=400, ef_construction=100)

    # Act
    override = json.loads(render_override("govec-scalar", environment))

    # Assert
    assert override == {
        "services": {
            "govec-scalar": {
                "environment": {"GOVEC_HNSW_EF_SEARCH": "400", "GOVEC_HNSW_EF_CONSTRUCTION": "100"},
            },
        },
    }


def test_sweep_series__asdict__nests_recall_and_latency_per_point() -> None:
    # Arrange
    point = SweepPoint(
        ef=100,
        k=10,
        recall=RecallStats(mean=0.99, min=0.9, max=1.0),
        latency=LatencyStats(mean_ms=1.2, p50_ms=1.1, p99_ms=2.5),
    )
    series = SweepSeries(service="qdrant", ef_construction=100, ef_applied_by="in place", points=[point])

    # Act
    payload = asdict(series)

    # Assert
    assert payload["points"] == [
        {
            "ef": 100,
            "k": 10,
            "recall": {"mean": 0.99, "min": 0.9, "max": 1.0},
            "latency": {"mean_ms": 1.2, "p50_ms": 1.1, "p99_ms": 2.5},
        },
    ]


def test_format_table__failed_series__reported_alongside_measured_ones() -> None:
    # Arrange
    point = SweepPoint(
        ef=50,
        k=10,
        recall=RecallStats(mean=0.95, min=0.7, max=1.0),
        latency=LatencyStats(mean_ms=1.0, p50_ms=0.9, p99_ms=2.0),
    )
    results: dict[str, SweepSeries | SweepFailure] = {
        "qdrant": SweepSeries(service="qdrant", ef_construction=100, ef_applied_by="in place", points=[point]),
        "chroma": SweepFailure(service="chroma", error="RuntimeError: boom"),
    }

    # Act
    table = format_table(results)

    # Assert
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[1].split()[:4] == ["qdrant", "50", "10", "0.9500"]
    assert lines[2] == f"{'chroma':<16} FAILED: RuntimeError: boom"
