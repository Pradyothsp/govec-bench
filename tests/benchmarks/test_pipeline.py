import pytest

from govec_bench.adapters.registry import select_databases
from govec_bench.benchmarks.insert import InsertStats
from govec_bench.benchmarks.memory import FootprintStats, MemoryResult, RamSample
from govec_bench.benchmarks.pipeline import (
    PIPELINE,
    QueryPoint,
    Record,
    parse_pipeline_args,
    result_to_json,
    shuffled,
)
from govec_bench.datasets.registry import DATASETS
from govec_bench.results import LatencyStats, RecallStats


def _latency(ms: float) -> LatencyStats:
    return LatencyStats(mean_ms=ms, p50_ms=ms, p99_ms=ms)


def _insert(ms: float) -> InsertStats:
    return InsertStats(total_s=ms * 10, per_vector_ms=ms, calls=_latency(ms))


def test_parse_pipeline_args__no_flags__small_size_and_shared_defaults() -> None:
    # Arrange
    argv: list[str] = []

    # Act
    args = parse_pipeline_args(argv)

    # Assert
    assert args.size == "small"
    assert args.bench.dataset == DATASETS["sift"]
    assert args.bench.databases == select_databases()


def test_parse_pipeline_args__size_large__selects_large() -> None:
    # Arrange
    argv = ["--size", "large", "--dataset", "dbpedia", "--db", "qdrant"]

    # Act
    args = parse_pipeline_args(argv)

    # Assert
    assert args.size == "large"
    assert args.bench.dataset == DATASETS["dbpedia"]
    assert set(args.bench.databases) == {"qdrant"}


def test_parse_pipeline_args__unknown_size__exits_with_usage_error() -> None:
    # Arrange
    argv = ["--size", "huge"]

    # Act
    with pytest.raises(SystemExit) as exc_info:
        parse_pipeline_args(argv)

    # Assert
    assert exc_info.value.code == 2


def test_pipeline__loaded_stage__measures_memory_before_queries_and_restarts_last() -> None:
    # Arrange
    loaded = PIPELINE[0]

    # Act
    labels = [step.label for step in loaded.steps]

    # Assert
    assert labels[:2] == ["baseline footprint", "batch insert"]
    assert labels.index("disk and RAM (RAM sampled for 60 s)") < labels.index("recall and query latency")
    assert labels[-1].endswith("restarts with the data loaded")


def test_result_to_json__full_record__every_metric_under_its_own_key() -> None:
    # Arrange
    record = Record(
        batch_insert=_insert(1.0),
        memory=MemoryResult(
            baseline=FootprintStats(
                disk_bytes=0, disk_allocated_bytes=0, ram_bytes=10, ram_process_bytes=8, ram_file_cache_bytes=2
            ),
            delta_disk_bytes=5,
            delta_disk_allocated_bytes=4,
            ram_series=[RamSample(t_s=0, ram_bytes=30, process_bytes=25, file_cache_bytes=5)],
            delta_ram_bytes_peak=20,
            delta_ram_bytes_last_sample=20,
        ),
        queries=[QueryPoint(k=10, recall=RecallStats(mean=0.99, min=0.8, max=1.0), latency=_latency(2.0))],
        cold_start_loaded=_latency(500.0),
        cold_start_empty=_latency(8.0),
        single_insert=_insert(3.0),
    )

    # Act
    payload = result_to_json(record, run_position=2)

    # Assert
    assert set(payload) == {
        "run_position",
        "batch_insert",
        "memory",
        "queries",
        "cold_start_loaded",
        "cold_start_empty",
        "single_insert",
    }
    assert payload["queries"] == {
        "k_10": {
            "recall": {"mean": 0.99, "min": 0.8, "max": 1.0},
            "latency": {"mean_ms": 2.0, "p50_ms": 2.0, "p99_ms": 2.0},
        },
    }


def test_result_to_json__failed_partway__keeps_what_was_measured_and_the_error() -> None:
    # Arrange
    record = Record(batch_insert=_insert(1.0))

    # Act
    payload = result_to_json(record, run_position=0, error="TimeoutError: boom")

    # Assert
    assert payload == {
        "run_position": 0,
        "error": "TimeoutError: boom",
        "batch_insert": {
            "total_s": 10.0,
            "per_vector_ms": 1.0,
            "calls": {"mean_ms": 1.0, "p50_ms": 1.0, "p99_ms": 1.0},
        },
    }


def test_pipeline__grpc_queries__run_right_after_the_rest_ones_on_the_same_index() -> None:
    # Arrange
    loaded = PIPELINE[0]

    # Act
    labels = [step.label for step in loaded.steps]

    # Assert
    rest = labels.index("recall and query latency")
    assert labels[rest + 1] == "recall and query latency over gRPC"


def test_result_to_json__grpc_queries__under_their_own_key() -> None:
    # Arrange
    point = QueryPoint(k=10, recall=RecallStats(mean=0.99, min=0.8, max=1.0), latency=_latency(1.5))
    record = Record(queries=[point], queries_grpc=[point])

    # Act
    payload = result_to_json(record, run_position=0)

    # Assert
    assert payload["queries_grpc"] == payload["queries"]


def test_shuffled__database_names__every_name_once() -> None:
    # Arrange
    names = list(select_databases())

    # Act
    order = shuffled(names)

    # Assert
    assert sorted(order) == sorted(names)
    assert names == list(select_databases())
