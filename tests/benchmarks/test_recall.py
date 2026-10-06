import pytest

from govec_bench.adapters.registry import select_databases
from govec_bench.benchmarks.recall import parse_recall_args
from govec_bench.datasets.registry import DATASETS


def test_parse_recall_args__no_flags__small_size_and_shared_defaults() -> None:
    # Arrange
    argv: list[str] = []

    # Act
    args = parse_recall_args(argv)

    # Assert
    assert args.size == "small"
    assert args.bench.dataset == DATASETS["sift"]
    assert args.bench.databases == select_databases()


def test_parse_recall_args__size_large__selects_large() -> None:
    # Arrange
    argv = ["--size", "large", "--dataset", "dbpedia", "--db", "govec"]

    # Act
    args = parse_recall_args(argv)

    # Assert
    assert args.size == "large"
    assert args.bench.dataset == DATASETS["dbpedia"]
    assert set(args.bench.databases) == {"govec"}


def test_parse_recall_args__unknown_size__exits_with_usage_error() -> None:
    # Arrange
    argv = ["--size", "huge"]

    # Act
    with pytest.raises(SystemExit) as exc_info:
        parse_recall_args(argv)

    # Assert
    assert exc_info.value.code == 2
