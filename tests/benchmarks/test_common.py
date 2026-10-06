import pytest

from govec_bench.adapters.registry import DATABASES
from govec_bench.benchmarks.common import parse_args
from govec_bench.datasets.registry import DATASETS


def test_parse_args__no_flags__selects_sift_and_every_database() -> None:
    # Arrange
    argv: list[str] = []

    # Act
    args = parse_args(argv)

    # Assert
    assert args.dataset == DATASETS["sift"]
    assert args.databases == DATABASES


def test_parse_args__dataset_flag__selects_that_dataset() -> None:
    # Arrange
    argv = ["--dataset", "dbpedia"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.dataset == DATASETS["dbpedia"]


def test_parse_args__db_flags__selects_only_those_databases() -> None:
    # Arrange
    argv = ["--db", "qdrant", "--db", "govec"]

    # Act
    args = parse_args(argv)

    # Assert
    assert set(args.databases) == {"govec", "qdrant"}


def test_parse_args__unknown_db__exits_with_usage_error() -> None:
    # Arrange
    argv = ["--db", "milvus"]

    # Act
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    # Assert
    assert exc_info.value.code == 2
