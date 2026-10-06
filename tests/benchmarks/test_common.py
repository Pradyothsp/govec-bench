import pytest

from govec_bench.adapters.registry import DATABASES, select_databases
from govec_bench.benchmarks.common import (
    MIN_FREE_DISK_BYTES,
    FreeSpace,
    parse_args,
    parse_df_available,
    running_alone,
    shortfalls,
)
from govec_bench.datasets.registry import DATASETS


def test_parse_args__no_flags__selects_sift_and_every_database() -> None:
    # Arrange
    argv: list[str] = []

    # Act
    args = parse_args(argv)

    # Assert
    assert args.dataset == DATASETS["sift"]
    assert args.databases == select_databases()


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


def test_running_alone__needs_dimensions_without_them__refuses_before_starting_anything() -> None:
    # Arrange
    database = DATABASES["govec-mmap"]

    # Act
    with pytest.raises(ValueError, match="vector width at startup") as exc_info, running_alone("govec-mmap", database):
        pass

    # Assert
    assert "govec-mmap" in str(exc_info.value)


@pytest.mark.parametrize(
    "output",
    [
        # GNU coreutils
        "Filesystem     1024-blocks     Used Available Capacity Mounted on\noverlay 61202244 20000000 41202244 33% /\n",
        # BusyBox (the GoVec image is Alpine)
        (
            "Filesystem           1024-blocks    Used Available Capacity Mounted on\n"
            "overlay               61202244  20000000  41202244  33% /\n"
        ),
    ],
)
def test_parse_df_available__posix_layout__available_kib_as_bytes(output: str) -> None:
    # Arrange
    expected = 41202244 * 1024

    # Act
    available = parse_df_available(output)

    # Assert
    assert available == expected


def test_shortfalls__one_disk_below_minimum__returns_only_that_one() -> None:
    # Arrange
    spaces = [FreeSpace("Docker's disk", 50 * 1000**3), FreeSpace("the Mac's disk", 20 * 1000**3)]

    # Act
    short = shortfalls(spaces, MIN_FREE_DISK_BYTES)

    # Assert
    assert short == [FreeSpace("the Mac's disk", 20 * 1000**3)]
