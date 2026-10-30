import json

import pytest

from govec_bench.adapters.registry import DATABASES, select_databases
from govec_bench.benchmarks.common import (
    MIN_FREE_DISK_BYTES,
    FreeSpace,
    parse_args,
    parse_df_available,
    render_override,
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


def test_parse_args__no_govec_flags__runs_the_pinned_image_on_file_settings() -> None:
    # Arrange
    argv = ["--db", "govec-mmap"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.databases["govec-mmap"] == DATABASES["govec-mmap"]
    assert args.databases["govec-mmap"].image_ref is None


def test_parse_args__govec_and_set__apply_to_every_govec_database_only() -> None:
    # Arrange
    argv = [
        "--db",
        "govec",
        "--db",
        "govec-mmap",
        "--db",
        "qdrant",
        "--govec",
        "HEAD",
        "--set",
        "hnsw_ef_construction=200",
    ]

    # Act
    args = parse_args(argv)

    # Assert
    govec, mmap, qdrant = args.databases["govec"], args.databases["govec-mmap"], args.databases["qdrant"]
    assert govec.env == {"GOVEC_HNSW_EF_CONSTRUCTION": "200"}
    assert mmap.env == {"GOVEC_ENABLE_MMAP": "true", "GOVEC_HNSW_EF_CONSTRUCTION": "200"}
    assert govec.image_ref == mmap.image_ref == "HEAD"
    assert qdrant == DATABASES["qdrant"]


def test_parse_args__set_overrides_a_file_setting__run_setting_wins() -> None:
    # Arrange
    argv = ["--db", "govec-mmap", "--set", "enable_mmap=false"]

    # Act
    args = parse_args(argv)

    # Assert
    assert args.databases["govec-mmap"].env["GOVEC_ENABLE_MMAP"] == "false"


@pytest.mark.parametrize("setting", ["hnsw_m", "=16", "hnsw_m="])
def test_parse_args__set_without_key_and_value__exits_with_usage_error(setting: str) -> None:
    # Arrange
    argv = ["--db", "govec", "--set", setting]

    # Act
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    # Assert
    assert exc_info.value.code == 2


def test_parse_args__govec_without_a_govec_database__refuses() -> None:
    # Arrange
    argv = ["--db", "qdrant", "--govec", "HEAD"]

    # Act
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    # Assert
    assert "no GoVec database" in str(exc_info.value.code)


def test_render_override__environment_and_image__set_on_that_service_only() -> None:
    # Arrange
    settings: dict[str, object] = {"environment": {"GOVEC_ENABLE_MMAP": "true"}, "image": "govec-bench/govec:abc"}

    # Act
    override = json.loads(render_override("govec", settings))

    # Assert
    assert override == {
        "services": {"govec": {"environment": {"GOVEC_ENABLE_MMAP": "true"}, "image": "govec-bench/govec:abc"}}
    }


def test_running_alone__needs_dimensions_without_them__refuses_before_starting_anything() -> None:
    # Arrange
    database = DATABASES["govec-mmap"]

    # Act
    with pytest.raises(ValueError, match="vector width at startup") as exc_info, running_alone(database):
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
