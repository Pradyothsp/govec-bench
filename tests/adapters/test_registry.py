import pytest

from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.registry import DATABASES, ENGINES, parse_databases, select_databases


def test_select_databases__no_names__returns_every_database_except_opt_in() -> None:
    # Arrange
    standard = {name: database for name, database in DATABASES.items() if not database.opt_in}

    # Act
    selected = select_databases()

    # Assert
    assert selected == standard
    assert "govec-mmap" not in selected


def test_select_databases__opt_in_named__returns_it() -> None:
    # Arrange
    names = ["govec", "govec-mmap"]

    # Act
    selected = select_databases(names)

    # Assert
    assert list(selected) == ["govec", "govec-mmap"]


def test_select_databases__names__returns_only_those_in_file_order() -> None:
    # Arrange
    names = ["qdrant", "govec"]

    # Act
    selected = select_databases(names)

    # Assert
    assert list(selected) == ["govec", "qdrant"]
    assert selected["qdrant"] == DATABASES["qdrant"]


def test_databases__every_database__runs_on_a_known_engine() -> None:
    # Arrange

    # Act
    services = {database.service for database in DATABASES.values()}

    # Assert
    assert services <= set(ENGINES)


def test_databases__grpc_client__every_database_on_an_engine_with_a_grpc_api_has_one() -> None:
    # Arrange

    # Act
    with_grpc = [name for name, database in DATABASES.items() if database.build_grpc_adapter is not None]

    # Assert
    assert with_grpc == ["govec", "govec-scalar", "govec-mmap", "govec-brute", "qdrant"]


def test_parse_databases__entry__takes_its_engine_and_settings() -> None:
    # Arrange
    text = "govec-x:\n  engine: govec\n  env:\n    GOVEC_ENABLE_MMAP: true\n    GOVEC_HNSW_M: 32\n  opt_in: true\n"

    # Act
    database = parse_databases(text)["govec-x"]

    # Assert
    assert database.name == "govec-x"
    assert database.service == "govec"
    assert database.build_adapter is GovecAdapter
    assert database.env == {"GOVEC_ENABLE_MMAP": "true", "GOVEC_HNSW_M": "32"}
    assert database.opt_in
    assert not database.needs_dimensions


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("x:\n  engine: milvus\n", "engine 'milvus'"),
        ("x:\n  opt_in: true\n", "engine None"),
        ("x:\n  engine: govec\n  optin: true\n", "unknown keys \\['optin'\\]"),
    ],
)
def test_parse_databases__bad_entry__raises_naming_it(text: str, message: str) -> None:
    # Arrange

    # Act
    with pytest.raises(ValueError, match=message) as exc_info:
        parse_databases(text)

    # Assert
    assert str(exc_info.value).startswith("x: ")


def test_parse_databases__empty_file__no_databases() -> None:
    # Arrange
    text = "# nothing yet\n"

    # Act
    databases = parse_databases(text)

    # Assert
    assert databases == {}


@pytest.mark.parametrize("text", ["- govec\n", "govec: govec\n"])
def test_parse_databases__not_a_mapping__raises(text: str) -> None:
    # Arrange

    # Act
    with pytest.raises(TypeError) as exc_info:
        parse_databases(text)

    # Assert
    assert "expected" in str(exc_info.value)
