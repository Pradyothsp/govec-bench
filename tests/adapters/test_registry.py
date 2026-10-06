from govec_bench.adapters.registry import DATABASES, select_databases


def test_select_databases__no_names__returns_every_database() -> None:
    # Arrange

    # Act
    selected = select_databases()

    # Assert
    assert selected == DATABASES


def test_select_databases__names__returns_only_those_in_registry_order() -> None:
    # Arrange
    names = ["qdrant", "govec"]

    # Act
    selected = select_databases(names)

    # Assert
    assert list(selected) == ["govec", "qdrant"]
    assert selected["qdrant"] == DATABASES["qdrant"]
