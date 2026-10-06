from govec_bench.adapters.registry import DATABASES, select_databases


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


def test_select_databases__names__returns_only_those_in_registry_order() -> None:
    # Arrange
    names = ["qdrant", "govec"]

    # Act
    selected = select_databases(names)

    # Assert
    assert list(selected) == ["govec", "qdrant"]
    assert selected["qdrant"] == DATABASES["qdrant"]
