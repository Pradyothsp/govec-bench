import struct
from pathlib import Path

import pytest

from govec_bench.datasets.synthetic import read_fvecs, read_ivecs


@pytest.fixture
def write_fvecs(tmp_path: Path):
    def _write(name: str, rows: list[list[float]]) -> Path:
        path = tmp_path / name
        with path.open("wb") as f:
            for row in rows:
                f.write(struct.pack("<i", len(row)))
                f.write(struct.pack(f"<{len(row)}f", *row))
        return path

    return _write


@pytest.fixture
def write_ivecs(tmp_path: Path):
    def _write(name: str, rows: list[list[int]]) -> Path:
        path = tmp_path / name
        with path.open("wb") as f:
            for row in rows:
                f.write(struct.pack("<i", len(row)))
                f.write(struct.pack(f"<{len(row)}i", *row))
        return path

    return _write


def test_read_fvecs__valid_file__returns_parsed_rows(write_fvecs) -> None:
    # Arrange
    rows = [[0.0, 16.0, 35.0, 5.0], [1.5, -2.5, 3.5, 4.5], [100.0, 200.0, 300.0, 400.0]]
    path = write_fvecs("test.fvecs", rows)

    # Act
    result = read_fvecs(path)

    # Assert
    assert result == rows


def test_read_fvecs__limit_given__returns_only_limited_rows(write_fvecs) -> None:
    # Arrange
    rows = [[float(i)] * 4 for i in range(10)]
    path = write_fvecs("test.fvecs", rows)

    # Act
    result = read_fvecs(path, limit=3)

    # Assert
    assert result == rows[:3]


def test_read_ivecs__valid_file__returns_parsed_rows(write_ivecs) -> None:
    # Arrange
    rows = [[2176, 3752, 882], [10, 20, 30]]
    path = write_ivecs("test.ivecs", rows)

    # Act
    result = read_ivecs(path)

    # Assert
    assert result == rows


def test_read_fvecs__empty_file__returns_empty_list(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "empty.fvecs"
    path.write_bytes(b"")

    # Act
    result = read_fvecs(path)

    # Assert
    assert result == []
