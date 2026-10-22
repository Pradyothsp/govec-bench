from pathlib import Path

from govec_bench.datasets.sift import read_fvecs, read_ivecs
from scripts.export_dataset import write_fvecs, write_ivecs


def test_write_fvecs__rows__read_back_unchanged(tmp_path: Path) -> None:
    # Arrange
    rows = [[0.0, 16.0, 35.0], [1.5, -2.5, 3.5]]
    path = tmp_path / "rows.fvecs"

    # Act
    write_fvecs(path, rows)

    # Assert
    assert read_fvecs(path) == rows


def test_write_ivecs__rows__read_back_unchanged(tmp_path: Path) -> None:
    # Arrange
    rows = [[3, 1, 4, 1], [5, 9, 2, 6]]
    path = tmp_path / "rows.ivecs"

    # Act
    write_ivecs(path, rows)

    # Assert
    assert read_ivecs(path) == rows
