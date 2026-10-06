import numpy as np

from govec_bench.datasets.base import ArrayItems, VectorDataset
from govec_bench.datasets.registry import DATASETS, SIZES, DatasetSizes, load_sized


def _dataset(name: str) -> VectorDataset:
    vectors = np.zeros((1, 2), dtype=np.float32)
    return VectorDataset(name=name, base=ArrayItems("v", vectors), queries=[[0.0, 0.0]])


def test_sizes__each_names_a_dataset_loader() -> None:
    # Arrange
    sizes = DATASETS["sift"]

    # Act
    loaders = [getattr(sizes, size) for size in SIZES]

    # Assert
    assert all(callable(loader) for loader in loaders)


def test_load_sized__each_size__calls_its_own_loader() -> None:
    # Arrange
    sizes = DatasetSizes(small=lambda: _dataset("small"), large=lambda: _dataset("large"), dimensions=2)

    # Act
    small = load_sized(sizes, "small")
    large = load_sized(sizes, "large")

    # Assert
    assert small.name == "small"
    assert large.name == "large"
