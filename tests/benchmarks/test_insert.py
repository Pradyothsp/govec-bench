import time
from typing import override

import numpy as np
import pytest

from govec_bench.adapters.base import InsertItem, QueryResult, Stats, VectorDBAdapter
from govec_bench.benchmarks.insert import build_insert_stats, measure_batch_insert, measure_single_insert
from govec_bench.datasets.base import ArrayItems
from govec_bench.types import Vector

INDEXING_S = 0.2


class BackgroundIndexingAdapter(VectorDBAdapter):
    # Stands in for a database whose inserts return before indexing does (Qdrant's).

    def __init__(self) -> None:
        self.indexed_waits = 0

    @override
    def insert(self, vector_id: str, vector: Vector, metadata: dict[str, str] | None = None) -> None:
        return

    @override
    def batch_insert(self, items: list[InsertItem]) -> None:
        return

    @override
    def wait_until_indexed(self) -> None:
        self.indexed_waits += 1
        time.sleep(INDEXING_S)

    @override
    def query(self, vector: Vector, k: int = 10) -> list[QueryResult]:
        return []

    @override
    def stats(self) -> Stats:
        return Stats(count=0)

    @override
    def reset(self) -> None:
        return


@pytest.fixture
def items() -> ArrayItems:
    return ArrayItems("v", np.zeros((1000, 4), dtype=np.float32))


def test_build_insert_stats__whole_load__per_vector_is_total_over_count() -> None:
    # Arrange
    total_s = 2.0
    vector_count = 1000

    # Act
    stats = build_insert_stats(total_s, vector_count, [1.0, 2.0, 3.0])

    # Assert
    assert stats.per_vector_ms == 2.0
    assert stats.calls.mean_ms == 2.0


def test_measure_batch_insert__background_indexing__counted_once_in_the_total_not_in_calls(
    items: ArrayItems,
) -> None:
    # Arrange
    adapter = BackgroundIndexingAdapter()

    # Act
    stats = measure_batch_insert(adapter, items, batch_size=100)

    # Assert
    assert adapter.indexed_waits == 1
    assert stats.total_s >= INDEXING_S
    assert stats.calls.p99_ms < INDEXING_S * 1000


def test_measure_single_insert__background_indexing__counted_once_in_the_total_not_in_calls(
    items: ArrayItems,
) -> None:
    # Arrange
    adapter = BackgroundIndexingAdapter()

    # Act
    stats = measure_single_insert(adapter, items)

    # Assert
    assert adapter.indexed_waits == 1
    assert stats.total_s >= INDEXING_S
    assert stats.calls.p99_ms < INDEXING_S * 1000
