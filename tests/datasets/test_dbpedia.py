from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from govec_bench.datasets.dbpedia import EMBEDDING_COLUMN, read_embeddings, split_base_and_queries


@pytest.fixture
def write_shard(tmp_path: Path):
    def _write(name: str, rows: list[list[float]]) -> Path:
        path = tmp_path / name
        pq.write_table(pa.table({"_id": [str(i) for i in range(len(rows))], EMBEDDING_COLUMN: rows}), path)
        return path

    return _write


def test_read_embeddings__several_shards__concatenates_rows_in_shard_order(write_shard) -> None:
    # Arrange
    first = write_shard("a.parquet", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    second = write_shard("b.parquet", [[0.7, 0.8, 0.9]])

    # Act
    embeddings = read_embeddings([first, second])

    # Assert
    expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(embeddings, expected)


def test_split_base_and_queries__enough_rows__takes_base_from_head_and_queries_from_tail() -> None:
    # Arrange
    embeddings = np.arange(10, dtype=np.float32).reshape(5, 2)

    # Act
    split = split_base_and_queries(embeddings, base_count=2, query_count=2)

    # Assert
    np.testing.assert_array_equal(split.base, embeddings[:2])
    np.testing.assert_array_equal(split.queries, embeddings[3:])


def test_split_base_and_queries__base_and_queries_would_overlap__raises_value_error() -> None:
    # Arrange
    embeddings = np.zeros((5, 2), dtype=np.float32)

    # Act / Assert
    with pytest.raises(ValueError, match="need 6 rows"):
        split_base_and_queries(embeddings, base_count=4, query_count=2)
