from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from govec_bench.datasets.base import ArrayItems, VectorDataset

RAW_DIR = Path("data/raw/dbpedia")

# The first three of the 26 shards of KShivendu/dbpedia-entities-openai-1M on Hugging Face:
# DBpedia entity abstracts embedded with OpenAI's text-embedding-ada-002 (1536 dimensions,
# unit length). 115,386 rows, enough for the 100k base set plus held-out queries.
SHARDS = (
    "train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    "train-00001-of-00026-2b24035a6390fdcb.parquet",
    "train-00002-of-00026-b05ce48965853dad.parquet",
)

EMBEDDING_COLUMN = "openai"

QUERY_COUNT = 100


class BaseAndQueries(NamedTuple):
    base: npt.NDArray[np.float32]
    queries: npt.NDArray[np.float32]


def read_embeddings(paths: list[Path]) -> npt.NDArray[np.float32]:
    # Flattening the list column straight into numpy skips building a Python float per value,
    # which for 115k x 1536 values would take minutes and gigabytes.
    shards = []
    for path in paths:
        column = pq.read_table(path, columns=[EMBEDDING_COLUMN]).column(EMBEDDING_COLUMN).combine_chunks()
        values = column.flatten().to_numpy().astype(np.float32)

        shards.append(values.reshape(len(column), -1))

    return np.concatenate(shards)


def split_base_and_queries(embeddings: npt.NDArray[np.float32], base_count: int, query_count: int) -> BaseAndQueries:
    # Queries come from the tail and the base from the head, so no query is ever in the base
    # set, for either dataset size. The source has no separate query set, unlike SIFT.
    if base_count + query_count > len(embeddings):
        msg = f"need {base_count + query_count} rows, have {len(embeddings)}"
        raise ValueError(msg)

    return BaseAndQueries(base=embeddings[:base_count], queries=embeddings[-query_count:])


def _load(name: str, base_count: int) -> VectorDataset:
    embeddings = read_embeddings([RAW_DIR / shard for shard in SHARDS])
    base, queries = split_base_and_queries(embeddings, base_count, QUERY_COUNT)

    return VectorDataset(name=name, base=ArrayItems(name, base), queries=queries.tolist())


def load_dbpedia10k() -> VectorDataset:
    return _load("dbpedia10k", 10_000)


def load_dbpedia100k() -> VectorDataset:
    return _load("dbpedia100k", 100_000)
