from collections.abc import Callable
from typing import NamedTuple

from govec_bench.datasets.base import VectorDataset
from govec_bench.datasets.dbpedia import load_dbpedia10k, load_dbpedia100k
from govec_bench.datasets.sift import load_sift10k, load_sift100k


class DatasetSizes(NamedTuple):
    small: Callable[[], VectorDataset]  # insert, query and recall: 10k vectors
    large: Callable[[], VectorDataset]  # memory and disk: 100k vectors


DATASETS: dict[str, DatasetSizes] = {
    # 128-dimensional image descriptors: the standard ANN benchmark set.
    "sift": DatasetSizes(small=load_sift10k, large=load_sift100k),
    # 1536-dimensional OpenAI text embeddings: what a RAG system actually stores.
    "dbpedia": DatasetSizes(small=load_dbpedia10k, large=load_dbpedia100k),
}
