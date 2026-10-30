"""Searching a loaded index: each query timed and graded in the same call.

The one implementation of a search measurement, for the pipeline's query steps and the ef sweep.
Recall and latency come from the same searches, so they describe the same work. Graded against
exact cosine neighbours (datasets/groundtruth.py), because every database searches with cosine:
SIFT's shipped ground truth is Euclidean, and grading against it marked correct answers wrong on
near-ties.
"""

import time
from collections.abc import Sequence
from typing import NamedTuple

from tqdm import tqdm

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.datasets.base import ArrayItems
from govec_bench.results import LatencyStats, RecallStats, compute_latency_stats, compute_recall_stats
from govec_bench.types import NeighborIndices, Vector


class SearchStats(NamedTuple):
    recall: RecallStats
    latency: LatencyStats


def warm_up(adapter: VectorDBAdapter, queries: Sequence[Vector], k: int) -> None:
    # One untimed pass, so the first k measured doesn't also pay for cold caches.
    for query in tqdm(queries, desc="warm-up", leave=False):
        adapter.query(query, k=k)


def measure_search(
    adapter: VectorDBAdapter,
    base: ArrayItems,
    queries: Sequence[Vector],
    groundtruth: Sequence[NeighborIndices],
    k: int,
    *,
    passes: int = 1,
) -> SearchStats:
    # passes: timed passes over the queries; more samples where a pass alone is too few to
    # separate the databases (the ef sweep's).
    latencies_ms = []
    recalls = []
    total = len(queries) * passes
    searches = ((query, truth) for _ in range(passes) for query, truth in zip(queries, groundtruth, strict=True))
    for query, truth in tqdm(searches, total=total, desc=f"queries k={k}", leave=False):
        start = time.perf_counter()
        results = adapter.query(query, k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        expected = {base.id_at(i) for i in truth[:k]}
        recalls.append(len(expected & {result.id for result in results}) / k)

    return SearchStats(compute_recall_stats(recalls), compute_latency_stats(latencies_ms))
