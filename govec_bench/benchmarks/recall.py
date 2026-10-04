from dataclasses import asdict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.registry import build_adapters
from govec_bench.benchmarks.common import load_dataset
from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.datasets.sift import load_sift10k
from govec_bench.results import RecallStats, compute_recall_stats, write_results
from govec_bench.types import NeighborIndices, Vector

K_VALUES = [1, 5, 10, 50]

# load_sift10k() assigns base vector IDs as f"sift10k_{i}", where i is the
# vector's position in siftsmall_base.fvecs -- the positions the ground-truth
# neighbor indices refer to.
ID_PREFIX = "sift10k_"

# Graded against exact cosine neighbors, computed here, because every database
# searches with cosine. SIFT's shipped siftsmall_groundtruth.ivecs is Euclidean:
# grading cosine results against it marked correct answers wrong on near-ties
# and capped even a perfect search at 98.0% recall@1.


def measure_recall(
    adapter: VectorDBAdapter,
    queries: list[Vector],
    groundtruth: list[NeighborIndices],
    k: int,
) -> RecallStats:
    recalls = []
    for query, truth in zip(queries, groundtruth, strict=True):
        expected = {f"{ID_PREFIX}{i}" for i in truth[:k]}
        results = adapter.query(query, k=k)
        got = {r.id for r in results}

        recalls.append(len(expected & got) / k)

    return compute_recall_stats(recalls)


def main() -> None:
    dataset = load_sift10k()
    groundtruth = exact_cosine_neighbors([item.vector for item in dataset.base], dataset.queries, max(K_VALUES))

    adapters = build_adapters()
    # Not in build_adapters() -- that dict is shared with insert.py/query.py,
    # which have no use for a second govec variant. Scoped to this benchmark
    # only, to measure the accuracy cost of int8 scalar quantization
    # (govec-config-scalar.yaml, the govec-scalar service on port 9699)
    # against the RAM/disk win already measured in memory.py.
    adapters["govec-scalar"] = GovecAdapter(port=9699)

    results: dict[str, object] = {}
    for name, adapter in adapters.items():
        adapter.reset()
        load_dataset(adapter, dataset.base)

        results[name] = {f"k_{k}": asdict(measure_recall(adapter, dataset.queries, groundtruth, k)) for k in K_VALUES}

    path = write_results(benchmark="recall", dataset="sift10k", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
