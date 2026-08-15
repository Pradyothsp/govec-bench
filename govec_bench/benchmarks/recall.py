from dataclasses import asdict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.registry import build_adapters
from govec_bench.benchmarks.common import load_dataset
from govec_bench.datasets.synthetic import load_sift10k
from govec_bench.results import RecallStats, compute_recall_stats, write_results
from govec_bench.types import NeighborIndices, Vector

K_VALUES = [1, 5, 10, 50]

# load_sift10k() assigns base vector IDs as f"sift10k_{i}", where i is the
# vector's position in siftsmall_base.fvecs -- the same positions the
# groundtruth file's neighbor indices refer to.
ID_PREFIX = "sift10k_"

# siftsmall_groundtruth.ivecs was computed with Euclidean distance. Under
# cosine distance (govec's/Chroma's default here), even brute-force (exact)
# search recalls just under 100% due to the metric mismatch -- not a bug. See
# docs/architecture/DISTANCE_METRICS.md in the govec repo.


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
    if dataset.groundtruth is None:
        msg = "sift10k dataset must include groundtruth for the recall benchmark"
        raise ValueError(msg)

    adapters = build_adapters()

    results: dict[str, object] = {}
    for name, adapter in adapters.items():
        adapter.reset()
        load_dataset(adapter, dataset.base)

        results[name] = {
            f"k_{k}": asdict(measure_recall(adapter, dataset.queries, dataset.groundtruth, k)) for k in K_VALUES
        }

    path = write_results(benchmark="recall", dataset="sift10k", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
