from dataclasses import asdict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.benchmarks.common import load_dataset, parse_args, running_alone
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.results import RecallStats, compute_recall_stats, write_results
from govec_bench.types import NeighborIndices, Vector

K_VALUES = [1, 5, 10, 50]

# Graded against exact cosine neighbors, computed here, because every database
# searches with cosine. SIFT's shipped siftsmall_groundtruth.ivecs is Euclidean:
# grading cosine results against it marked correct answers wrong on near-ties
# and capped even a perfect search at 98.0% recall@1.


def measure_recall(
    adapter: VectorDBAdapter,
    base: ArrayItems,
    queries: list[Vector],
    groundtruth: list[NeighborIndices],
    k: int,
) -> RecallStats:
    recalls = []
    for query, truth in zip(queries, groundtruth, strict=True):
        expected = {base.id_at(i) for i in truth[:k]}
        results = adapter.query(query, k=k)
        got = {r.id for r in results}

        recalls.append(len(expected & got) / k)

    return compute_recall_stats(recalls)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.small()
    groundtruth = exact_cosine_neighbors(dataset.base.vectors, dataset.queries, max(K_VALUES))

    results: dict[str, object] = {}
    for name, database in args.databases.items():
        print(f"Measuring recall: {name}...")
        with running_alone(name, database) as adapter:
            load_dataset(adapter, dataset.base)

            results[name] = {
                f"k_{k}": asdict(measure_recall(adapter, dataset.base, dataset.queries, groundtruth, k))
                for k in K_VALUES
            }

    path = write_results(benchmark="recall", dataset=dataset.name, results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
