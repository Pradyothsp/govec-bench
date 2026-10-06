from collections.abc import Sequence
from dataclasses import asdict
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.benchmarks.common import (
    BenchArgs,
    add_size_argument,
    build_parser,
    load_dataset,
    require_free_disk,
    running_alone,
    to_bench_args,
)
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.datasets.registry import Size, load_sized
from govec_bench.results import RecallStats, compute_recall_stats, write_results
from govec_bench.types import NeighborIndices, Vector

K_VALUES = [1, 5, 10, 50]

# Graded against exact cosine neighbors, computed here, because every database
# searches with cosine. SIFT's shipped siftsmall_groundtruth.ivecs is Euclidean:
# grading cosine results against it marked correct answers wrong on near-ties
# and capped even a perfect search at 98.0% recall@1.


class RecallArgs(NamedTuple):
    bench: BenchArgs
    size: Size  # large: the memory benchmark's 100k set, to check recall holds as the graph grows


def parse_recall_args(argv: Sequence[str] | None = None) -> RecallArgs:
    parser = build_parser()
    add_size_argument(parser)
    args = parser.parse_args(argv)

    return RecallArgs(bench=to_bench_args(args), size=args.size)


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
    args = parse_recall_args()
    if args.size == "large":
        require_free_disk()
    dataset = load_sized(args.bench.dataset, args.size)
    groundtruth = exact_cosine_neighbors(dataset.base.vectors, dataset.queries, max(K_VALUES))

    results: dict[str, object] = {}
    for name, database in args.bench.databases.items():
        print(f"Measuring recall: {name}...")
        with running_alone(name, database, dimensions=args.bench.dataset.dimensions) as adapter:
            load_dataset(adapter, dataset.base)

            results[name] = {
                f"k_{k}": asdict(measure_recall(adapter, dataset.base, dataset.queries, groundtruth, k))
                for k in K_VALUES
            }

    path = write_results(benchmark="recall", dataset=dataset.name, results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
