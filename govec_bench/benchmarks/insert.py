import time
from dataclasses import asdict

from tqdm import tqdm

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.benchmarks.common import parse_args, running_alone
from govec_bench.datasets.base import ArrayItems
from govec_bench.results import LatencyStats, compute_latency_stats, write_results

BATCH_SIZE = 100


def measure_single_insert(adapter: VectorDBAdapter, items: ArrayItems) -> LatencyStats:
    latencies_ms = []
    # The bar updates outside the timed call, so it costs nothing in the measurement.
    for item in tqdm(items, desc="single inserts", unit="vec", leave=False):
        start = time.perf_counter()
        adapter.insert(item.id, item.vector, item.metadata)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return compute_latency_stats(latencies_ms)


def measure_batch_insert(adapter: VectorDBAdapter, items: ArrayItems, batch_size: int) -> LatencyStats:
    # Each sample is a batch's wall time divided by its size, so this is
    # directly comparable to the per-vector single-insert latency above.
    latencies_ms = []
    for i in tqdm(range(0, len(items), batch_size), desc="batch inserts", unit="batch", leave=False):
        batch = items[i : i + batch_size]

        start = time.perf_counter()
        adapter.batch_insert(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies_ms.append(elapsed_ms / len(batch))

    return compute_latency_stats(latencies_ms)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.small()

    results: dict[str, object] = {}
    for name, database in args.databases.items():
        print(f"Measuring insert latency: {name}...")
        with running_alone(name, database, dimensions=args.dataset.dimensions) as adapter:
            single_stats = measure_single_insert(adapter, dataset.base)

            adapter.reset()
            batch_stats = measure_batch_insert(adapter, dataset.base, BATCH_SIZE)

        results[name] = {
            "single": asdict(single_stats),
            "batch": asdict(batch_stats),
        }

    path = write_results(benchmark="insert_latency", dataset=dataset.name, results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
