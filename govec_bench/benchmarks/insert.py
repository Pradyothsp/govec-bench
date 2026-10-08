import time
from dataclasses import asdict, dataclass

from tqdm import tqdm

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.benchmarks.common import parse_args, running_alone
from govec_bench.datasets.base import ArrayItems
from govec_bench.results import LatencyStats, compute_latency_stats, write_results

BATCH_SIZE = 100


@dataclass(frozen=True, slots=True)
class InsertStats:
    # The whole load, from the first insert until every vector is indexed: what it costs to make
    # the data searchable, however a database splits that between its inserts and the background.
    total_s: float
    per_vector_ms: float
    # Each call timed on its own, per vector (a batch's time divided by its size). Leaves out
    # indexing a database does after the call returns (Qdrant's); the two above don't.
    calls: LatencyStats


def build_insert_stats(total_s: float, vector_count: int, call_latencies_ms: list[float]) -> InsertStats:
    return InsertStats(
        total_s=total_s,
        per_vector_ms=total_s * 1000 / vector_count,
        calls=compute_latency_stats(call_latencies_ms),
    )


def measure_single_insert(adapter: VectorDBAdapter, items: ArrayItems) -> InsertStats:
    latencies_ms = []
    load_start = time.perf_counter()
    # The bar updates outside each timed call; inside the total it costs microseconds per vector.
    for item in tqdm(items, desc="single inserts", unit="vec", leave=False):
        start = time.perf_counter()
        adapter.insert(item.id, item.vector, item.metadata)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    adapter.wait_until_indexed()
    total_s = time.perf_counter() - load_start

    return build_insert_stats(total_s, len(items), latencies_ms)


def measure_batch_insert(adapter: VectorDBAdapter, items: ArrayItems, batch_size: int) -> InsertStats:
    # Each call sample is a batch's wall time divided by its size, so it compares directly with
    # the per-vector single-insert latency above.
    latencies_ms = []
    load_start = time.perf_counter()
    for i in tqdm(range(0, len(items), batch_size), desc="batch inserts", unit="batch", leave=False):
        batch = items[i : i + batch_size]

        start = time.perf_counter()
        adapter.batch_insert(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies_ms.append(elapsed_ms / len(batch))

    adapter.wait_until_indexed()
    total_s = time.perf_counter() - load_start

    return build_insert_stats(total_s, len(items), latencies_ms)


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
