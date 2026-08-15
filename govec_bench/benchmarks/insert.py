import time
from dataclasses import asdict

from govec_bench.adapters.base import InsertItem, VectorDBAdapter
from govec_bench.adapters.registry import build_adapters
from govec_bench.datasets.synthetic import load_sift10k
from govec_bench.results import LatencyStats, compute_latency_stats, write_results

BATCH_SIZE = 100


def measure_single_insert(adapter: VectorDBAdapter, items: list[InsertItem]) -> LatencyStats:
    latencies_ms = []
    for item in items:
        start = time.perf_counter()
        adapter.insert(item.id, item.vector, item.metadata)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return compute_latency_stats(latencies_ms)


def measure_batch_insert(adapter: VectorDBAdapter, items: list[InsertItem], batch_size: int) -> LatencyStats:
    # Each sample is a batch's wall time divided by its size, so this is
    # directly comparable to the per-vector single-insert latency above.
    latencies_ms = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        start = time.perf_counter()
        adapter.batch_insert(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies_ms.append(elapsed_ms / len(batch))

    return compute_latency_stats(latencies_ms)


def main() -> None:
    dataset = load_sift10k()
    adapters = build_adapters()

    results: dict[str, object] = {}
    for name, adapter in adapters.items():
        adapter.reset()
        single_stats = measure_single_insert(adapter, dataset.base)

        adapter.reset()
        batch_stats = measure_batch_insert(adapter, dataset.base, BATCH_SIZE)

        results[name] = {
            "single": asdict(single_stats),
            "batch": asdict(batch_stats),
        }

    path = write_results(benchmark="insert_latency", dataset="sift10k", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
