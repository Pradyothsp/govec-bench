import time
from dataclasses import asdict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.datasets.synthetic import load_sift10k
from govec_bench.results import LatencyStats, compute_latency_stats, write_results
from govec_bench.types import Vector

K_VALUES = [1, 5, 10, 50]


def measure_query_latency(adapter: VectorDBAdapter, queries: list[Vector], k: int) -> LatencyStats:
    latencies_ms = []
    for query in queries:
        start = time.perf_counter()
        adapter.query(query, k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return compute_latency_stats(latencies_ms)


def main() -> None:
    dataset = load_sift10k()
    adapter = GovecAdapter()

    adapter.reset()
    adapter.batch_insert(dataset.base)

    results_by_k = {f"k_{k}": asdict(measure_query_latency(adapter, dataset.queries, k)) for k in K_VALUES}

    path = write_results(benchmark="query_latency", dataset="sift10k", results={"govec": results_by_k})
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
