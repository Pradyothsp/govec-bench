import time
from dataclasses import asdict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.registry import build_adapters
from govec_bench.benchmarks.common import load_dataset
from govec_bench.datasets.registry import dataset_from_args
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
    dataset = dataset_from_args().small()
    adapters = build_adapters()
    # Not in build_adapters() -- see insert.py/recall.py's identical comment:
    # a second govec variant to measure int8 scalar quantization's query cost
    # (govec-config-scalar.yaml, the govec-scalar service on port 9699).
    adapters["govec-scalar"] = GovecAdapter(port=9699)

    results: dict[str, object] = {}
    for name, adapter in adapters.items():
        adapter.reset()
        load_dataset(adapter, dataset.base)

        results[name] = {f"k_{k}": asdict(measure_query_latency(adapter, dataset.queries, k)) for k in K_VALUES}

    path = write_results(benchmark="query_latency", dataset=dataset.name, results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
