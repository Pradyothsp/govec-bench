from collections.abc import Callable
from dataclasses import asdict
from functools import partial

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.qdrant_adapter import QdrantAdapter
from govec_bench.benchmarks.common import compose, wait_until_queryable
from govec_bench.results import LatencyStats, compute_latency_stats, write_results

ITERATIONS = 5
POLL_TIMEOUT_S = 60

# service name (docker-compose.yml) -> adapter constructor
SERVICES: dict[str, Callable[[], VectorDBAdapter]] = {
    "govec": GovecAdapter,
    "govec-scalar": partial(GovecAdapter, port=8002),
    "chroma": ChromaAdapter,
    "qdrant": QdrantAdapter,
}


def measure_cold_start(service: str, build_adapter: Callable[[], VectorDBAdapter]) -> LatencyStats:
    # Clear any state left over from other benchmarks before the first stop,
    # so every iteration -- including the first -- starts from an empty index
    # rather than whatever a prior insert/recall run happened to leave behind.
    build_adapter().reset()

    latencies_ms = []
    for _ in range(ITERATIONS):
        compose("stop", service)
        compose("start", service)
        latencies_ms.append(wait_until_queryable(build_adapter, POLL_TIMEOUT_S))

    return compute_latency_stats(latencies_ms)


def main() -> None:
    results: dict[str, object] = {}
    for name, build_adapter in SERVICES.items():
        print(f"Measuring cold start: {name}...")
        results[name] = asdict(measure_cold_start(name, build_adapter))

    path = write_results(benchmark="cold_start", dataset="n/a", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
