from collections.abc import Callable
from dataclasses import asdict

from tqdm import tqdm

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.benchmarks.common import compose, parse_args, running_alone, wait_until_queryable
from govec_bench.results import LatencyStats, compute_latency_stats, write_results

ITERATIONS = 5
POLL_TIMEOUT_S = 60


def measure_cold_start(service: str, build_adapter: Callable[[], VectorDBAdapter]) -> LatencyStats:
    # running_alone() hands over a fresh, empty container, so every iteration
    # restarts an empty index.
    latencies_ms = []
    for _ in tqdm(range(ITERATIONS), desc="restarts", leave=False):
        compose("stop", service)
        compose("start", service)
        latencies_ms.append(wait_until_queryable(build_adapter, POLL_TIMEOUT_S))

    return compute_latency_stats(latencies_ms)


def main() -> None:
    args = parse_args()

    results: dict[str, object] = {}
    for name, database in args.databases.items():
        print(f"Measuring cold start: {name}...")
        with running_alone(name, database, dimensions=args.dataset.dimensions):
            results[name] = asdict(measure_cold_start(name, database.build_adapter))

    path = write_results(benchmark="cold_start", dataset="n/a", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
