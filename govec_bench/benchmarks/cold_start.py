import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.results import LatencyStats, compute_latency_stats, write_results

DOCKER = shutil.which("docker")

ITERATIONS = 5
POLL_INTERVAL_S = 0.05
POLL_TIMEOUT_S = 60

# service name (docker-compose.yml) -> adapter constructor
SERVICES: dict[str, Callable[[], VectorDBAdapter]] = {
    "govec": GovecAdapter,
    "chroma": ChromaAdapter,
}


def _compose(*args: str) -> None:
    if DOCKER is None:
        msg = "docker is required for the cold start benchmark but was not found on PATH"
        raise RuntimeError(msg)
    subprocess.run(  # noqa: S603 -- fixed args, service names come from the hardcoded SERVICES dict above
        [DOCKER, "compose", *args],
        check=True,
        capture_output=True,
    )


def _wait_until_queryable(build_adapter: Callable[[], VectorDBAdapter], timeout_s: float) -> float:
    # "Queryable" == the adapter's own reset() succeeds -- the same admin call
    # every other benchmark already uses to get a clean starting state, so it
    # doubles as a real, adapter-agnostic readiness probe (constructing the
    # adapter itself makes the first network round-trip for Chroma).
    start = time.perf_counter()
    deadline = start + timeout_s
    while time.perf_counter() < deadline:
        try:
            build_adapter().reset()
            return (time.perf_counter() - start) * 1000
        except Exception:  # noqa: BLE001 -- expected while the service is still coming up; keep polling
            time.sleep(POLL_INTERVAL_S)

    msg = f"service did not become queryable within {timeout_s}s"
    raise TimeoutError(msg)


def measure_cold_start(service: str, build_adapter: Callable[[], VectorDBAdapter]) -> LatencyStats:
    # Clear any state left over from other benchmarks before the first stop,
    # so every iteration -- including the first -- starts from an empty index
    # rather than whatever a prior insert/recall run happened to leave behind.
    build_adapter().reset()

    latencies_ms = []
    for _ in range(ITERATIONS):
        _compose("stop", service)
        _compose("start", service)
        latencies_ms.append(_wait_until_queryable(build_adapter, POLL_TIMEOUT_S))

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
