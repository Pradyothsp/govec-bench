import shutil
import subprocess
import time
from collections.abc import Callable

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.datasets.base import ArrayItems

SETUP_BATCH_SIZE = 100

DOCKER = shutil.which("docker")

POLL_INTERVAL_S = 0.05


def load_dataset(adapter: VectorDBAdapter, items: ArrayItems, batch_size: int = SETUP_BATCH_SIZE) -> None:
    # Chunked rather than one giant batch_insert call -- some backends (e.g.
    # Chroma) enforce a max batch size well below typical dataset sizes.
    for i in range(0, len(items), batch_size):
        adapter.batch_insert(items[i : i + batch_size])


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    if DOCKER is None:
        msg = "docker is required for this benchmark but was not found on PATH"
        raise RuntimeError(msg)
    return subprocess.run(  # noqa: S603 -- fixed args, callers pass hardcoded container/service names
        [DOCKER, *args],
        check=True,
        capture_output=True,
        text=True,
    )


def compose(*args: str) -> None:
    docker("compose", *args)


def wait_until_queryable(build_adapter: Callable[[], VectorDBAdapter], timeout_s: float) -> float:
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
