import argparse
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.registry import DATABASES, Database, select_databases
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.registry import DATASETS, DatasetSizes

SETUP_BATCH_SIZE = 100

DOCKER = shutil.which("docker")

COMPOSE_FILE = Path("docker-compose.yml")

POLL_INTERVAL_S = 0.05

# Generous: a fresh container, not a restart, so the image's first start is inside the wait.
START_TIMEOUT_S = 120


class BenchArgs(NamedTuple):
    dataset: DatasetSizes
    databases: dict[str, Database]


def build_parser() -> argparse.ArgumentParser:
    # The arguments every benchmark shares; a benchmark with its own adds them to this parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="sift")
    parser.add_argument(
        "--db",
        action="append",
        choices=list(DATABASES),
        help="run only this database; repeat for several (default: all)",
    )
    return parser


def to_bench_args(args: argparse.Namespace) -> BenchArgs:
    return BenchArgs(dataset=DATASETS[args.dataset], databases=select_databases(args.db))


def parse_args(argv: Sequence[str] | None = None) -> BenchArgs:
    return to_bench_args(build_parser().parse_args(argv))


def load_dataset(adapter: VectorDBAdapter, items: ArrayItems, batch_size: int = SETUP_BATCH_SIZE) -> None:
    # Chunked rather than one giant batch_insert call -- some backends (e.g.
    # Chroma) enforce a max batch size well below typical dataset sizes.
    for i in range(0, len(items), batch_size):
        adapter.batch_insert(items[i : i + batch_size])

    adapter.wait_until_settled()


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


def container_of(service: str) -> str:
    # Asked of Compose rather than kept in the registry, so docker-compose.yml stays the only
    # place container names are set.
    return docker("compose", "ps", "--quiet", service).stdout.strip()


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


@contextmanager
def running_alone(service: str, database: Database, overrides: Sequence[Path] = ()) -> Iterator[VectorDBAdapter]:
    # Only the database under test runs. Idle neighbours still work in the background (Chroma
    # compacts, Qdrant optimizes, Go collects garbage) and hold their loaded data in the same
    # Docker VM; at 2 CPUs each, four containers claim every core of an 8-core laptop. Each
    # database starts from a fresh container and empty volumes, and leaves none behind.
    # overrides are Compose files layered over docker-compose.yml for this start only, so a
    # benchmark can change a service's settings without editing the published file.
    files = [arg for path in (COMPOSE_FILE, *overrides) for arg in ("--file", str(path))] if overrides else []

    compose("down", "--volumes")
    compose(*files, "up", "--detach", service)
    try:
        wait_until_queryable(database.build_adapter, START_TIMEOUT_S)

        yield database.build_adapter()
    finally:
        compose("down", "--volumes")
