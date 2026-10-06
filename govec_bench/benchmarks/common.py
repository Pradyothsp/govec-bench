import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.registry import DATABASES, Database, select_databases
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.registry import DATASETS, SIZES, DatasetSizes

SETUP_BATCH_SIZE = 100

DOCKER = shutil.which("docker")

COMPOSE_FILE = Path("docker-compose.yml")

POLL_INTERVAL_S = 0.05

# Generous: a fresh container, not a restart, so the image's first start is inside the wait.
START_TIMEOUT_S = 120

# Qdrant writes tens of GB of temporary files while it optimizes a 100k load of 1536-d vectors
# (about 30 GB seen). Below this a 100k run can fail partway, after its longest loads.
MIN_FREE_DISK_BYTES = 35 * 1000**3

# Docker Desktop's virtual disk is a sparse file that grows into the Mac's own free space.
DOCKER_DESKTOP_DISK_DIR = Path.home() / "Library/Containers/com.docker.docker/Data/vms/0/data"

# Any service will do for reading Docker's disk: it only runs df in a throwaway container.
DISK_PROBE_SERVICE = "govec"


class FreeSpace(NamedTuple):
    where: str
    free_bytes: int


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


def add_size_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", choices=SIZES, default="small", help="10k (small) or 100k (large) base vectors")


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


def parse_df_available(output: str) -> int:
    # `df -Pk` (POSIX layout): a header line, then one line per filesystem; field 4 is the
    # available space in 1 KiB blocks.
    return int(output.strip().splitlines()[-1].split()[3]) * 1024


def free_space() -> list[FreeSpace]:
    # Inside Docker's disk is where the volumes live, so that's what fills up; on Docker Desktop
    # it's a virtual disk with its own size limit, separate from the Mac's free space.
    output = docker("compose", "run", "--rm", "--no-deps", "--entrypoint", "df", DISK_PROBE_SERVICE, "-Pk", "/")
    spaces = [FreeSpace("Docker's disk", parse_df_available(output.stdout))]

    if DOCKER_DESKTOP_DISK_DIR.exists():
        spaces.append(
            FreeSpace("the Mac's disk, which Docker's grows into", shutil.disk_usage(DOCKER_DESKTOP_DISK_DIR).free)
        )

    return spaces


def shortfalls(spaces: Sequence[FreeSpace], min_bytes: int) -> list[FreeSpace]:
    return [space for space in spaces if space.free_bytes < min_bytes]


def require_free_disk(min_bytes: int = MIN_FREE_DISK_BYTES) -> None:
    # Checked before a 100k run starts, so a full disk stops it with a reason instead of failing
    # an hour in, partway through a database.
    short = shortfalls(free_space(), min_bytes)
    if short:
        details = "; ".join(f"{space.where} has {space.free_bytes / 1000**3:.1f} GB free" for space in short)
        msg = f"need at least {min_bytes / 1000**3:.0f} GB free for a 100k run: {details}"
        raise SystemExit(msg)


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


def render_override(service: str, environment: dict[str, str]) -> str:
    # JSON is valid YAML, so Compose reads this as an override file without a YAML dependency.
    return json.dumps({"services": {service: {"environment": environment}}}, indent=2)


@contextmanager
def running_alone(
    service: str, database: Database, overrides: Sequence[Path] = (), dimensions: int | None = None
) -> Iterator[VectorDBAdapter]:
    # Only the database under test runs. Idle neighbours still work in the background (Chroma
    # compacts, Qdrant optimizes, Go collects garbage) and hold their loaded data in the same
    # Docker VM; at 2 CPUs each, four containers claim every core of an 8-core laptop. Each
    # database starts from a fresh container and empty volumes, and leaves none behind.
    # overrides are Compose files layered over docker-compose.yml for this start only, so a
    # benchmark can change a service's settings without editing the published file.
    with tempfile.TemporaryDirectory() as tmp:
        layered = list(overrides)
        if database.needs_dimensions:
            if dimensions is None:
                msg = f"{service} must be told the vector width at startup: pass the dataset's dimensions"
                raise ValueError(msg)
            dims_override = Path(tmp) / "dimensions.override.yml"
            dims_override.write_text(render_override(service, {"GOVEC_DIMENSIONS": str(dimensions)}))
            layered.append(dims_override)

        files = [arg for path in (COMPOSE_FILE, *layered) for arg in ("--file", str(path))] if layered else []

        compose("down", "--volumes")
        compose(*files, "up", "--detach", service)
        try:
            wait_until_queryable(database.build_adapter, START_TIMEOUT_S)

            yield database.build_adapter()
        finally:
            compose("down", "--volumes")
