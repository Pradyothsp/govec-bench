import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, TypedDict

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.registry import DATABASES, Database, select_databases
from govec_bench.benchmarks.images import GovecImage, resolve_govec_image
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.registry import DATASETS, SIZES, DatasetSizes
from govec_bench.results_file import StartedAs

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


# The GoVec image or settings a run asked for, on top of databases.yaml and the pinned image.
GOVEC_SERVICE = "govec"


def _setting(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key or not value:
        msg = f"expected KEY=VALUE, got {text!r}"
        raise argparse.ArgumentTypeError(msg)

    return f"GOVEC_{key.upper()}", value


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
    parser.add_argument(
        "--govec",
        metavar="REF",
        help=(
            "run GoVec from this instead of the pinned image: an image (anything with a colon, "
            "e.g. govec:local) or a git ref of the GoVec repo (HEAD, v0.2.1, a branch), built from "
            "a clean copy of that commit"
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        type=_setting,
        dest="settings",
        metavar="KEY=VALUE",
        help=(
            "change a GoVec setting for this run: a key of govec-config.yaml as its environment "
            "variable without GOVEC_, e.g. hnsw_ef_construction=200; repeat for several"
        ),
    )
    return parser


def add_size_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", choices=SIZES, default="small", help="10k (small) or 100k (large) base vectors")


def with_govec(databases: dict[str, Database], image_ref: str | None, settings: dict[str, str]) -> dict[str, Database]:
    # Applied to every GoVec database in the run; the others have no GoVec image or settings.
    if not image_ref and not settings:
        return databases

    if not any(database.service == GOVEC_SERVICE for database in databases.values()):
        msg = "--govec and --set change GoVec, but no GoVec database is selected"
        raise SystemExit(msg)

    return {
        name: database._replace(env={**database.env, **settings}, image_ref=image_ref)
        if database.service == GOVEC_SERVICE
        else database
        for name, database in databases.items()
    }


def to_bench_args(args: argparse.Namespace) -> BenchArgs:
    databases = with_govec(select_databases(args.db), args.govec, dict(args.settings or []))

    return BenchArgs(dataset=DATASETS[args.dataset], databases=databases)


def parse_args(argv: Sequence[str] | None = None) -> BenchArgs:
    return to_bench_args(build_parser().parse_args(argv))


def load_dataset(adapter: VectorDBAdapter, items: ArrayItems, batch_size: int = SETUP_BATCH_SIZE) -> None:
    # Chunked rather than one giant batch_insert call -- some backends (e.g.
    # Chroma) enforce a max batch size well below typical dataset sizes.
    for i in range(0, len(items), batch_size):
        adapter.batch_insert(items[i : i + batch_size])

    adapter.wait_until_indexed()
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


def wait_until_queryable(build_adapter: Callable[[], VectorDBAdapter], timeout_s: float) -> None:
    # "Queryable" == the adapter's own reset() succeeds -- the same admin call
    # every other benchmark already uses to get a clean starting state, so it
    # doubles as a real, adapter-agnostic readiness probe (constructing the
    # adapter itself makes the first network round-trip for Chroma).
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            build_adapter().reset()
        except Exception:  # noqa: BLE001 -- expected while the service is still coming up; keep polling
            time.sleep(POLL_INTERVAL_S)
        else:
            return

    msg = f"service did not become queryable within {timeout_s}s"
    raise TimeoutError(msg)


class Limits(TypedDict):
    memory: str  # Compose takes sizes as strings only; a bare number is bytes


class Resources(TypedDict):
    limits: Limits


class Deploy(TypedDict):
    resources: Resources


class ServiceOverride(TypedDict, total=False):
    # What a Compose override file changes on one service.
    environment: dict[str, str]
    image: str
    deploy: Deploy
    memswap_limit: str


def override_of(database: Database, environment: dict[str, str], govec_image: GovecImage | None) -> ServiceOverride:
    settings: ServiceOverride = {}
    if environment:
        settings["environment"] = environment
    if govec_image is not None:
        settings["image"] = govec_image.image
    if database.memory_bytes is not None:
        # Swap capped at the limit too, or Docker allows as much swap again and the cap moves
        # pages to swap instead of making them be read back.
        settings["deploy"] = {"resources": {"limits": {"memory": str(database.memory_bytes)}}}
        settings["memswap_limit"] = str(database.memory_bytes)
    return settings


def render_override(service: str, settings: ServiceOverride) -> str:
    # JSON is valid YAML, so Compose reads this as an override file without a YAML dependency.
    return json.dumps({"services": {service: settings}}, indent=2)


# What each database ran as, filled in by running_alone() from the started container, for the
# results file: the image docker-compose.yml pins is only what was asked for.
STARTED: dict[str, StartedAs] = {}


def _check_started(database: Database, environment: dict[str, str], govec_image: GovecImage | None) -> None:
    # Proof the settings reached the container: a typo would otherwise start it on its config
    # file's values, and every result would measure the setup it was meant to change.
    (inspected,) = json.loads(docker("inspect", container_of(database.service)).stdout)
    env = set(inspected["Config"]["Env"] or [])

    missing = [f"{key}={value}" for key, value in environment.items() if f"{key}={value}" not in env]
    if missing:
        msg = f"{database.name} started without {missing}"
        raise RuntimeError(msg)

    image = inspected["Config"]["Image"]
    if govec_image is not None and image != govec_image.image:
        msg = f"{database.name} started on {image}, not {govec_image.image}"
        raise RuntimeError(msg)

    memory_limit = inspected["HostConfig"]["Memory"]
    if database.memory_bytes is not None and memory_limit != database.memory_bytes:
        msg = f"{database.name} started with a {memory_limit}-byte memory limit, not {database.memory_bytes}"
        raise RuntimeError(msg)

    # The database's own settings, not what one start varies (the ef sweep's ef): a benchmark
    # that varies a setting records it with each measurement.
    STARTED[database.name] = {
        "service": database.service,
        "image": image,
        "image_id": inspected["Image"],
        "govec_commit": govec_image.commit if govec_image else None,
        "env": database.env,
        "memory_limit_bytes": memory_limit,
    }


def setup_of(databases: dict[str, Database]) -> dict[str, StartedAs]:
    # For the results file: what each database that started ran as.
    return {name: STARTED[name] for name in databases if name in STARTED}


@contextmanager
def running_alone(
    database: Database, env: dict[str, str] | None = None, dimensions: int | None = None
) -> Iterator[VectorDBAdapter]:
    # Only the database under test runs. Idle neighbours still work in the background (Chroma
    # compacts, Qdrant optimizes, Go collects garbage) and hold their loaded data in the same
    # Docker VM; at 2 CPUs each, four containers claim every core of an 8-core laptop. Each
    # database starts from a fresh container and empty volumes, and leaves none behind.
    # Its settings, plus env for this start only, go in a Compose file layered over
    # docker-compose.yml, so no setup needs a service or config file of its own.
    environment = {**database.env, **(env or {})}
    if database.needs_dimensions:
        if dimensions is None:
            msg = f"{database.name} must be told the vector width at startup: pass the dataset's dimensions"
            raise ValueError(msg)
        environment["GOVEC_DIMENSIONS"] = str(dimensions)

    govec_image = resolve_govec_image(database.image_ref) if database.image_ref else None

    settings = override_of(database, environment, govec_image)

    with tempfile.TemporaryDirectory() as tmp:
        files = []
        if settings:
            override = Path(tmp) / "run.override.yml"
            override.write_text(render_override(database.service, settings))
            files = ["--file", str(COMPOSE_FILE), "--file", str(override)]

        compose("down", "--volumes")
        compose(*files, "up", "--detach", database.service)
        try:
            _check_started(database, environment, govec_image)
            wait_until_queryable(database.build_adapter, START_TIMEOUT_S)

            yield database.build_adapter()
        finally:
            compose("down", "--volumes")
