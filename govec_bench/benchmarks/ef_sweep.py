"""Recall@k against query latency as search ef grows, for every database on equal build settings.

Every database builds its graph with M=16, ef_construction=100 and cosine, then answers the
dataset's queries at each search ef. How ef changes differs by database:

- Qdrant takes it per query (search_params.hnsw_ef): one load, then every ef on the same graph.
- Chroma: a fresh container and collection per ef, created with hnsw:search_ef and read back to
  check. Chroma 1.4.4 accepts collection.modify(ef_search) and reports it applied, but its
  queries keep the ef the collection was created with.
- GoVec has no per-query ef, and up to 0.2.0 a restart couldn't change it either (a snapshot's
  saved ef_search overrode the configured one). So each ef is a fresh container started with
  GOVEC_HNSW_EF_SEARCH set, and a fresh load.

Chroma's and GoVec's points are each their own build, so their curves carry build-to-build noise
Qdrant's doesn't.

GoVec is also swept once at ef_construction=200, its default up to 0.2.0, as its own line.
"""

import json
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.qdrant_adapter import QdrantAdapter
from govec_bench.adapters.registry import Database
from govec_bench.benchmarks.common import (
    BenchArgs,
    build_parser,
    container_of,
    docker,
    load_dataset,
    running_alone,
    to_bench_args,
)
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.results import LatencyStats, RecallStats, compute_latency_stats, compute_recall_stats, write_results
from govec_bench.types import NeighborIndices, Vector

EF_VALUES = (25, 50, 100, 200, 400)

# k=50 is where ef below k stops mattering: every database searches with max(ef, k).
K_VALUES = (10, 50)

# Timed passes over the queries per point: one pass of 100 queries at ~1 ms each was too few
# samples to separate the databases on SIFT.
TIMED_PASSES = 5

# Equal build effort for every database: Chroma's and Qdrant's default.
EF_CONSTRUCTION = 100

# GoVec's default up to 0.2.0, swept as an extra line on the float32 service.
GOVEC_DEFAULT_EF_CONSTRUCTION = 200
GOVEC_DEFAULT_LINE_SERVICE = "govec"

# Swept when no --db is given. govec-scalar is left out until the int8 work needs its curve;
# --db govec-scalar still sweeps it.
DEFAULT_SERVICES = ("govec", "chroma", "qdrant")


class SweepArgs(NamedTuple):
    bench: BenchArgs
    ef_values: tuple[int, ...]
    k_values: tuple[int, ...]


class Workload(NamedTuple):
    base: ArrayItems
    queries: list[Vector]
    groundtruth: list[NeighborIndices]


class SeriesPlan(NamedTuple):
    label: str  # the key results are written under
    service: str
    ef_construction: int


@dataclass(frozen=True, slots=True)
class SweepPoint:
    ef: int
    k: int
    recall: RecallStats
    latency: LatencyStats


@dataclass(frozen=True, slots=True)
class SweepSeries:
    service: str
    ef_construction: int
    ef_applied_by: str
    points: list[SweepPoint]


@dataclass(frozen=True, slots=True)
class SweepFailure:
    service: str
    error: str


type SearchEfSetter = Callable[[VectorDBAdapter, int], None]

# Starts a database ready to load at (search ef, ef_construction).
type FreshAt = Callable[[str, Database, int, int], AbstractContextManager[VectorDBAdapter]]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        msg = f"must be at least 1, got {number}"
        raise ValueError(msg)

    return number


def parse_sweep_args(argv: Sequence[str] | None = None) -> SweepArgs:
    parser = build_parser()
    parser.add_argument(
        "--ef",
        action="append",
        type=_positive_int,
        help=f"search ef to measure; repeat for several (default: {', '.join(map(str, EF_VALUES))})",
    )
    parser.add_argument(
        "--k",
        action="append",
        type=_positive_int,
        help=f"k to grade recall at; repeat for several (default: {', '.join(map(str, K_VALUES))})",
    )
    args = parser.parse_args(argv)

    if args.db is None:
        args.db = list(DEFAULT_SERVICES)

    return SweepArgs(
        bench=to_bench_args(args),
        ef_values=tuple(sorted(set(args.ef or EF_VALUES))),
        k_values=tuple(sorted(set(args.k or K_VALUES))),
    )


def _set_qdrant_ef(adapter: VectorDBAdapter, ef: int) -> None:
    if not isinstance(adapter, QdrantAdapter):
        msg = f"expected a QdrantAdapter, got {type(adapter).__name__}"
        raise TypeError(msg)

    adapter.set_search_ef(ef)


def _check_container_env(service: str, expected: dict[str, str]) -> None:
    # Proof the override reached the container: a typo in a service name would otherwise start it
    # on its config file's ef, and every point would measure the same search.
    env = json.loads(docker("inspect", "--format", "{{json .Config.Env}}", container_of(service)).stdout)
    missing = [f"{key}={value}" for key, value in expected.items() if f"{key}={value}" not in env]
    if missing:
        msg = f"{service} started without {missing}"
        raise RuntimeError(msg)


@contextmanager
def _govec_at(service: str, database: Database, ef: int, ef_construction: int) -> Iterator[VectorDBAdapter]:
    environment = govec_env(ef_search=ef, ef_construction=ef_construction)

    with tempfile.TemporaryDirectory() as tmp:
        override = Path(tmp) / "ef-sweep.override.yml"
        override.write_text(render_override(service, environment))

        with running_alone(service, database, overrides=(override,)) as adapter:
            _check_container_env(service, environment)
            yield adapter


@contextmanager
def _chroma_at(service: str, database: Database, ef: int, ef_construction: int) -> Iterator[VectorDBAdapter]:
    with running_alone(service, database) as adapter:
        if not isinstance(adapter, ChromaAdapter):
            msg = f"expected a ChromaAdapter, got {type(adapter).__name__}"
            raise TypeError(msg)

        adapter.recreate_with_search_ef(ef)

        # The sweep leaves Chroma's build settings at its defaults; check they are the ones claimed.
        hnsw = adapter.hnsw_configuration()
        if hnsw.get("ef_construction") != ef_construction:
            msg = f"Chroma built with hnsw configuration {hnsw}, expected ef_construction={ef_construction}"
            raise RuntimeError(msg)

        yield adapter


# Databases whose search ef changes on a loaded graph, and how.
IN_PLACE: dict[str, SearchEfSetter] = {
    "qdrant": _set_qdrant_ef,
}

# Databases swept with a fresh container and load per ef, and how each starts.
REBUILD_PER_EF: dict[str, FreshAt] = {
    "govec": _govec_at,
    "govec-scalar": _govec_at,
    "chroma": _chroma_at,
}


def plan_series(services: Sequence[str]) -> list[SeriesPlan]:
    plans = []
    for service in services:
        plans.append(SeriesPlan(label=service, service=service, ef_construction=EF_CONSTRUCTION))

        if service == GOVEC_DEFAULT_LINE_SERVICE:
            plans.append(
                SeriesPlan(
                    label=f"{service}@efc{GOVEC_DEFAULT_EF_CONSTRUCTION}",
                    service=service,
                    ef_construction=GOVEC_DEFAULT_EF_CONSTRUCTION,
                )
            )

    return plans


def govec_env(ef_search: int, ef_construction: int) -> dict[str, str]:
    return {
        "GOVEC_HNSW_EF_SEARCH": str(ef_search),
        "GOVEC_HNSW_EF_CONSTRUCTION": str(ef_construction),
    }


def render_override(service: str, environment: dict[str, str]) -> str:
    # JSON is valid YAML, so Compose reads this as an override file without a YAML dependency.
    return json.dumps({"services": {service: {"environment": environment}}}, indent=2)


def measure_point(adapter: VectorDBAdapter, workload: Workload, ef: int, k: int) -> SweepPoint:
    latencies_ms = []
    recalls = []
    for _ in range(TIMED_PASSES):
        for query, truth in zip(workload.queries, workload.groundtruth, strict=True):
            start = time.perf_counter()
            results = adapter.query(query, k=k)
            latencies_ms.append((time.perf_counter() - start) * 1000)

            expected = {workload.base.id_at(i) for i in truth[:k]}
            recalls.append(len(expected & {r.id for r in results}) / k)

    return SweepPoint(ef=ef, k=k, recall=compute_recall_stats(recalls), latency=compute_latency_stats(latencies_ms))


def measure_ef(adapter: VectorDBAdapter, workload: Workload, ef: int, k_values: Sequence[int]) -> list[SweepPoint]:
    # One untimed pass first, so the first ef measured doesn't also pay for cold caches.
    for query in workload.queries:
        adapter.query(query, k=max(k_values))

    return [measure_point(adapter, workload, ef, k) for k in k_values]


def _load_checked(adapter: VectorDBAdapter, base: ArrayItems) -> None:
    load_dataset(adapter, base)

    count = adapter.stats().count
    if count != len(base):
        msg = f"loaded {len(base)} vectors but the database reports {count}"
        raise RuntimeError(msg)


def sweep_in_place(
    service: str, database: Database, workload: Workload, sweep: SweepArgs, set_ef: SearchEfSetter
) -> list[SweepPoint]:
    points = []
    with running_alone(service, database) as adapter:
        _load_checked(adapter, workload.base)

        for ef in sweep.ef_values:
            print(f"  ef={ef}")
            set_ef(adapter, ef)
            points.extend(measure_ef(adapter, workload, ef, sweep.k_values))

    return points


def sweep_by_rebuild(
    plan: SeriesPlan, database: Database, workload: Workload, sweep: SweepArgs, fresh_at: FreshAt
) -> list[SweepPoint]:
    points = []
    for ef in sweep.ef_values:
        print(f"  ef={ef} (fresh container)")
        with fresh_at(plan.service, database, ef, plan.ef_construction) as adapter:
            _load_checked(adapter, workload.base)

            points.extend(measure_ef(adapter, workload, ef, sweep.k_values))

    return points


def run_series(plan: SeriesPlan, database: Database, workload: Workload, sweep: SweepArgs) -> SweepSeries:
    if plan.service in IN_PLACE:
        points = sweep_in_place(plan.service, database, workload, sweep, IN_PLACE[plan.service])
        return SweepSeries(plan.service, plan.ef_construction, "loaded graph, ef changed in place", points)

    if plan.service in REBUILD_PER_EF:
        fresh_at = REBUILD_PER_EF[plan.service]
        points = sweep_by_rebuild(plan, database, workload, sweep, fresh_at)
        return SweepSeries(plan.service, plan.ef_construction, "fresh container and load per ef", points)

    msg = f"no way to set search ef for {plan.service}"
    raise ValueError(msg)


def format_table(results: dict[str, SweepSeries | SweepFailure]) -> str:
    lines = [f"{'series':<16} {'ef':>4} {'k':>3} {'recall':>7} {'min':>6} {'mean ms':>8} {'p50 ms':>7} {'p99 ms':>7}"]
    for label, series in results.items():
        if isinstance(series, SweepFailure):
            lines.append(f"{label:<16} FAILED: {series.error}")
            continue

        lines.extend(
            f"{label:<16} {p.ef:>4} {p.k:>3} {p.recall.mean:>7.4f} {p.recall.min:>6.2f} "
            f"{p.latency.mean_ms:>8.3f} {p.latency.p50_ms:>7.3f} {p.latency.p99_ms:>7.3f}"
            for p in series.points
        )

    return "\n".join(lines)


def main() -> None:
    sweep = parse_sweep_args()
    dataset = sweep.bench.dataset.small()
    workload = Workload(
        base=dataset.base,
        queries=dataset.queries,
        groundtruth=exact_cosine_neighbors(dataset.base.vectors, dataset.queries, max(sweep.k_values)),
    )

    results: dict[str, SweepSeries | SweepFailure] = {}
    for plan in plan_series(list(sweep.bench.databases)):
        print(f"Sweeping ef: {plan.label} (ef_construction={plan.ef_construction})...")
        try:
            results[plan.label] = run_series(plan, sweep.bench.databases[plan.service], workload, sweep)
        except Exception as exc:  # noqa: BLE001 -- one database failing shouldn't discard the others' results
            traceback.print_exc()
            results[plan.label] = SweepFailure(service=plan.service, error=f"{type(exc).__name__}: {exc}")

    path = write_results(
        benchmark="ef_sweep",
        dataset=dataset.name,
        results={label: asdict(series) for label, series in results.items()},
    )
    print(format_table(results))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
