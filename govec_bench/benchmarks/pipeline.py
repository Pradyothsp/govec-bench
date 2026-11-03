"""Every benchmark from one load per database, at one size.

The pipeline is PIPELINE below: a list of stages, each run in its own fresh container, each a list
of steps. A step is a function of the shared Run: it reads what earlier steps left there and
records its measurement on run.record. Reorder, add or drop a step by editing a list; the results
file holds whatever the steps recorded.

The first stage loads the dataset once and measures everything against that index, so every
number for a database describes the same index. The other two are short: cold start with nothing
loaded, and one-at-a-time inserts, always on the 10k set (100k of them would take most of an hour
on Chroma and show nothing the 10k set doesn't).

Each step has a name, and --steps runs only some (with what they need: the load, for
anything measured on the loaded index): `--steps memory --size large` is what the memory
benchmark was, `--steps queries` the query and recall ones. This is the only implementation of
each measurement, so a fix to one lands in every result.

The ef sweep stays separate: it changes Qdrant's ef on one loaded graph, which arms of an
experiment (a container each) can't, and it shares the search measurement (search.py).
"""

import argparse
import random
import time
import traceback
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, NamedTuple

from tqdm import tqdm

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.registry import Database
from govec_bench.benchmarks.common import (
    POLL_INTERVAL_S,
    BenchArgs,
    add_size_argument,
    build_parser,
    compose,
    container_of,
    require_free_disk,
    running_alone,
    setup_of,
    to_bench_args,
    wait_until_queryable,
)
from govec_bench.benchmarks.concurrency import ConcurrencyPoint, measure_concurrency
from govec_bench.benchmarks.concurrency import K as CONCURRENT_K
from govec_bench.benchmarks.insert import BATCH_SIZE, InsertStats, measure_batch_insert, measure_single_insert
from govec_bench.benchmarks.memory import (
    FootprintStats,
    MemoryResult,
    build_memory_result,
    measure_footprint,
    memory_result_to_json,
    sample_ram_series,
    stable_disk_usage,
)
from govec_bench.benchmarks.search import measure_search, warm_up
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.datasets.registry import DatasetSizes, Size, load_sized
from govec_bench.results import LatencyStats, RecallStats, compute_latency_stats, write_results
from govec_bench.types import NeighborIndices, Vector

K_VALUES = (1, 5, 10, 50)

RESTARTS = 5

# A restart with 100k vectors loaded reads them all back before it answers; an empty one takes ms.
LOADED_START_TIMEOUT_S = 300
EMPTY_START_TIMEOUT_S = 60


class PipelineArgs(NamedTuple):
    bench: BenchArgs
    size: Size
    steps: list[str] | None  # None: every step


class Workload(NamedTuple):
    base: ArrayItems
    queries: list[Vector]
    groundtruth: list[NeighborIndices]
    single_insert_items: ArrayItems  # the 10k set, whatever the size
    dimensions: int


@dataclass(frozen=True, slots=True)
class QueryPoint:
    k: int
    recall: RecallStats
    latency: LatencyStats


@dataclass(slots=True)
class Record:
    # One field per measurement; a step that doesn't run leaves its field empty.
    batch_insert: InsertStats | None = None
    memory: MemoryResult | None = None
    queries: list[QueryPoint] = field(default_factory=list)
    queries_grpc: list[QueryPoint] = field(default_factory=list)  # same index, other transport
    concurrent_queries: list[ConcurrencyPoint] = field(default_factory=list)
    concurrent_queries_grpc: list[ConcurrencyPoint] = field(default_factory=list)
    cold_start_loaded: LatencyStats | None = None
    cold_start_empty: LatencyStats | None = None
    single_insert: InsertStats | None = None


@dataclass(slots=True)
class Run:
    # What steps share for one database. The runner sets adapter and container for each stage.
    database: Database
    workload: Workload
    record: Record = field(default_factory=Record)
    adapter: VectorDBAdapter | None = None
    container: str = ""
    baseline: FootprintStats | None = None  # the empty container's footprint, before loading

    def require_adapter(self) -> VectorDBAdapter:
        if self.adapter is None:
            msg = "a step that needs a database ran outside a stage"
            raise RuntimeError(msg)

        return self.adapter


# Told each stage and step as it starts, besides the printed line: bench run's live view uses it.
Progress = Callable[[Literal["stage", "step"], str], None]


def _no_progress(_kind: Literal["stage", "step"], _label: str) -> None:
    pass


class Step(NamedTuple):
    name: str  # what --steps and an experiment's steps call it
    label: str
    run: Callable[[Run], None]
    needs: tuple[str, ...] = ()  # steps that must run before it in its stage


class Stage(NamedTuple):
    label: str
    steps: list[Step]


def parse_pipeline_args(argv: Sequence[str] | None = None) -> PipelineArgs:
    parser = build_parser()
    add_size_argument(parser)
    parser.add_argument(
        "--steps",
        type=lambda text: [name.strip() for name in text.split(",") if name.strip()],
        help=f"run only these steps, and what they need, comma-separated: {', '.join(MEASUREMENTS)}",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    if args.steps is not None:
        try:
            select_steps(PIPELINE, args.steps)
        except ValueError as exc:
            parser.error(str(exc))

    return PipelineArgs(bench=to_bench_args(args), size=args.size, steps=args.steps)


# What each step prints as soon as it has its result.


def show(line: str) -> None:
    print(f"      -> {line}")


def latency_text(stats: LatencyStats) -> str:
    return f"mean {stats.mean_ms:.2f} ms, p50 {stats.p50_ms:.2f}, p99 {stats.p99_ms:.2f}"


def insert_text(stats: InsertStats) -> str:
    return (
        f"{stats.total_s:.1f} s until indexed, {stats.per_vector_ms:.2f} ms per vector; "
        f"calls alone {latency_text(stats.calls)}"
    )


def mb(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:,.0f} MB"


# Steps: each records one measurement, or prepares the index for the next.


def measure_baseline(run: Run) -> None:
    run.baseline = measure_footprint(run.container, run.database.disk_paths)

    show(f"empty: disk {mb(run.baseline.disk_allocated_bytes)} allocated, RAM {mb(run.baseline.ram_bytes)}")


def batch_insert(run: Run) -> None:
    run.record.batch_insert = measure_batch_insert(run.require_adapter(), run.workload.base, BATCH_SIZE)

    show(insert_text(run.record.batch_insert))


def settle(run: Run) -> None:
    adapter = run.require_adapter()
    adapter.wait_until_settled()

    count = adapter.stats().count
    if count != len(run.workload.base):
        msg = f"loaded {len(run.workload.base)} vectors but the database reports {count}"
        raise RuntimeError(msg)

    show(f"{count:,} vectors, settled")


def save(run: Run) -> None:
    # Chroma and Qdrant save as they go; GoVec saves when told to (auto-save is off in the bench
    # config), so its disk size and its loaded restart both need the snapshot written.
    adapter = run.require_adapter()
    if isinstance(adapter, GovecAdapter):
        adapter.flush()
        show("snapshot written")
    else:
        show("saves as it goes; nothing to do")


def measure_memory(run: Run) -> None:
    if run.baseline is None:
        msg = "measure_memory needs measure_baseline earlier in the stage"
        raise RuntimeError(msg)

    disk_after = stable_disk_usage(run.container, run.database.disk_paths)
    run.record.memory = build_memory_result(run.baseline, disk_after, sample_ram_series(run.container))

    memory = run.record.memory
    baseline = memory.baseline
    show(f"added disk: {mb(memory.delta_disk_bytes)} apparent, {mb(memory.delta_disk_allocated_bytes)} allocated")
    for sample in memory.ram_series:
        show(
            f"added RAM at {sample.t_s} s: {mb(sample.ram_bytes - baseline.ram_bytes)} (docker stats); "
            f"process {mb(sample.process_bytes - baseline.ram_process_bytes)}, "
            f"file cache {mb(sample.file_cache_bytes - baseline.ram_file_cache_bytes)}"
        )


def measure_query_points(adapter: VectorDBAdapter, workload: Workload, label: str) -> list[QueryPoint]:
    warm_up(adapter, workload.queries, max(K_VALUES))

    points = []
    for k in K_VALUES:
        search = measure_search(adapter, workload.base, workload.queries, workload.groundtruth, k)
        point = QueryPoint(k=k, recall=search.recall, latency=search.latency)
        points.append(point)

        recall = f"recall {point.recall.mean:.4f} (worst {point.recall.min:.2f})"
        show(f"k={k}{label}: {recall}; {latency_text(point.latency)}")

    return points


def measure_queries(run: Run) -> None:
    run.record.queries = measure_query_points(run.require_adapter(), run.workload, "")


def measure_queries_grpc(run: Run) -> None:
    # The same index and container, queried over gRPC instead of REST: only the transport differs.
    # Recall should match the REST step exactly; a difference would mean a different search.
    if run.database.build_grpc_adapter is None:
        show("no gRPC client for this database; skipped")
        return

    run.record.queries_grpc = measure_query_points(run.database.build_grpc_adapter(), run.workload, " over gRPC")


def concurrency_text(point: ConcurrencyPoint) -> str:
    errors = f", {point.errors} errors" if point.errors else ""
    return (
        f"{point.concurrency:>2} clients: {point.throughput_qps:,.0f} queries/s; {latency_text(point.latency)}, "
        f"p99.9 {point.p999_ms:.2f}; recall {point.recall_mean:.4f}; "
        f"CPU database {point.server_cpu_cores:.2f}, clients {point.client_cpu_cores:.2f}{errors}"
    )


def _expected_ids(workload: Workload) -> list[frozenset[str]]:
    return [frozenset(workload.base.id_at(i) for i in truth[:CONCURRENT_K]) for truth in workload.groundtruth]


def measure_concurrent_queries(run: Run) -> None:
    run.record.concurrent_queries = measure_concurrency(
        run.database.build_adapter,
        run.container,
        run.workload.queries,
        _expected_ids(run.workload),
        report=lambda point: show(concurrency_text(point)),
    )


def measure_concurrent_queries_grpc(run: Run) -> None:
    if run.database.build_grpc_adapter is None:
        show("no gRPC client for this database; skipped")
        return

    run.record.concurrent_queries_grpc = measure_concurrency(
        run.database.build_grpc_adapter,
        run.container,
        run.workload.queries,
        _expected_ids(run.workload),
        report=lambda point: show(f"{concurrency_text(point)} (gRPC)"),
    )


def wait_until_answering(build_adapter: Callable[[], VectorDBAdapter], probe: Vector, timeout_s: float) -> None:
    # Not common.wait_until_queryable: its probe is reset(), which would delete the loaded data.
    # Answering a query means the data is back, not only that the process is up.
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            if build_adapter().query(probe, k=1):
                return
        except Exception:  # noqa: BLE001, S110 -- expected while the data is still loading; keep polling
            pass
        time.sleep(POLL_INTERVAL_S)

    msg = f"loaded database did not answer within {timeout_s}s"
    raise TimeoutError(msg)


def measure_restarts(service: str, wait: Callable[[], None]) -> LatencyStats:
    # Timed from before `compose start`: the server boots while that command runs, so starting
    # the clock after it timed one request.
    latencies_ms = []
    for _ in tqdm(range(RESTARTS), desc="restarts", leave=False):
        compose("stop", service)

        start = time.perf_counter()
        compose("start", service)
        wait()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return compute_latency_stats(latencies_ms)


def measure_loaded_cold_start(run: Run) -> None:
    database, probe = run.database, run.workload.queries[0]
    run.record.cold_start_loaded = measure_restarts(
        database.service, lambda: wait_until_answering(database.build_adapter, probe, LOADED_START_TIMEOUT_S)
    )

    show(f"until it answers: {latency_text(run.record.cold_start_loaded)}")


def measure_empty_cold_start(run: Run) -> None:
    # The stage's container is fresh and empty, so every restart restarts an empty index.
    database = run.database
    run.record.cold_start_empty = measure_restarts(
        database.service, lambda: wait_until_queryable(database.build_adapter, EMPTY_START_TIMEOUT_S)
    )

    show(f"until it answers: {latency_text(run.record.cold_start_empty)}")


def single_insert(run: Run) -> None:
    run.record.single_insert = measure_single_insert(run.require_adapter(), run.workload.single_insert_items)

    show(insert_text(run.record.single_insert))


# The pipeline: stages in order, each in a fresh container; steps in order within a stage.
PIPELINE: list[Stage] = [
    Stage(
        "loaded index",
        [
            Step("baseline", "baseline footprint", measure_baseline),
            Step("batch-insert", "batch insert", batch_insert),
            Step("settle", "settle", settle, needs=("batch-insert",)),
            Step("save", "save", save, needs=("settle",)),
            Step("memory", "disk and RAM (RAM sampled for 60 s)", measure_memory, needs=("baseline", "save")),
            Step("queries", "recall and query latency", measure_queries, needs=("settle",)),
            Step("queries-grpc", "recall and query latency over gRPC", measure_queries_grpc, needs=("settle",)),
            Step(
                "concurrent-queries",
                "throughput and latency with concurrent clients",
                measure_concurrent_queries,
                needs=("settle",),
            ),
            Step(
                "concurrent-queries-grpc",
                "throughput and latency with concurrent clients over gRPC",
                measure_concurrent_queries_grpc,
                needs=("settle",),
            ),
            Step(
                "restart-loaded",
                f"{RESTARTS} restarts with the data loaded",
                measure_loaded_cold_start,
                needs=("save",),
            ),
        ],
    ),
    Stage("empty", [Step("restart-empty", f"{RESTARTS} restarts empty", measure_empty_cold_start)]),
    Stage("single inserts", [Step("single-insert", "one-at-a-time inserts, 10k", single_insert)]),
]

# The steps that measure something; the rest prepare the index for them.
MEASUREMENTS = (
    "batch-insert",
    "memory",
    "queries",
    "queries-grpc",
    "concurrent-queries",
    "concurrent-queries-grpc",
    "restart-loaded",
    "restart-empty",
    "single-insert",
)


def steps_of(pipeline: Sequence[Stage]) -> list[str]:
    return [step.name for stage in pipeline for step in stage.steps]


def select_steps(pipeline: Sequence[Stage], names: Collection[str]) -> list[Stage]:
    # The named steps and every step they need, in pipeline order; a stage with none is left out,
    # so its container never starts.
    steps = {step.name: step for stage in pipeline for step in stage.steps}
    unknown = sorted(set(names) - set(steps))
    if unknown:
        msg = f"no step {', '.join(unknown)}; the steps are {', '.join(steps)}"
        raise ValueError(msg)

    wanted: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name not in wanted:
            wanted.add(name)
            pending.extend(steps[name].needs)

    selected = [Stage(stage.label, [step for step in stage.steps if step.name in wanted]) for stage in pipeline]
    return [stage for stage in selected if stage.steps]


def run_pipeline(run: Run, pipeline: Sequence[Stage], progress: Progress = _no_progress) -> None:
    for stage in pipeline:
        print(f"  [{stage.label}]")
        progress("stage", stage.label)
        with running_alone(run.database, dimensions=run.workload.dimensions) as adapter:
            run.adapter = adapter
            run.container = container_of(run.database.service)

            for step in stage.steps:
                print(f"    {step.label}...")
                progress("step", step.label)
                step.run(run)

        run.adapter = None


def shuffled(names: Sequence[str]) -> list[str]:
    # A fresh database order every run. A night of runs drifts (heat, background load), and in a
    # fixed order that drift lands on the same database every time; medians can't remove it.
    return random.sample(list(names), k=len(names))


def result_to_json(result: Record, run_position: int, error: str | None = None) -> dict[str, object]:
    # A database that failed partway keeps what it measured before the error. run_position is
    # where it ran in this run's order (0 = first), to check results for an order effect.
    payload: dict[str, object] = {"run_position": run_position}
    if error is not None:
        payload["error"] = error
    for name in ("batch_insert", "cold_start_loaded", "cold_start_empty", "single_insert"):
        stats: InsertStats | LatencyStats | None = getattr(result, name)
        if stats is not None:
            payload[name] = asdict(stats)

    if result.memory is not None:
        payload["memory"] = memory_result_to_json(result.memory)

    for name, points in (("queries", result.queries), ("queries_grpc", result.queries_grpc)):
        if points:
            payload[name] = {
                f"k_{point.k}": {"recall": asdict(point.recall), "latency": asdict(point.latency)} for point in points
            }

    for name, levels in (
        ("concurrent_queries", result.concurrent_queries),
        ("concurrent_queries_grpc", result.concurrent_queries_grpc),
    ):
        if levels:
            payload[name] = {f"c_{point.concurrency}": asdict(point) for point in levels}

    return payload


def build_workload(sizes: DatasetSizes, size: Size) -> tuple[str, Workload]:
    # The dataset's name, and what every database is loaded and graded with. Built once and
    # shared: the exact neighbours of 100k vectors take a while to compute.
    dataset = load_sized(sizes, size)
    workload = Workload(
        base=dataset.base,
        queries=dataset.queries,
        groundtruth=exact_cosine_neighbors(dataset.base.vectors, dataset.queries, max(K_VALUES)),
        single_insert_items=sizes.small().base,
        dimensions=sizes.dimensions,
    )
    return dataset.name, workload


def measure_database(
    database: Database,
    workload: Workload,
    run_position: int,
    progress: Progress = _no_progress,
    pipeline: Sequence[Stage] = PIPELINE,
) -> dict[str, object]:
    # Every stage for one database. A failure partway is recorded with what came before it, so
    # one database failing doesn't discard the others' results.
    run = Run(database=database, workload=workload)
    error = None
    try:
        run_pipeline(run, pipeline, progress)
    except Exception as exc:  # noqa: BLE001 -- recorded in the results instead
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"

    return result_to_json(run.record, run_position, error)


def main() -> None:
    args = parse_pipeline_args()
    if args.size == "large":
        require_free_disk()
    dataset_name, workload = build_workload(args.bench.dataset, args.size)

    pipeline = select_steps(PIPELINE, args.steps) if args.steps else PIPELINE
    databases = args.bench.databases
    order = shuffled(list(databases))
    print(f"Run order: {', '.join(order)}")

    results: dict[str, object] = {}
    for position, name in enumerate(order):
        print(f"{name}:")
        results[name] = measure_database(databases[name], workload, position, pipeline=pipeline)

    # Registry order in the file, whatever order they ran in, so every results file lists databases alike.
    results = {name: results[name] for name in databases}
    path = write_results(
        benchmark="pipeline",
        dataset=dataset_name,
        results=results,
        setup=setup_of(databases),
        steps=steps_of(pipeline) if args.steps else None,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
