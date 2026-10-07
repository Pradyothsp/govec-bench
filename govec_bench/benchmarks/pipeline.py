"""Every benchmark from one load per database, at one size.

The pipeline is PIPELINE below: a list of stages, each run in its own fresh container, each a list
of steps. A step is a function of the shared Run: it reads what earlier steps left there and
records its measurement on run.record. Reorder, add or drop a step by editing a list; the results
file holds whatever the steps recorded.

The first stage loads the dataset once and measures everything against that index, so every
number for a database describes the same index. The other two are short: cold start with nothing
loaded, and one-at-a-time inserts, always on the 10k set (100k of them would take most of an hour
on Chroma and show nothing the 10k set doesn't).

The ef sweep stays separate: GoVec and Chroma need a build per ef.
"""

import argparse
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

from tqdm import tqdm

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.registry import Database
from govec_bench.benchmarks.cold_start import measure_cold_start
from govec_bench.benchmarks.common import (
    POLL_INTERVAL_S,
    BenchArgs,
    add_size_argument,
    build_parser,
    compose,
    container_of,
    require_free_disk,
    running_alone,
    to_bench_args,
)
from govec_bench.benchmarks.insert import BATCH_SIZE, measure_batch_insert, measure_single_insert
from govec_bench.benchmarks.memory import (
    FootprintStats,
    MemoryResult,
    build_memory_result,
    measure_footprint,
    memory_result_to_json,
    sample_ram_series,
    stable_disk_usage,
)
from govec_bench.datasets.base import ArrayItems
from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.datasets.registry import Size, load_sized
from govec_bench.results import LatencyStats, RecallStats, compute_latency_stats, compute_recall_stats, write_results
from govec_bench.types import NeighborIndices, Vector

K_VALUES = (1, 5, 10, 50)

RESTARTS = 5

# A restart with 100k vectors loaded reads them all back before it answers; an empty one takes ms.
LOADED_START_TIMEOUT_S = 300


class PipelineArgs(NamedTuple):
    bench: BenchArgs
    size: Size


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
    batch_insert: LatencyStats | None = None
    memory: MemoryResult | None = None
    queries: list[QueryPoint] = field(default_factory=list)
    queries_grpc: list[QueryPoint] = field(default_factory=list)  # same index, other transport
    cold_start_loaded: LatencyStats | None = None
    cold_start_empty: LatencyStats | None = None
    single_insert: LatencyStats | None = None


@dataclass(slots=True)
class Run:
    # What steps share for one database. The runner sets adapter and container for each stage.
    service: str
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


class Step(NamedTuple):
    label: str
    run: Callable[[Run], None]


class Stage(NamedTuple):
    label: str
    steps: list[Step]


def parse_pipeline_args(argv: Sequence[str] | None = None) -> PipelineArgs:
    parser = build_parser()
    add_size_argument(parser)
    args: argparse.Namespace = parser.parse_args(argv)

    return PipelineArgs(bench=to_bench_args(args), size=args.size)


# What each step prints as soon as it has its result.


def show(line: str) -> None:
    print(f"      -> {line}")


def latency_text(stats: LatencyStats) -> str:
    return f"mean {stats.mean_ms:.2f} ms, p50 {stats.p50_ms:.2f}, p99 {stats.p99_ms:.2f}"


def mb(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:,.0f} MB"


# Steps: each records one measurement, or prepares the index for the next.


def measure_baseline(run: Run) -> None:
    run.baseline = measure_footprint(run.container, run.database.disk_paths)

    show(f"empty: disk {mb(run.baseline.disk_allocated_bytes)} allocated, RAM {mb(run.baseline.ram_bytes)}")


def batch_insert(run: Run) -> None:
    run.record.batch_insert = measure_batch_insert(run.require_adapter(), run.workload.base, BATCH_SIZE)

    show(f"per vector: {latency_text(run.record.batch_insert)}")


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


def query_point(adapter: VectorDBAdapter, workload: Workload, k: int) -> QueryPoint:
    # Each query timed and graded in the same call, so recall and latency describe the same search.
    latencies_ms = []
    recalls = []
    pairs = zip(workload.queries, workload.groundtruth, strict=True)
    for query, truth in tqdm(pairs, total=len(workload.queries), desc=f"queries k={k}", leave=False):
        start = time.perf_counter()
        results = adapter.query(query, k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        expected = {workload.base.id_at(i) for i in truth[:k]}
        recalls.append(len(expected & {r.id for r in results}) / k)

    return QueryPoint(k=k, recall=compute_recall_stats(recalls), latency=compute_latency_stats(latencies_ms))


def measure_query_points(adapter: VectorDBAdapter, workload: Workload, label: str) -> list[QueryPoint]:
    # One untimed pass first, so k=1 doesn't also pay for cold caches.
    for query in tqdm(workload.queries, desc="warm-up", leave=False):
        adapter.query(query, k=max(K_VALUES))

    points = []
    for k in K_VALUES:
        point = query_point(adapter, workload, k)
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


def wait_until_answering(build_adapter: Callable[[], VectorDBAdapter], probe: Vector, timeout_s: float) -> float:
    # Not common.wait_until_queryable: its probe is reset(), which would delete the loaded data.
    # Answering a query means the data is back, not only that the process is up.
    start = time.perf_counter()
    deadline = start + timeout_s
    while time.perf_counter() < deadline:
        try:
            if build_adapter().query(probe, k=1):
                return (time.perf_counter() - start) * 1000
        except Exception:  # noqa: BLE001, S110 -- expected while the data is still loading; keep polling
            pass
        time.sleep(POLL_INTERVAL_S)

    msg = f"loaded database did not answer within {timeout_s}s"
    raise TimeoutError(msg)


def measure_loaded_cold_start(run: Run) -> None:
    probe = run.workload.queries[0]
    latencies_ms = []
    for _ in tqdm(range(RESTARTS), desc="restarts", leave=False):
        compose("stop", run.service)
        compose("start", run.service)
        latencies_ms.append(wait_until_answering(run.database.build_adapter, probe, LOADED_START_TIMEOUT_S))

    run.record.cold_start_loaded = compute_latency_stats(latencies_ms)

    show(f"until it answers: {latency_text(run.record.cold_start_loaded)}")


def measure_empty_cold_start(run: Run) -> None:
    run.record.cold_start_empty = measure_cold_start(run.service, run.database.build_adapter)

    show(f"until it answers: {latency_text(run.record.cold_start_empty)}")


def single_insert(run: Run) -> None:
    run.record.single_insert = measure_single_insert(run.require_adapter(), run.workload.single_insert_items)

    show(f"per vector: {latency_text(run.record.single_insert)}")


# The pipeline: stages in order, each in a fresh container; steps in order within a stage.
PIPELINE: list[Stage] = [
    Stage(
        "loaded index",
        [
            Step("baseline footprint", measure_baseline),
            Step("batch insert", batch_insert),
            Step("settle", settle),
            Step("save", save),
            Step("disk and RAM (RAM sampled for 60 s)", measure_memory),
            Step("recall and query latency", measure_queries),
            Step("recall and query latency over gRPC", measure_queries_grpc),
            Step(f"{RESTARTS} restarts with the data loaded", measure_loaded_cold_start),
        ],
    ),
    Stage("empty", [Step(f"{RESTARTS} restarts empty", measure_empty_cold_start)]),
    Stage("single inserts", [Step("one-at-a-time inserts, 10k", single_insert)]),
]


def run_pipeline(run: Run, pipeline: Sequence[Stage]) -> None:
    for stage in pipeline:
        print(f"  [{stage.label}]")
        with running_alone(run.service, run.database, dimensions=run.workload.dimensions) as adapter:
            run.adapter = adapter
            run.container = container_of(run.service)

            for step in stage.steps:
                print(f"    {step.label}...")
                step.run(run)

        run.adapter = None


def result_to_json(result: Record, error: str | None = None) -> dict[str, object]:
    # A database that failed partway keeps what it measured before the error.
    payload: dict[str, object] = {} if error is None else {"error": error}
    for name in ("batch_insert", "cold_start_loaded", "cold_start_empty", "single_insert"):
        stats: LatencyStats | None = getattr(result, name)
        if stats is not None:
            payload[name] = asdict(stats)

    if result.memory is not None:
        payload["memory"] = memory_result_to_json(result.memory)

    for name, points in (("queries", result.queries), ("queries_grpc", result.queries_grpc)):
        if points:
            payload[name] = {
                f"k_{point.k}": {"recall": asdict(point.recall), "latency": asdict(point.latency)} for point in points
            }

    return payload


def main() -> None:
    args = parse_pipeline_args()
    if args.size == "large":
        require_free_disk()
    dataset = load_sized(args.bench.dataset, args.size)
    workload = Workload(
        base=dataset.base,
        queries=dataset.queries,
        groundtruth=exact_cosine_neighbors(dataset.base.vectors, dataset.queries, max(K_VALUES)),
        single_insert_items=args.bench.dataset.small().base,
        dimensions=args.bench.dataset.dimensions,
    )

    results: dict[str, object] = {}
    for name, database in args.bench.databases.items():
        print(f"{name}:")
        run = Run(service=name, database=database, workload=workload)
        error = None
        try:
            run_pipeline(run, PIPELINE)
        except Exception as exc:  # noqa: BLE001 -- one database failing shouldn't discard the others' results
            traceback.print_exc()
            error = f"{type(exc).__name__}: {exc}"

        results[name] = result_to_json(run.record, error)

    path = write_results(benchmark="pipeline", dataset=dataset.name, results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
