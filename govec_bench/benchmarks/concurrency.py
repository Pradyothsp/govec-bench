"""Many clients querying a loaded index at once: throughput and latency per concurrency level.

Each level runs that many client processes, each with the database's own Python client and its
own connection, querying in a loop for a fixed window. Processes, not threads: a Python client
spends most of a query encoding and decoding, under the interpreter lock, so threads would
measure the lock instead of the database.

The clients share the machine with the database. On a 4-vCPU runner the database gets 2 CPUs
(its cap) and the clients the other 2, so at some level the clients run out first. Each level
records both: the database's CPU from its cgroup, the clients' from their own process time.
Throughput that stops rising while the database is under its cap means the clients were the
limit, and a result there says so.

Recall is graded on every query, so a database can't look faster under load by answering worse.
"""

import multiprocessing
import queue
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.benchmarks.common import docker
from govec_bench.results import LatencyStats, compute_latency_stats
from govec_bench.types import Vector

CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 32)
K = 10

# Every client starts at the same moment; what they do in the first WARMUP_S isn't counted, so
# connections are open and caches warm before the window opens.
WARMUP_S = 3.0
WINDOW_S = 20.0
# Spawning a client process and importing a database's client library takes a second or two each,
# more with 32 at once on a few cores.
CLIENT_START_TIMEOUT_S = 300.0
WARMUP_QUERIES = 5
P999_MIN_SAMPLES = 1000

# spawn, not fork: the same start on macOS and Linux, and no copy of the parent's state.
_CONTEXT = multiprocessing.get_context("spawn")


@dataclass(frozen=True, slots=True)
class ConcurrencyPoint:
    concurrency: int
    queries: int  # answered inside the window
    throughput_qps: float
    latency: LatencyStats
    p999_ms: float
    recall_mean: float
    server_cpu_cores: float  # the database container's CPU over the window; its cap is 2
    client_cpu_cores: float  # every client process's CPU over the window, added up
    errors: int


class ClientJob(NamedTuple):
    # What one client process queries with; sent to it when it starts.
    build_adapter: Callable[[], VectorDBAdapter]
    queries: Sequence[Vector]
    expected: Sequence[frozenset[str]]  # each query's true top-K IDs
    first: int  # the query it starts at, so clients don't all send the same one at once


class ClientResult(NamedTuple):
    latencies_ms: list[float]
    recalls: list[float]
    cpu_s: float
    errors: int  # queries that raised inside the window
    failure: str | None = None  # why the client itself failed; then the rest is empty


class _Window(NamedTuple):
    start: float  # time.monotonic(), the same clock in every process on a machine
    end: float


def parse_cpu_usage_us(cpu_stat: str) -> int:
    # cgroup v2's cpu.stat: "usage_usec <n>" among other lines.
    for line in cpu_stat.splitlines():
        key, _, value = line.partition(" ")
        if key == "usage_usec":
            return int(value)

    msg = "cpu.stat has no usage_usec"

    raise ValueError(msg)


def _server_cpu_us(container: str) -> int:
    # cgroup v1 reports nanoseconds in cpuacct.usage; v2 microseconds in cpu.stat.
    script = (
        "if [ -f /sys/fs/cgroup/cpu.stat ]; then cat /sys/fs/cgroup/cpu.stat; "
        "else echo usage_usec $(( $(cat /sys/fs/cgroup/cpuacct/cpuacct.usage) / 1000 )); fi"
    )

    return parse_cpu_usage_us(docker("exec", container, "sh", "-c", script).stdout)


def _client(
    job: ClientJob,
    ready: Barrier,
    window_box: multiprocessing.Queue[_Window],
    results: multiprocessing.Queue[ClientResult],
) -> None:
    # One client process: build a client, warm it, wait for every other client, then query in a
    # loop until the window closes. Only queries that start and end inside the window count.
    try:
        adapter = job.build_adapter()
        for i in range(WARMUP_QUERIES):
            adapter.query(job.queries[(job.first + i) % len(job.queries)], k=K)
        ready.wait(CLIENT_START_TIMEOUT_S)
        window = window_box.get(timeout=CLIENT_START_TIMEOUT_S)
        results.put(_query_until(adapter, job, window))

    except Exception as exc:  # noqa: BLE001 -- sent to the parent, which raises it
        # Break the barrier, so the parent and the other clients stop waiting for this one.
        ready.abort()
        results.put(ClientResult([], [], 0.0, 0, failure=f"{type(exc).__name__}: {exc}"))


def _query_until(adapter: VectorDBAdapter, job: ClientJob, window: _Window) -> ClientResult:
    latencies_ms: list[float] = []
    recalls: list[float] = []
    errors = 0
    cpu_at_start: float | None = None
    cpu_at_end = 0.0
    i = job.first

    while (now := time.monotonic()) < window.end:
        if cpu_at_start is None and now >= window.start:
            cpu_at_start = time.process_time()

        index = i % len(job.queries)
        i += 1

        start = time.monotonic()
        try:
            found = adapter.query(job.queries[index], k=K)

        except Exception:  # noqa: BLE001 -- counted; a database that errors under load is a result
            if start >= window.start:
                errors += 1
            continue

        end = time.monotonic()

        if start >= window.start and end <= window.end:
            latencies_ms.append((end - start) * 1000)
            recalls.append(len(job.expected[index] & {result.id for result in found}) / K)
            cpu_at_end = time.process_time()

    cpu_s = cpu_at_end - cpu_at_start if cpu_at_start is not None else 0.0
    return ClientResult(latencies_ms, recalls, max(cpu_s, 0.0), errors)


def _first_failure(results: multiprocessing.Queue[ClientResult]) -> str | None:
    try:
        while True:
            result = results.get(timeout=1)
            if result.failure is not None:
                return result.failure

    except queue.Empty:
        return "no client said why"


def measure_level(
    build_adapter: Callable[[], VectorDBAdapter],
    container: str,
    queries: Sequence[Vector],
    expected: Sequence[frozenset[str]],
    concurrency: int,
) -> ConcurrencyPoint:
    ready = _CONTEXT.Barrier(concurrency + 1)
    window_box: multiprocessing.Queue[_Window] = _CONTEXT.Queue()
    results: multiprocessing.Queue[ClientResult] = _CONTEXT.Queue()

    clients = [
        _CONTEXT.Process(
            target=_client,
            args=(
                ClientJob(build_adapter, queries, expected, n * len(queries) // concurrency),
                ready,
                window_box,
                results,
            ),
            daemon=True,
        )
        for n in range(concurrency)
    ]

    for client in clients:
        client.start()

    try:
        try:
            ready.wait(CLIENT_START_TIMEOUT_S)

        except threading.BrokenBarrierError as exc:
            msg = f"a client failed to start at concurrency {concurrency}: {_first_failure(results)}"
            raise RuntimeError(msg) from exc

        now = time.monotonic()
        window = _Window(start=now + WARMUP_S, end=now + WARMUP_S + WINDOW_S)

        for _ in clients:
            window_box.put(window)

        time.sleep(max(window.start - time.monotonic(), 0))
        server_cpu_start = _server_cpu_us(container)
        time.sleep(max(window.end - time.monotonic(), 0))
        server_cpu_end = _server_cpu_us(container)

        collected = [results.get(timeout=CLIENT_START_TIMEOUT_S) for _ in clients]

    finally:
        for client in clients:
            client.join(timeout=10)
            if client.is_alive():
                client.terminate()

    failures = [result.failure for result in collected if result.failure is not None]
    if failures:
        msg = f"a client failed at concurrency {concurrency}: {failures[0]}"
        raise RuntimeError(msg)

    return build_point(concurrency, collected, (server_cpu_end - server_cpu_start) / 1e6)


def p999(latencies_ms: Sequence[float]) -> float:
    # Under a thousand samples there is no 99.9th percentile to estimate: the slowest one stands in.
    if len(latencies_ms) < P999_MIN_SAMPLES:
        return max(latencies_ms)

    return statistics.quantiles(latencies_ms, n=1000)[998]


def build_point(concurrency: int, clients: Sequence[ClientResult], server_cpu_s: float) -> ConcurrencyPoint:
    latencies_ms = [latency for client in clients for latency in client.latencies_ms]
    recalls = [recall for client in clients for recall in client.recalls]
    if not latencies_ms:
        msg = f"no query finished inside the {WINDOW_S:.0f} s window at concurrency {concurrency}"
        raise RuntimeError(msg)

    return ConcurrencyPoint(
        concurrency=concurrency,
        queries=len(latencies_ms),
        throughput_qps=len(latencies_ms) / WINDOW_S,
        latency=compute_latency_stats(latencies_ms),
        p999_ms=p999(latencies_ms),
        recall_mean=statistics.mean(recalls),
        server_cpu_cores=server_cpu_s / WINDOW_S,
        client_cpu_cores=sum(client.cpu_s for client in clients) / WINDOW_S,
        errors=sum(client.errors for client in clients),
    )


def measure_concurrency(
    build_adapter: Callable[[], VectorDBAdapter],
    container: str,
    queries: Sequence[Vector],
    expected: Sequence[frozenset[str]],
    *,
    levels: Sequence[int] = CONCURRENCY_LEVELS,
    report: Callable[[ConcurrencyPoint], None] = lambda _point: None,
) -> list[ConcurrencyPoint]:
    points = []
    for concurrency in levels:
        point = measure_level(build_adapter, container, queries, expected, concurrency)
        points.append(point)
        report(point)

    return points
