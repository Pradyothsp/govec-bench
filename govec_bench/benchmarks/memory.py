import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.qdrant_adapter import QdrantAdapter
from govec_bench.benchmarks.common import compose, docker, load_dataset, wait_until_queryable
from govec_bench.datasets.sift import load_sift100k
from govec_bench.results import write_results

# Generous: this waits out a full container recreation (docker compose down
# -v && up -d), not just a process restart like cold_start.py's POLL_TIMEOUT_S.
POLL_TIMEOUT_S = 120

# RAM checkpoints, seconds after both loads finish. Not a single "settled"
# snapshot: govec's RSS drops sharply over the first few minutes post-load as
# Go's scavenger returns freed pages to the OS (see STATUS.md #14) -- a
# one-shot measurement just encodes whatever moment you happened to sample.
# Shortened to (0, 60) for routine runs -- STATUS.md #15 already established
# the full descent curve out to t=300 (RSS is still descending even there, so
# it was never a "settled" number either); this is a quick before/after
# sanity check, not a re-run of that full characterization. The only
# timing-independent figure is #14's pprof inuse_space (live heap).
RAM_SAMPLE_DELAYS_S = (0, 60)

GOVEC_CONTAINER = "govec-bench-govec"
GOVEC_DISK_PATHS = ("/data/govec_data.bin", "/data/govec.wal")
GOVEC_SCALAR_CONTAINER = "govec-bench-govec-scalar"
CHROMA_CONTAINER = "govec-bench-chroma"
CHROMA_DISK_PATHS = ("/data",)
QDRANT_CONTAINER = "govec-bench-qdrant"
QDRANT_DISK_PATHS = ("/qdrant/storage",)

# name -> (adapter constructor, container name, on-disk paths to measure)
DBS: dict[str, tuple[Callable[[], VectorDBAdapter], str, tuple[str, ...]]] = {
    "govec": (GovecAdapter, GOVEC_CONTAINER, GOVEC_DISK_PATHS),
    "govec-scalar": (partial(GovecAdapter, port=9699), GOVEC_SCALAR_CONTAINER, GOVEC_DISK_PATHS),
    "chroma": (ChromaAdapter, CHROMA_CONTAINER, CHROMA_DISK_PATHS),
    "qdrant": (QdrantAdapter, QDRANT_CONTAINER, QDRANT_DISK_PATHS),
}

_SIZE_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}


@dataclass(frozen=True, slots=True)
class FootprintStats:
    disk_bytes: int
    ram_bytes: int


class RamSample(NamedTuple):
    t_s: int
    ram_bytes: int


@dataclass(frozen=True, slots=True)
class MemoryResult:
    baseline: FootprintStats
    delta_disk_bytes: int
    ram_series: list[RamSample]
    delta_ram_bytes_peak: int
    delta_ram_bytes_last_sample: int


def _du_bytes(container: str, paths: tuple[str, ...]) -> int:
    # govec_data.bin/govec.wal don't exist until the first flush/write on a
    # freshly created container -- `du` on a missing path errors, so probe
    # existence first and treat "missing" as 0 bytes.
    total = 0
    for path in paths:
        result = docker("exec", container, "sh", "-c", f"test -e '{path}' && du -sb '{path}' || echo 0")
        total += int(result.stdout.split()[0])
    return total


DISK_STABILIZE_POLL_S = 3.0
DISK_STABILIZE_TIMEOUT_S = 120.0


def _stable_du_bytes(container: str, paths: tuple[str, ...]) -> int:
    # Qdrant keeps merging/vacuuming on-disk segments well after indexing
    # itself catches up -- a du taken right after load can land mid-merge,
    # holding old+new segment copies at once (observed: 2.85GB immediately
    # after a 100k load, settling to 231MB five seconds later). Poll until two
    # consecutive reads agree instead of trusting a single snapshot.
    deadline = time.monotonic() + DISK_STABILIZE_TIMEOUT_S
    previous = _du_bytes(container, paths)
    while time.monotonic() < deadline:
        time.sleep(DISK_STABILIZE_POLL_S)
        current = _du_bytes(container, paths)
        if current == previous:
            return current
        previous = current
    return previous


def _parse_docker_mem_size(size: str) -> int:
    # Longest suffix first: "MiB".endswith("B") is also true, so checking "B"
    # before "MiB" would strip only the "B" and leave "16.46Mi" behind.
    for unit in sorted(_SIZE_UNITS, key=len, reverse=True):
        if size.endswith(unit):
            return int(float(size.removesuffix(unit)) * _SIZE_UNITS[unit])
    msg = f"unrecognized docker stats memory size: {size!r}"
    raise ValueError(msg)


def _ram_bytes(container: str) -> int:
    # "docker stats" reads the container's cgroup memory directly -- no
    # per-database code needed, works identically for govec and Chroma.
    used = docker("stats", "--no-stream", "--format", "{{.MemUsage}}", container).stdout
    return _parse_docker_mem_size(used.split("/")[0].strip())


def _measure(container: str, disk_paths: tuple[str, ...]) -> FootprintStats:
    return FootprintStats(disk_bytes=_stable_du_bytes(container, disk_paths), ram_bytes=_ram_bytes(container))


def _ram_series(containers: dict[str, str]) -> dict[str, list[RamSample]]:
    # Sampled in the same pass for every DB at each checkpoint, so all DBs
    # share one wall-clock timeline instead of paying the wait once per DB.
    series: dict[str, list[RamSample]] = {name: [] for name in containers}
    start = time.monotonic()
    for delay in RAM_SAMPLE_DELAYS_S:
        wait = delay - (time.monotonic() - start)
        if wait > 0:
            time.sleep(wait)
        for name, container in containers.items():
            series[name].append(RamSample(t_s=delay, ram_bytes=_ram_bytes(container)))
    return series


def _build_result(baseline: FootprintStats, disk_after: int, ram_series: list[RamSample]) -> MemoryResult:
    return MemoryResult(
        baseline=baseline,
        delta_disk_bytes=disk_after - baseline.disk_bytes,
        ram_series=ram_series,
        delta_ram_bytes_peak=ram_series[0].ram_bytes - baseline.ram_bytes,
        delta_ram_bytes_last_sample=ram_series[-1].ram_bytes - baseline.ram_bytes,
    )


def _result_to_json(result: MemoryResult) -> dict[str, object]:
    return {
        "baseline": asdict(result.baseline),
        "delta_disk_bytes": result.delta_disk_bytes,
        "ram_series": [sample._asdict() for sample in result.ram_series],
        "delta_ram_bytes_peak": result.delta_ram_bytes_peak,
        "delta_ram_bytes_last_sample": result.delta_ram_bytes_last_sample,
    }


def main() -> None:
    # Fresh containers + a fresh named volume: repeated benchmark runs today
    # left Chroma's /data at 382MB of orphaned collection directories from
    # past reset() calls, which chromadb's own reset() doesn't clean up on
    # disk. A stale volume would make the disk numbers meaningless.
    print("Resetting containers to a clean state (docker compose down -v && up -d)...")
    compose("down", "-v")
    compose("up", "-d")

    print("Waiting for containers to become queryable...")
    adapters: dict[str, VectorDBAdapter] = {}
    for name, (adapter_cls, _container, _paths) in DBS.items():
        wait_until_queryable(adapter_cls, POLL_TIMEOUT_S)
        adapters[name] = adapter_cls()

    baselines = {name: _measure(container, paths) for name, (_cls, container, paths) in DBS.items()}

    dataset = load_sift100k()

    for name, adapter in adapters.items():
        print(f"Loading {len(dataset.base)} vectors into {name}...")
        load_dataset(adapter, dataset.base)
        if isinstance(adapter, GovecAdapter):
            adapter.flush()  # compact the WAL into the on-disk snapshot before measuring

    print("Waiting for on-disk footprint to stabilize (background segment merges/vacuums)...")
    disk_after = {name: _stable_du_bytes(container, paths) for name, (_cls, container, paths) in DBS.items()}

    print(f"Sampling RAM at t={RAM_SAMPLE_DELAYS_S}s after load...")
    ram_series = _ram_series({name: container for name, (_cls, container, _paths) in DBS.items()})

    results: dict[str, object] = {
        name: _result_to_json(_build_result(baselines[name], disk_after[name], ram_series[name])) for name in DBS
    }

    path = write_results(benchmark="memory", dataset="sift100k", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
