from collections.abc import Callable, Iterable
from functools import partial
from typing import NamedTuple

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.qdrant_adapter import QdrantAdapter


class Database(NamedTuple):
    # A constructor, not an adapter: constructing one connects (Chroma's does), so it can only
    # be built once that database's container is running.
    build_adapter: Callable[[], VectorDBAdapter]
    disk_paths: tuple[str, ...]  # what the memory benchmark measures inside the container
    # Must be told the vector width at startup (GoVec's mmap store sizes its slots from it), so
    # running_alone() passes it as GOVEC_DIMENSIONS. Everything else learns it from the first insert.
    needs_dimensions: bool = False
    # Runs only when named with --db: a variant for a specific comparison, not the standard suite.
    opt_in: bool = False


GOVEC_DISK_PATHS = ("/data/govec_data.bin", "/data/govec.wal")

# Keyed by docker-compose.yml service name, which is also the name results are written under.
# The one list of databases: every benchmark runs these, except opt-in ones, unless --db narrows it.
DATABASES: dict[str, Database] = {
    "govec": Database(GovecAdapter, GOVEC_DISK_PATHS),
    # GoVec again with int8 scalar quantization (govec-config-scalar.yaml), to measure what
    # quantization costs and saves against float32.
    "govec-scalar": Database(partial(GovecAdapter, port=9699), GOVEC_DISK_PATHS),
    # GoVec again with vectors in memory-mapped files (govec-config-mmap.yaml), to measure what
    # moving them out of the Go heap costs and saves.
    "govec-mmap": Database(
        partial(GovecAdapter, port=9700),
        (*GOVEC_DISK_PATHS, "/data/govec_mmap"),
        needs_dimensions=True,
        opt_in=True,
    ),
    "chroma": Database(ChromaAdapter, ("/data",)),
    "qdrant": Database(QdrantAdapter, ("/qdrant/storage",)),
}


def select_databases(names: Iterable[str] | None = None) -> dict[str, Database]:
    if names is None:
        return {name: database for name, database in DATABASES.items() if not database.opt_in}

    selected = set(names)
    # Registry order, whatever order the names came in, so results files always list databases alike.
    return {name: database for name, database in DATABASES.items() if name in selected}
