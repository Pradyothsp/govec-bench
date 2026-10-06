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


GOVEC_DISK_PATHS = ("/data/govec_data.bin", "/data/govec.wal")

# Keyed by docker-compose.yml service name, which is also the name results are written under.
# The one list of databases: every benchmark runs these, unless --db narrows it.
DATABASES: dict[str, Database] = {
    "govec": Database(GovecAdapter, GOVEC_DISK_PATHS),
    # GoVec again with int8 scalar quantization (govec-config-scalar.yaml), to measure what
    # quantization costs and saves against float32.
    "govec-scalar": Database(partial(GovecAdapter, port=9699), GOVEC_DISK_PATHS),
    "chroma": Database(ChromaAdapter, ("/data",)),
    "qdrant": Database(QdrantAdapter, ("/qdrant/storage",)),
}


def select_databases(names: Iterable[str] | None = None) -> dict[str, Database]:
    if names is None:
        return dict(DATABASES)

    selected = set(names)
    # Registry order, whatever order the names came in, so results files always list databases alike.
    return {name: database for name, database in DATABASES.items() if name in selected}
