from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import NamedTuple

import yaml

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.qdrant_adapter import QdrantAdapter

DATABASES_FILE = Path(__file__).resolve().parents[2] / "databases.yaml"


class Engine(NamedTuple):
    # A constructor, not an adapter: constructing one connects (Chroma's does), so it can only
    # be built once the container is running.
    build_adapter: Callable[[], VectorDBAdapter]
    disk_paths: tuple[str, ...]  # what the memory benchmark measures inside the container
    # A second client to the same container over another transport, for comparing query latency
    # on the same index. None for engines the bench drives one way only.
    build_grpc_adapter: Callable[[], VectorDBAdapter] | None = None


# One per docker-compose.yml service. A path that doesn't exist measures 0 (GoVec's mmap
# directory, unless mmap is on), so one list serves every setup of an engine.
ENGINES: dict[str, Engine] = {
    "govec": Engine(
        GovecAdapter,
        ("/data/govec_data.bin", "/data/govec.wal", "/data/govec_mmap"),
        partial(GovecAdapter, port=9698, protocol="grpc"),
    ),
    "chroma": Engine(ChromaAdapter, ("/data",)),
    "qdrant": Engine(QdrantAdapter, ("/qdrant/storage",), partial(QdrantAdapter, grpc=True)),
}


class Database(NamedTuple):
    name: str  # what results are written under
    service: str  # the docker-compose.yml service it runs as: its engine
    build_adapter: Callable[[], VectorDBAdapter]
    disk_paths: tuple[str, ...]
    env: dict[str, str]  # settings changed from the engine's config, applied at start
    # Must be told the vector width at startup (GoVec's mmap store sizes its slots from it), so
    # running_alone() passes it as GOVEC_DIMENSIONS. Everything else learns it from the first insert.
    needs_dimensions: bool = False
    # Runs only when named with --db: a setup for a specific comparison, not the standard suite.
    opt_in: bool = False
    build_grpc_adapter: Callable[[], VectorDBAdapter] | None = None
    # What to run instead of the image docker-compose.yml pins: an image, or a git ref of the
    # GoVec repo to build (see benchmarks/images.py). None runs the pin.
    image_ref: str | None = None
    # The container's memory limit instead of docker-compose.yml's, with no swap beyond it, so
    # data larger than the limit has to be read back from disk. None keeps the file's.
    memory_bytes: int | None = None


_DATABASE_KEYS = {"engine", "env", "opt_in", "needs_dimensions"}


def env_value(value: object) -> str:
    # YAML reads true, yes and on as booleans and 16 as a number; environment variables are
    # strings, and Go parses "true".
    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def parse_databases(text: str) -> dict[str, Database]:
    entries = yaml.safe_load(text) or {}
    if not isinstance(entries, dict):
        msg = f"expected databases by name, got a {type(entries).__name__}"
        raise TypeError(msg)

    databases: dict[str, Database] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            msg = f"{name}: expected its settings (engine: ...), got {entry!r}"
            raise TypeError(msg)

        unknown = set(entry) - _DATABASE_KEYS
        if unknown:
            msg = f"{name}: unknown keys {sorted(unknown)}; expected some of {sorted(_DATABASE_KEYS)}"
            raise ValueError(msg)

        engine_name = entry.get("engine")
        if engine_name not in ENGINES:
            msg = f"{name}: engine {engine_name!r} is not one of {sorted(ENGINES)}"
            raise ValueError(msg)

        engine = ENGINES[engine_name]
        databases[name] = Database(
            name=name,
            service=engine_name,
            build_adapter=engine.build_adapter,
            disk_paths=engine.disk_paths,
            env={key: env_value(value) for key, value in entry.get("env", {}).items()},
            needs_dimensions=entry.get("needs_dimensions", False),
            opt_in=entry.get("opt_in", False),
            build_grpc_adapter=engine.build_grpc_adapter,
        )

    return databases


# The one list of databases: every benchmark runs these, except opt-in ones, unless --db narrows it.
DATABASES: dict[str, Database] = parse_databases(DATABASES_FILE.read_text())


def select_databases(names: Iterable[str] | None = None) -> dict[str, Database]:
    if names is None:
        return {name: database for name, database in DATABASES.items() if not database.opt_in}

    selected = set(names)
    # File order, whatever order the names came in, so results files always list databases alike.
    return {name: database for name, database in DATABASES.items() if name in selected}
