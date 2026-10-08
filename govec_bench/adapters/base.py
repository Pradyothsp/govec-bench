from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from govec_bench.types import Vector

# Every database searches with this ef unless a benchmark sets another: Chroma's and Qdrant's
# adapters pass it explicitly rather than rely on their defaults matching. GoVec's comes from
# hnsw_ef_search in govec-config*.yaml, which must match it.
DEFAULT_SEARCH_EF = 100


@dataclass(frozen=True, slots=True)
class InsertItem:
    id: str
    vector: Vector
    metadata: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class QueryResult:
    id: str
    score: float
    metadata: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Stats:
    count: int
    memory_bytes: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


class VectorDBAdapter(ABC):
    @abstractmethod
    def insert(self, vector_id: str, vector: Vector, metadata: dict[str, str] | None = None) -> None: ...

    @abstractmethod
    def batch_insert(self, items: list[InsertItem]) -> None: ...

    @abstractmethod
    def query(self, vector: Vector, k: int = 10) -> list[QueryResult]: ...

    @abstractmethod
    def stats(self) -> Stats: ...

    @abstractmethod
    def reset(self) -> None: ...

    def wait_until_indexed(self) -> None:
        # Returns once every vector inserted so far is in the search index. Called once at the
        # end of a load, inside the insert benchmark's timing, so a database that indexes in the
        # background pays for it there. GoVec and Chroma index before an insert returns.
        return

    def wait_until_settled(self) -> None:
        # Called once after a full load, before anything is measured against it. A database that
        # keeps reorganizing data in the background after its inserts return waits for that here.
        return
