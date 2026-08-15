from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from govec_bench.types import Vector


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
