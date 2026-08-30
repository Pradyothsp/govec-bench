from abc import ABC, abstractmethod

from autoregistry import Registry

from govec_bench.adapters.base import QueryResult


class LexicalIndex(Registry, ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> list[QueryResult]: ...
