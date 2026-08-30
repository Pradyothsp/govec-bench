from abc import ABC, abstractmethod

from autoregistry import Registry

from govec_bench.adapters.base import QueryResult


class Reranker(Registry, ABC):
    @abstractmethod
    def rerank(self, question: str, results: list[QueryResult], k: int) -> list[QueryResult]: ...
