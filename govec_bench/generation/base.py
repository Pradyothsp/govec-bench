from abc import ABC, abstractmethod

from autoregistry import Registry

from govec_bench.adapters.base import QueryResult


class Generator(Registry, ABC):
    @abstractmethod
    def generate(self, question: str, chunks: list[QueryResult]) -> str: ...
