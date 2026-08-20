from abc import ABC, abstractmethod

from autoregistry import Registry

from govec_bench.types import Vector


class Embedder(Registry, ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[Vector]: ...

    @abstractmethod
    def embed_query(self, text: str) -> Vector: ...
