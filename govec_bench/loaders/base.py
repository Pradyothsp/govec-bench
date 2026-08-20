from abc import ABC, abstractmethod
from typing import NamedTuple

from autoregistry import Registry


class RawDocument(NamedTuple):
    source: str
    text: str


class Loader(Registry, ABC):
    @abstractmethod
    def load(self) -> list[RawDocument]: ...
