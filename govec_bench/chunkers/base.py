from abc import ABC, abstractmethod
from typing import NamedTuple

from autoregistry import Registry


class Section(NamedTuple):
    heading: str
    body: str


class Chunker(Registry, ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Section]: ...
