from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload, override

import numpy as np
import numpy.typing as npt

from govec_bench.adapters.base import InsertItem
from govec_bench.types import Vector


class ArrayItems(Sequence[InsertItem]):
    """Base vectors held as one float32 array, turned into InsertItems only when read.

    100k text embeddings at 1536 dimensions are 154M floats: about 600 MB as an array, but
    roughly 5 GB as Python lists of floats. Benchmarks read items outside their timed
    sections, so building each one on demand costs no measured time.
    """

    def __init__(self, id_prefix: str, vectors: npt.NDArray[np.float32]) -> None:
        self._id_prefix = id_prefix
        self.vectors = vectors

    def id_at(self, index: int) -> str:
        return f"{self._id_prefix}_{index}"

    @overload
    def __getitem__(self, index: int) -> InsertItem: ...

    @overload
    def __getitem__(self, index: slice) -> list[InsertItem]: ...

    @override
    def __getitem__(self, index: int | slice) -> InsertItem | list[InsertItem]:
        if isinstance(index, slice):
            return [self._item(i) for i in range(*index.indices(len(self)))]

        if not -len(self) <= index < len(self):
            msg = f"index {index} out of range for {len(self)} items"
            raise IndexError(msg)
        return self._item(index % len(self))

    @override
    def __len__(self) -> int:
        return len(self.vectors)

    @override
    def __iter__(self) -> Iterator[InsertItem]:
        return (self._item(i) for i in range(len(self)))

    def _item(self, index: int) -> InsertItem:
        return InsertItem(id=self.id_at(index), vector=self.vectors[index].tolist())


@dataclass(frozen=True, slots=True)
class VectorDataset:
    name: str
    base: ArrayItems
    queries: list[Vector]
