import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from govec_bench.adapters.base import InsertItem
from govec_bench.types import NeighborIndices, Vector

RAW_DIR = Path("data/raw")


@dataclass(frozen=True, slots=True)
class SiftDataset:
    base: list[InsertItem]
    queries: list[Vector]
    groundtruth: list[NeighborIndices] | None


class _RawVecsData(NamedTuple):
    data: bytes
    dim: int


def _read_vecs_bytes(path: Path, limit: int | None) -> _RawVecsData:
    with path.open("rb") as f:
        header = f.read(4)
        if not header:
            return _RawVecsData(data=b"", dim=0)

        dim = struct.unpack("<i", header)[0]
        f.seek(0)

        record_size = 4 + dim * 4
        data = f.read(record_size * limit) if limit is not None else f.read()

    return _RawVecsData(data=data, dim=dim)


def read_fvecs(path: Path, limit: int | None = None) -> list[Vector]:
    data, dim = _read_vecs_bytes(path, limit)

    values: array[float] = array("f")
    values.frombytes(data)

    stride = dim + 1
    record_count = len(values) // stride

    return [values[i * stride + 1 : i * stride + 1 + dim].tolist() for i in range(record_count)]


def read_ivecs(path: Path, limit: int | None = None) -> list[NeighborIndices]:
    data, dim = _read_vecs_bytes(path, limit)

    values: array[int] = array("i")
    values.frombytes(data)

    stride = dim + 1
    record_count = len(values) // stride

    return [values[i * stride + 1 : i * stride + 1 + dim].tolist() for i in range(record_count)]


def load_sift10k() -> SiftDataset:
    d = RAW_DIR / "siftsmall"
    base = read_fvecs(d / "siftsmall_base.fvecs")

    return SiftDataset(
        base=[InsertItem(id=f"sift10k_{i}", vector=v) for i, v in enumerate(base)],
        queries=read_fvecs(d / "siftsmall_query.fvecs"),
        groundtruth=read_ivecs(d / "siftsmall_groundtruth.ivecs"),
    )


def load_sift100k() -> SiftDataset:
    d = RAW_DIR / "sift"
    base = read_fvecs(d / "sift_base.fvecs", limit=100_000)

    return SiftDataset(
        base=[InsertItem(id=f"sift100k_{i}", vector=v) for i, v in enumerate(base)],
        queries=read_fvecs(d / "sift_query.fvecs"),
        # sift_groundtruth.ivecs is computed against the full 1M-vector base set,
        # not valid for this 100k slice -- SIFT100K is latency/memory only, no recall.
        groundtruth=None,
    )
