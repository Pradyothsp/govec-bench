import struct
from array import array
from pathlib import Path
from typing import NamedTuple

import numpy as np

from govec_bench.datasets.base import ArrayItems, VectorDataset
from govec_bench.types import NeighborIndices, Vector

RAW_DIR = Path("data/raw")


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


def load_sift10k() -> VectorDataset:
    d = RAW_DIR / "siftsmall"
    base = read_fvecs(d / "siftsmall_base.fvecs")

    # SIFT's shipped siftsmall_groundtruth.ivecs isn't loaded: it's Euclidean, and recall is
    # graded against exact cosine neighbours instead (see datasets/groundtruth.py).
    return VectorDataset(
        name="sift10k",
        base=ArrayItems("sift10k", np.asarray(base, dtype=np.float32)),
        queries=read_fvecs(d / "siftsmall_query.fvecs"),
    )


def load_sift100k() -> VectorDataset:
    d = RAW_DIR / "sift"
    base = read_fvecs(d / "sift_base.fvecs", limit=100_000)

    return VectorDataset(
        name="sift100k",
        base=ArrayItems("sift100k", np.asarray(base, dtype=np.float32)),
        queries=read_fvecs(d / "sift_query.fvecs"),
    )
