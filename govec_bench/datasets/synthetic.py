import struct
from array import array
from dataclasses import dataclass
from pathlib import Path

from govec_bench.adapters.base import InsertItem

RAW_DIR = Path("data/raw")


@dataclass(frozen=True, slots=True)
class SiftDataset:
    base: list[InsertItem]
    queries: list[list[float]]
    groundtruth: list[list[int]] | None


def _read_vecs_bytes(path: Path, limit: int | None) -> tuple[bytes, int]:
    with path.open("rb") as f:
        header = f.read(4)
        if not header:
            return b"", 0
        dim = struct.unpack("<i", header)[0]
        f.seek(0)
        record_size = 4 + dim * 4
        data = f.read(record_size * limit) if limit is not None else f.read()
    return data, dim


def read_fvecs(path: Path, limit: int | None = None) -> list[list[float]]:
    data, dim = _read_vecs_bytes(path, limit)
    values: array[float] = array("f")
    values.frombytes(data)
    stride = dim + 1
    record_count = len(values) // stride
    return [values[i * stride + 1 : i * stride + 1 + dim].tolist() for i in range(record_count)]


def read_ivecs(path: Path, limit: int | None = None) -> list[list[int]]:
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
