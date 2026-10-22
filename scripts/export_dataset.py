#!/usr/bin/env python3
"""Write a dataset to files GoVec's own Go benchmarks read, so they run on the same vectors.

The output directory holds three files in the SIFT corpus formats:

    base.fvecs         the vectors to index, in order: row i is stored as "v{i}"
    query.fvecs        the queries, none of which is in the base set
    groundtruth.ivecs  each query's exact top-k base rows by cosine similarity, nearest first

Both formats are one record per row: a little-endian int32 count, then that many float32
(fvecs) or int32 (ivecs) values.
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from govec_bench.datasets.groundtruth import exact_cosine_neighbors
from govec_bench.datasets.registry import DATASETS, SIZES, load_sized

EXPORT_DIR = Path("data/export")

# Deeper than the benchmarks' top-10, so recall at a larger k needs no new export.
TRUTH_K = 100


def write_fvecs(path: Path, rows: npt.ArrayLike) -> None:
    _write_vecs(path, np.asarray(rows, dtype="<f4"))


def write_ivecs(path: Path, rows: npt.ArrayLike) -> None:
    _write_vecs(path, np.asarray(rows, dtype="<i4"))


def _write_vecs(path: Path, values: npt.NDArray[np.float32] | npt.NDArray[np.int32]) -> None:
    # Each record's count prefix is an int32; viewing it as the values' 4-byte type lets one
    # array hold whole records and be written in a single call.
    counts = np.full((len(values), 1), values.shape[1], dtype="<i4").view(values.dtype)

    np.hstack([counts, values]).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dbpedia")
    parser.add_argument("--size", choices=SIZES, default="small")
    args = parser.parse_args()

    dataset = load_sized(DATASETS[args.dataset], args.size)
    out = EXPORT_DIR / dataset.name
    out.mkdir(parents=True, exist_ok=True)

    base = dataset.base.vectors
    print(f"Computing exact top-{TRUTH_K} for {len(dataset.queries)} queries over {len(base)} vectors...")
    truth = exact_cosine_neighbors(base, dataset.queries, TRUTH_K)

    write_fvecs(out / "base.fvecs", base)
    write_fvecs(out / "query.fvecs", dataset.queries)
    write_ivecs(out / "groundtruth.ivecs", truth)
    print(f"Done. Files in {out}/")


if __name__ == "__main__":
    main()
