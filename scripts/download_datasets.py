#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from govec_bench.datasets.dbpedia import SHARDS as DBPEDIA_SHARDS

RAW_DIR = Path("data/raw")
CURL = shutil.which("curl")


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    url: str


DBPEDIA_URL = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data"

DATASETS: dict[str, list[Dataset]] = {
    "sift": [
        Dataset(name="siftsmall.tar.gz", url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"),
        Dataset(name="sift.tar.gz", url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"),
    ],
    # The shards the loader reads: about 1.1 GB, enough for the 100k base set plus queries.
    "dbpedia": [Dataset(name=f"dbpedia/{shard}", url=f"{DBPEDIA_URL}/{shard}") for shard in DBPEDIA_SHARDS],
}


def download(dataset: Dataset) -> Path:
    dest = RAW_DIR / dataset.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset.name}...")
    if CURL is None:
        msg = "curl is required to download datasets but was not found on PATH"
        raise RuntimeError(msg)
    # curl (not urllib) because the FTP server's passive-mode data connection only
    # works over IPv6 -- urllib gives no way to force that, curl negotiates it correctly.
    subprocess.run(  # noqa: S603 -- url comes from the hardcoded DATASETS list above, not user input
        [CURL, "-fsSL", "-o", str(dest), dataset.url],
        check=True,
    )
    return dest


def extract(archive: Path) -> None:
    print(f"Extracting {archive.name}...")
    with tarfile.open(archive, mode="r:gz") as tar:
        tar.extractall(RAW_DIR, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="sift")
    args = parser.parse_args()

    for dataset in DATASETS[args.dataset]:
        path = download(dataset)
        if path.name.endswith(".tar.gz"):
            extract(path)
    print(f"Done. Files in {RAW_DIR}/")


if __name__ == "__main__":
    main()
