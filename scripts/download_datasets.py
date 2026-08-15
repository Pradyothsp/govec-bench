#!/usr/bin/env python3
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path("data/raw")
CURL = shutil.which("curl")


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    url: str


DATASETS = [
    Dataset(name="siftsmall.tar.gz", url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"),
    Dataset(name="sift.tar.gz", url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"),
]


def download(dataset: Dataset) -> Path:
    dest = RAW_DIR / dataset.name
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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        archive = download(dataset)
        extract(archive)
    print(f"Done. Files in {RAW_DIR}/")


if __name__ == "__main__":
    main()
