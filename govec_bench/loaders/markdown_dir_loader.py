from pathlib import Path
from typing import override

from govec_bench.loaders.base import Loader, RawDocument

DEFAULT_DOCS_DIR = Path("data/docs")


class MarkdownDirLoader(Loader, name="markdown-dir"):
    def __init__(self, docs_dir: Path = DEFAULT_DOCS_DIR) -> None:
        self._docs_dir = docs_dir

    @override
    def load(self) -> list[RawDocument]:
        documents = []

        for path in sorted(self._docs_dir.rglob("*.md")):
            source = path.relative_to(self._docs_dir).with_suffix("").as_posix().replace("/", "_")
            documents.append(RawDocument(source=source, text=path.read_text()))

        return documents
