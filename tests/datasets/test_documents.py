from typing import override

from govec_bench.chunkers.base import Chunker, Section
from govec_bench.datasets.documents import DocumentChunk, load_chunks, to_insert_items
from govec_bench.embeddings.base import Embedder
from govec_bench.loaders.base import Loader, RawDocument
from govec_bench.types import Vector


class _FakeLoader(Loader, skip=True):
    @override
    def load(self) -> list[RawDocument]:
        return [RawDocument(source="a", text="doc a"), RawDocument(source="b", text="doc b")]


class _FakeChunker(Chunker, skip=True):
    @override
    def chunk(self, text: str) -> list[Section]:
        return [Section("first", f"{text} one"), Section("second", f"{text} two")]


def test_load_chunks__loader_and_chunker__returns_one_document_chunk_per_section() -> None:
    # Arrange

    # Act
    result = load_chunks(_FakeLoader(), _FakeChunker())

    # Assert
    assert result == [
        DocumentChunk(id="a_0", text="doc a one", source="a", heading="first"),
        DocumentChunk(id="a_1", text="doc a two", source="a", heading="second"),
        DocumentChunk(id="b_0", text="doc b one", source="b", heading="first"),
        DocumentChunk(id="b_1", text="doc b two", source="b", heading="second"),
    ]


class _FakeEmbedder(Embedder, skip=True):
    @override
    def embed_texts(self, texts: list[str]) -> list[Vector]:
        return [[float(len(text))] for text in texts]

    @override
    def embed_query(self, text: str) -> Vector:
        return [float(len(text))]


def test_to_insert_items__chunks__embeds_and_attaches_source_and_heading_metadata() -> None:
    # Arrange
    chunks = [
        DocumentChunk(id="a_0", text="Body one.", source="a", heading="One"),
        DocumentChunk(id="a_1", text="Body two.", source="a", heading="Two"),
    ]

    # Act
    result = to_insert_items(chunks, _FakeEmbedder())

    # Assert
    assert [item.id for item in result] == ["a_0", "a_1"]
    assert result[0].metadata == {"source": "a", "heading": "One", "text": "Body one."}
    assert result[0].vector == [float(len("One\n\nBody one."))]
