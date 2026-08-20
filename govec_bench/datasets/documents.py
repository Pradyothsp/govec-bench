from dataclasses import dataclass

from govec_bench.adapters.base import InsertItem
from govec_bench.chunkers.base import Chunker
from govec_bench.embeddings.base import Embedder
from govec_bench.loaders.base import Loader


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    heading: str


def load_chunks(loader: Loader, chunker: Chunker) -> list[DocumentChunk]:
    chunks = []

    for document in loader.load():
        for i, section in enumerate(chunker.chunk(document.text)):
            chunks.append(
                DocumentChunk(
                    id=f"{document.source}_{i}", text=section.body, source=document.source, heading=section.heading
                )
            )

    return chunks


def to_insert_items(chunks: list[DocumentChunk], embedder: Embedder) -> list[InsertItem]:
    texts = [f"{chunk.heading}\n\n{chunk.text}" for chunk in chunks]
    vectors = embedder.embed_texts(texts)

    return [
        InsertItem(
            id=chunk.id,
            vector=vector,
            metadata={"source": chunk.source, "heading": chunk.heading, "text": chunk.text},
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
