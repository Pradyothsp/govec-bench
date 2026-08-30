from typing import override

from rank_bm25 import BM25Okapi

from govec_bench.adapters.base import QueryResult
from govec_bench.datasets.documents import DocumentChunk
from govec_bench.retrieval.base import LexicalIndex


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index(LexicalIndex, name="bm25"):
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        corpus = [_tokenize(f"{chunk.heading}\n\n{chunk.text}") for chunk in chunks]
        self._bm25 = BM25Okapi(corpus)

    @override
    def search(self, query: str, k: int) -> list[QueryResult]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            QueryResult(
                id=self._chunks[i].id,
                score=float(scores[i]),
                metadata={
                    "source": self._chunks[i].source,
                    "heading": self._chunks[i].heading,
                    "text": self._chunks[i].text,
                },
            )
            for i in ranked
        ]
