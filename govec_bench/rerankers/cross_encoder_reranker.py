from typing import cast, override

from sentence_transformers import CrossEncoder

from govec_bench.adapters.base import QueryResult
from govec_bench.rerankers.base import Reranker

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker, name="cross-encoder"):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = CrossEncoder(model_name)

    @override
    def rerank(self, question: str, results: list[QueryResult], k: int) -> list[QueryResult]:
        if not results:
            return []

        pairs = [(question, (r.metadata or {}).get("text", "")) for r in results]
        scores = cast(list[float], self._model.predict(pairs).tolist())

        reranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)

        return [QueryResult(id=r.id, score=s, metadata=r.metadata) for r, s in reranked[:k]]
