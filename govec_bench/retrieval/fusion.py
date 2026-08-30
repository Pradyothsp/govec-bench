from govec_bench.adapters.base import QueryResult

RRF_K = 60


def fuse(dense: list[QueryResult], lexical: list[QueryResult], k: int = RRF_K) -> list[QueryResult]:
    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, str] | None] = {}

    for results in (dense, lexical):
        for rank, result in enumerate(results, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
            metadata.setdefault(result.id, result.metadata)

    ranked_ids = sorted(scores, key=lambda result_id: scores[result_id], reverse=True)

    return [QueryResult(id=result_id, score=scores[result_id], metadata=metadata[result_id]) for result_id in ranked_ids]
