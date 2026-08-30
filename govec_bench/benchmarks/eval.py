from collections.abc import Callable

import sqlite_utils

from govec_bench.adapters.base import QueryResult
from govec_bench.adapters.registry import build_adapters
from govec_bench.benchmarks.common import load_dataset
from govec_bench.chunkers.markdown_heading_chunker import MarkdownHeadingChunker
from govec_bench.chunkers.size_bound_chunker import SizeBoundChunker
from govec_bench.datasets.documents import load_chunks, to_insert_items
from govec_bench.embeddings.base import Embedder
from govec_bench.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder  # noqa: F401
from govec_bench.eval.questions import QUESTIONS
from govec_bench.eval.store import (
    DB_PATH,
    EvalQuestion,
    EvalResult,
    ensure_schema,
    get_db,
    insert_questions,
    insert_results,
    insert_run,
    recall_at_k,
)
from govec_bench.loaders.markdown_dir_loader import MarkdownDirLoader
from govec_bench.rerankers.cross_encoder_reranker import CrossEncoderReranker
from govec_bench.retrieval.bm25_index import BM25Index
from govec_bench.retrieval.fusion import fuse

TOP_K = 5
CANDIDATE_K = 20
DENSE_ONLY_STAGE = "dense_only"
HYBRID_STAGE = "hybrid"
HYBRID_RERANK_STAGE = "hybrid_rerank"
ADAPTER_NAME = "govec"


def to_retrieved(results: list[QueryResult]) -> list[dict[str, object]]:
    return [
        {
            "score": r.score,
            "source": (r.metadata or {}).get("source", ""),
            "heading": (r.metadata or {}).get("heading", ""),
        }
        for r in results
    ]


def _matches(chunk: dict[str, object], source: str, heading: str | None) -> bool:
    if chunk["source"] != source:
        return False
    return heading is None or heading in str(chunk["heading"])


def score(retrieved: list[dict[str, object]], question: EvalQuestion) -> tuple[bool, int | None]:
    if question.expected_source == "NONE":
        return False, None

    acceptable = [(question.expected_source, question.expected_heading), *question.alternates]

    for rank, chunk in enumerate(retrieved, start=1):
        if any(_matches(chunk, source, heading) for source, heading in acceptable):
            return True, rank

    return False, None


def run_stage(
    db: sqlite_utils.Database,
    stage: str,
    questions: list[EvalQuestion],
    retrieve: Callable[[str], list[QueryResult]],
) -> None:
    run_id = insert_run(db, stage=stage)

    results = []
    for question in questions:
        retrieved = to_retrieved(retrieve(question.question))
        hit, rank = score(retrieved, question)
        results.append(
            EvalResult(run_id=run_id, question_id=question.question_id, retrieved=retrieved, hit=hit, rank=rank)
        )

    insert_results(db, results)

    misses = [r.question_id for r in results if not r.hit]
    print(f"{stage}: recall@{TOP_K} = {recall_at_k(db, stage):.1%}  (run_id={run_id}, {len(misses)} misses)")
    print(f"  misses: {misses}")


def main() -> None:
    chunker = SizeBoundChunker(MarkdownHeadingChunker())
    chunks = load_chunks(MarkdownDirLoader(), chunker)

    embedder = Embedder["sentence-transformers"]()
    items = to_insert_items(chunks, embedder)

    adapter = build_adapters()[ADAPTER_NAME]
    adapter.reset()
    load_dataset(adapter, items)

    lexical_index = BM25Index(chunks)
    reranker = CrossEncoderReranker()

    db = get_db()
    ensure_schema(db)
    insert_questions(db, QUESTIONS)

    def dense_retrieve(question_text: str) -> list[QueryResult]:
        return adapter.query(embedder.embed_query(question_text), k=TOP_K)

    def fused_candidates(question_text: str) -> list[QueryResult]:
        query_vector = embedder.embed_query(question_text)
        dense_results = adapter.query(query_vector, k=CANDIDATE_K)
        lexical_results = lexical_index.search(question_text, k=CANDIDATE_K)
        return fuse(dense_results, lexical_results)

    def hybrid_retrieve(question_text: str) -> list[QueryResult]:
        return fused_candidates(question_text)[:TOP_K]

    def hybrid_rerank_retrieve(question_text: str) -> list[QueryResult]:
        return reranker.rerank(question_text, fused_candidates(question_text), k=TOP_K)

    print(f"Using {DB_PATH} (adapter={ADAPTER_NAME})")
    run_stage(db, DENSE_ONLY_STAGE, QUESTIONS, dense_retrieve)
    run_stage(db, HYBRID_STAGE, QUESTIONS, hybrid_retrieve)
    run_stage(db, HYBRID_RERANK_STAGE, QUESTIONS, hybrid_rerank_retrieve)


if __name__ == "__main__":
    main()
