from dataclasses import dataclass

import streamlit as st

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.registry import build_adapters
from govec_bench.benchmarks.common import load_dataset
from govec_bench.chunkers.markdown_heading_chunker import MarkdownHeadingChunker
from govec_bench.chunkers.size_bound_chunker import SizeBoundChunker
from govec_bench.datasets.documents import load_chunks, to_insert_items
from govec_bench.embeddings.base import Embedder
from govec_bench.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder  # noqa: F401
from govec_bench.loaders.markdown_dir_loader import MarkdownDirLoader
from govec_bench.types import Vector

TOP_K_DEFAULT = 5

EXAMPLE_QUESTIONS = [
    "What causes govec's batch insert to be slower than Chroma's?",
    "Why did scalar quantization recall collapse to near-random?",
    "How do I enable metadata filtering in govec, and what's the default?",
    "How do I configure multi-tenant namespaces in govec?",
]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    score: float
    source: str
    heading: str
    text: str


@dataclass(frozen=True, slots=True)
class DatabaseResult:
    db_name: str
    chunks: list[RetrievedChunk]


@dataclass(frozen=True, slots=True)
class QueryResult:
    question: str
    results: list[DatabaseResult]
    answer: str | None = None


@st.cache_resource(show_spinner="Loading corpus and connecting to vector databases...")
def load_corpus() -> tuple[Embedder, dict[str, VectorDBAdapter]]:
    chunker = SizeBoundChunker(MarkdownHeadingChunker())
    chunks = load_chunks(MarkdownDirLoader(), chunker)

    embedder = Embedder["sentence-transformers"]()
    items = to_insert_items(chunks, embedder)

    adapters = {}
    for name, adapter in build_adapters().items():
        try:
            adapter.reset()
        except Exception:  # noqa: BLE001, S112 -- service just isn't reachable, skip it
            continue

        load_dataset(adapter, items)
        adapters[name] = adapter

    return embedder, adapters


def run_query(question: str, query_vector: Vector, adapters: dict[str, VectorDBAdapter], k: int) -> QueryResult:
    results = []

    for name, adapter in adapters.items():
        chunks = [
            RetrievedChunk(
                score=max(0.0, min(1.0, r.score)),
                source=(r.metadata or {}).get("source", ""),
                heading=(r.metadata or {}).get("heading", ""),
                text=(r.metadata or {}).get("text", ""),
            )
            for r in adapter.query(query_vector, k=k)
        ]
        results.append(DatabaseResult(db_name=name, chunks=chunks))

    return QueryResult(question=question, results=results)


def run_query_with_retry(
    question: str, embedder: Embedder, adapters: dict[str, VectorDBAdapter], k: int
) -> QueryResult:
    query_vector = embedder.embed_query(question)

    try:
        return run_query(question, query_vector, adapters, k)
    except Exception:  # noqa: BLE001 -- a cached DB connection may have been invalidated by an external reset (e.g. a benchmark script run elsewhere); reconnect once and retry
        load_corpus.clear()
        _, adapters = load_corpus()
        return run_query(question, query_vector, adapters, k)


def render_result(result: QueryResult) -> None:
    st.caption(
        "Scores often agree to 5-6 decimal places across databases -- at this corpus size (690 chunks), "
        "govec, Chroma, and Qdrant are computing the same cosine similarity on the same vectors, so their "
        "results converge this closely. See STATUS.md §25."
    )

    columns = st.columns(len(result.results))

    for column, db_result in zip(columns, result.results, strict=True):
        with column:
            st.markdown(f"**{db_result.db_name}**")

            for rank, chunk in enumerate(db_result.chunks, start=1):
                st.progress(chunk.score, text=f"{rank}. {chunk.heading}  ·  {chunk.score:.8f}")
                st.caption(chunk.source)
                with st.expander("text"):
                    st.write(chunk.text)


def main() -> None:
    st.set_page_config(page_title="govec-bench RAG demo", layout="wide")
    st.title("RAG retrieval demo")
    st.caption(
        "Ask a question against govec, govec-python, and govec-bench's own docs, "
        "and compare retrieval across live vector databases."
    )

    embedder, adapters = load_corpus()

    if not adapters:
        st.error("No vector databases are reachable. Run `task docker:up`, then reload this page.")
        st.stop()

    st.caption(f"Connected: {', '.join(adapters)}")

    st.write("Try:")
    example_columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(example_columns, EXAMPLE_QUESTIONS, strict=True):
        with column, st.container():
            if st.button(example, use_container_width=True):
                st.session_state.question = example

    question = st.text_input(
        "Question", key="question", placeholder="Ask something about govec, govec-python, or this benchmark suite..."
    )
    k = st.slider("Top k", min_value=1, max_value=10, value=TOP_K_DEFAULT)

    if not question:
        return

    result = run_query_with_retry(question, embedder, adapters, k)
    render_result(result)


if __name__ == "__main__":
    main()
