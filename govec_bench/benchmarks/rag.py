from dataclasses import asdict, dataclass

from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.registry import build_adapters
from govec_bench.benchmarks.common import load_dataset
from govec_bench.chunkers.markdown_heading_chunker import MarkdownHeadingChunker
from govec_bench.chunkers.size_bound_chunker import SizeBoundChunker
from govec_bench.datasets.documents import load_chunks, to_insert_items
from govec_bench.embeddings.base import Embedder
from govec_bench.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder  # noqa: F401
from govec_bench.loaders.markdown_dir_loader import MarkdownDirLoader
from govec_bench.results import write_results
from govec_bench.types import Vector

TOP_K = 5

QUESTIONS = [
    "Why did scalar quantization recall collapse to near-random?",
    "What causes govec's batch insert to be slower than Chroma's?",
    "How does govec-python's client authenticate to the server?",
    "What is GOMEMLIMIT used for in this project?",
    "How do I enable metadata filtering in govec, and what's the default?",
    "Why does govec support both cosine and euclidean distance metrics?",
    "How does govec avoid holding the full vector dataset in RAM?",
    "What is the IDMapper and why does govec need it?",
    "What still needs to happen before govec-python can be published to PyPI?",
    "Does govec-bench use Pinecone for its vector database comparisons?",
    # Negative controls: no chunk in the corpus should score highly here --
    # multi-tenancy is explicitly out of scope per govec's AGENTS.md, and
    # Kubernetes/Helm is never mentioned anywhere in either repo.
    "How do I configure multi-tenant namespaces in govec?",
    "What Helm chart does govec-bench use to deploy to Kubernetes?",
]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    score: float
    source: str
    heading: str


def query_adapter(adapter: VectorDBAdapter, query_vector: Vector, k: int) -> list[RetrievedChunk]:
    results = adapter.query(query_vector, k=k)

    return [
        RetrievedChunk(
            score=result.score,
            source=(result.metadata or {}).get("source", ""),
            heading=(result.metadata or {}).get("heading", ""),
        )
        for result in results
    ]


def main() -> None:
    chunker = SizeBoundChunker(MarkdownHeadingChunker())
    chunks = load_chunks(MarkdownDirLoader(), chunker)

    embedder = Embedder["sentence-transformers"]()
    items = to_insert_items(chunks, embedder)

    adapters = build_adapters()
    for adapter in adapters.values():
        adapter.reset()
        load_dataset(adapter, items)

    results: dict[str, object] = {}
    for question in QUESTIONS:
        query_vector = embedder.embed_query(question)
        results[question] = {
            name: [asdict(chunk) for chunk in query_adapter(adapter, query_vector, TOP_K)]
            for name, adapter in adapters.items()
        }

    path = write_results(benchmark="rag", dataset="govec-docs", results=results)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
