from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter
from govec_bench.adapters.qdrant_adapter import QdrantAdapter


def build_adapters() -> dict[str, VectorDBAdapter]:
    return {
        "govec": GovecAdapter(),
        "chroma": ChromaAdapter(),
        "qdrant": QdrantAdapter(),
    }
