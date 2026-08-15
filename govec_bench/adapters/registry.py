from govec_bench.adapters.base import VectorDBAdapter
from govec_bench.adapters.chroma_adapter import ChromaAdapter
from govec_bench.adapters.govec_adapter import GovecAdapter


def build_adapters() -> dict[str, VectorDBAdapter]:
    return {
        "govec": GovecAdapter(),
        "chroma": ChromaAdapter(),
    }
