from typing import override

from govec import GoVecClient
from govec.models import InsertRequest

from govec_bench.adapters.base import InsertItem, QueryResult, Stats, VectorDBAdapter
from govec_bench.types import Vector


class GovecAdapter(VectorDBAdapter):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str = "",
        *,
        tls: bool = False,
    ) -> None:
        self._client = GoVecClient(host=host, port=port, api_key=api_key, protocol="rest", tls=tls)

    @override
    def insert(self, vector_id: str, vector: Vector, metadata: dict[str, str] | None = None) -> None:
        self._client.insert(vector_id=vector_id, dense_vector=vector, metadata=metadata)

    @override
    def batch_insert(self, items: list[InsertItem]) -> None:
        requests = [InsertRequest(id=item.id, vector=item.vector, metadata=item.metadata) for item in items]

        self._client.insert_many(requests)

    @override
    def query(self, vector: Vector, k: int = 10) -> list[QueryResult]:
        results = self._client.search(dense_vector=vector, k=k)

        return [QueryResult(id=r.id, score=r.score, metadata=r.meta) for r in results]

    @override
    def stats(self) -> Stats:
        info = self._client.info()
        stats = self._client.get_stats()

        return Stats(
            count=stats.vector_count,
            extra={
                "index_type": info.index_type,
                "quantization": info.quantization,
                "dimensions": info.dimensions,
                "enable_mmap": info.enable_mmap,
            },
        )

    @override
    def reset(self) -> None:
        self._client.reset()

    def flush(self) -> None:
        # govec-specific: compacts the WAL into an on-disk snapshot
        # (POST /api/v1/admin/flush). Not part of VectorDBAdapter -- Chroma
        # persists incrementally on its own and has no equivalent call.
        self._client.flush()
