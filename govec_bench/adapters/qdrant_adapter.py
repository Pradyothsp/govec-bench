import time
import uuid
from collections.abc import Mapping, Sequence
from typing import override

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, HnswConfigDiff, OptimizersConfigDiff, PointStruct, VectorParams

from govec_bench.adapters.base import InsertItem, QueryResult, Stats, VectorDBAdapter
from govec_bench.types import Vector

COLLECTION_NAME = "govec_bench"

# upsert(wait=True) only waits for the write to be durable -- Qdrant's HNSW
# indexing happens in a separate background optimizer job that can lag behind
# ingest. govec and Chroma are both fully synchronous (a vector is graph-
# connected by the time insert/batch_insert returns), so batch_insert polls
# here until Qdrant's indexer has actually caught up too -- otherwise recall/
# query benchmarks that load-then-immediately-query (no gap in between,
# unlike separate insert-latency vs recall benchmark runs) could measure
# against a partially-indexed graph. Real cost: Qdrant's reported insert
# latency reflects "durable and indexed," not just "durable" -- a fairer
# number for cross-adapter comparison, even if less flattering on its own.
_INDEXING_POLL_INTERVAL_S = 0.05
# 30s was fine at SIFT10K scale (STATUS.md §22) but too tight at SIFT100K --
# indexing catch-up cost grows with collection size, and memory.py's own
# DISK_STABILIZE_TIMEOUT_S (120s) already documents the same class of Qdrant
# background-merge lag at this scale.
_INDEXING_POLL_TIMEOUT_S = 120.0

# Qdrant point IDs must be a u64 or a UUID -- arbitrary strings like our
# "sift10k_12345" IDs are rejected outright. Map deterministically into this
# namespace instead of storing a random UUID, so re-inserting the same
# vector_id always maps to the same point (upsert semantics, matching every
# other adapter) -- and stash the original string in the payload so query()
# can hand back the ID callers actually asked for.
_ID_NAMESPACE = uuid.UUID("6f6e0a4e-9c1a-4b8e-8f0a-000000000000")
_ORIGINAL_ID_KEY = "_govec_bench_id"


def _to_point_id(vector_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, vector_id))


class QdrantAdapter(VectorDBAdapter):
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self._client = QdrantClient(url=f"http://{host}:{port}")
        # Qdrant needs the vector dimension at collection-creation time, unlike
        # Chroma's lazy collection -- create it on first insert instead, once
        # a real vector tells us the dimension. Avoids hardcoding a dataset's
        # dimension into the adapter or changing the VectorDBAdapter interface.
        self._collection_ready = False

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_ready:
            return
        if not self._client.collection_exists(COLLECTION_NAME):
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                # Qdrant's defaults skip HNSW indexing entirely below
                # ~10,000 KB of vector data (and does a full/brute-force scan
                # below full_scan_threshold points) -- for SIFT10K-scale
                # benchmarks that means no index ever gets built, silently
                # comparing govec/Chroma's always-on HNSW against Qdrant doing
                # a flat scan (verified live: indexed_vectors_count stayed 0
                # at Qdrant's stock defaults for a fresh SIFT10K load). Both
                # thresholds forced low so Qdrant always builds and uses the
                # real HNSW graph, matching the other two adapters --
                # m/ef_construct are left at Qdrant's own defaults, only
                # whether indexing happens at all is being forced here, not
                # how it's tuned.
                hnsw_config=HnswConfigDiff(full_scan_threshold=10),  # 10 is the API's minimum
                # indexing_threshold=0 does NOT mean "always index" -- it means
                # the opposite, disabling indexing entirely (verified against
                # Qdrant's docs after an initial attempt at 0 silently built no
                # index at all). 1 (KB) is effectively "index almost
                # immediately" without actually disabling it.
                optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
            )
        self._collection_ready = True

    def _to_point(self, item: InsertItem) -> PointStruct:
        payload: dict[str, object] = {_ORIGINAL_ID_KEY: item.id}
        if item.metadata:
            payload.update(item.metadata)
        return PointStruct(id=_to_point_id(item.id), vector=item.vector, payload=payload)

    @override
    def insert(self, vector_id: str, vector: Vector, metadata: dict[str, str] | None = None) -> None:
        # Deliberately doesn't wait for indexing (unlike batch_insert below):
        # this is only ever called in a tight one-at-a-time loop (insert.py's
        # single-insert benchmark, 10,000 sequential calls). Waiting for the
        # *whole collection's* indexed_vectors_count to catch up to
        # points_count after every single item is a fundamentally different,
        # ever-growing cost as the collection grows -- not a stable per-item
        # one -- and it isn't needed for correctness here: nothing else reads
        # this adapter's data back within the same benchmark run, unlike
        # batch_insert, which recall/query/memory benchmarks query
        # immediately after loading.
        item = InsertItem(id=vector_id, vector=vector, metadata=metadata)
        self._ensure_collection(len(vector))
        self._client.upsert(collection_name=COLLECTION_NAME, points=[self._to_point(item)], wait=True)

    @override
    def batch_insert(self, items: list[InsertItem]) -> None:
        if not items:
            return
        self._ensure_collection(len(items[0].vector))
        points: Sequence[PointStruct] = [self._to_point(item) for item in items]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        self._wait_for_indexing()

    def _wait_for_indexing(self) -> None:
        deadline = time.monotonic() + _INDEXING_POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                info = self._client.get_collection(COLLECTION_NAME)
            except ResponseHandlingException:
                # Transient: get_collection() can itself time out while
                # Qdrant's segment optimizer is heavily loaded mid-merge on a
                # large batch load (observed on a 100k load). Treat like "not
                # ready yet" and keep polling rather than failing the whole
                # load on one slow HTTP response.
                time.sleep(_INDEXING_POLL_INTERVAL_S)
                continue
            if info.indexed_vectors_count is not None and info.indexed_vectors_count >= (info.points_count or 0):
                return
            time.sleep(_INDEXING_POLL_INTERVAL_S)
        msg = f"Qdrant indexing did not catch up within {_INDEXING_POLL_TIMEOUT_S}s"
        raise TimeoutError(msg)

    @override
    def query(self, vector: Vector, k: int = 10) -> list[QueryResult]:
        result = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=k,
            with_payload=True,
        )

        return [
            QueryResult(
                id=_original_id(point.payload),
                score=point.score,
                metadata=_to_str_metadata(point.payload),
            )
            for point in result.points
        ]

    @override
    def stats(self) -> Stats:
        if not self._collection_ready:
            return Stats(count=0)
        info = self._client.get_collection(COLLECTION_NAME)
        return Stats(count=info.points_count or 0)

    @override
    def reset(self) -> None:
        if self._client.collection_exists(COLLECTION_NAME):
            self._client.delete_collection(COLLECTION_NAME)
        self._collection_ready = False


def _original_id(payload: Mapping[str, object] | None) -> str:
    if not payload or _ORIGINAL_ID_KEY not in payload:
        msg = "Qdrant point missing its original-ID payload field"
        raise RuntimeError(msg)
    return str(payload[_ORIGINAL_ID_KEY])


def _to_str_metadata(payload: Mapping[str, object] | None) -> dict[str, str] | None:
    if not payload:
        return None
    meta = {k: str(v) for k, v in payload.items() if k != _ORIGINAL_ID_KEY}
    return meta or None
