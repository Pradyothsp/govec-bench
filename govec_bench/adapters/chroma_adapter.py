from collections.abc import Mapping, Sequence
from typing import override

import chromadb
from chromadb.api.collection_configuration import HNSWConfiguration
from chromadb.api.types import CollectionMetadata, Metadatas

from govec_bench.adapters.base import InsertItem, QueryResult, Stats, VectorDBAdapter
from govec_bench.types import Vector

COLLECTION_NAME = "govec_bench"


def _to_str_metadata(meta: Mapping[str, object] | None) -> dict[str, str] | None:
    # Chroma metadata values can be richer than our dict[str, str] contract
    # (float, list, ...); stringify them to keep QueryResult.metadata's type.
    if not meta:
        return None

    return {k: str(v) for k, v in meta.items()}


class ChromaAdapter(VectorDBAdapter):
    def __init__(self, host: str = "localhost", port: int = 8001) -> None:
        self._client = chromadb.HttpClient(host=host, port=port)
        # None leaves ef_search at Chroma's default; recreate_with_search_ef() sets it.
        self._search_ef: int | None = None
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME, metadata=self._metadata())

    def _metadata(self) -> CollectionMetadata:
        metadata: CollectionMetadata = {"hnsw:space": "cosine"}
        if self._search_ef is not None:
            metadata["hnsw:search_ef"] = self._search_ef

        return metadata

    @override
    def insert(self, vector_id: str, vector: Vector, metadata: dict[str, str] | None = None) -> None:
        embeddings: list[Sequence[float]] = [vector]
        self._collection.upsert(ids=[vector_id], embeddings=embeddings, metadatas=[metadata] if metadata else None)

    @override
    def batch_insert(self, items: list[InsertItem]) -> None:
        ids = [item.id for item in items]
        embeddings: list[Sequence[float]] = [item.vector for item in items]
        # Chroma rejects a metadatas list containing None entries -- only pass
        # it through when at least one item actually has metadata, coercing
        # any individual None to an empty dict for the mixed case.
        metadatas: Metadatas | None = (
            [item.metadata or {} for item in items] if any(item.metadata for item in items) else None
        )

        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    @override
    def query(self, vector: Vector, k: int = 10) -> list[QueryResult]:
        result = self._collection.query(query_embeddings=[vector], n_results=k)

        ids = result["ids"][0]
        distances = result["distances"]
        metadatas = result["metadatas"]
        if distances is None or metadatas is None:
            msg = "Chroma query response missing distances/metadatas"
            raise RuntimeError(msg)

        return [
            # Collection is configured for cosine space, so distance -> similarity
            # matches govec's convention: score = 1 - distance.
            QueryResult(id=id_, score=1 - dist, metadata=_to_str_metadata(meta))
            for id_, dist, meta in zip(ids, distances[0], metadatas[0], strict=True)
        ]

    @override
    def stats(self) -> Stats:
        return Stats(count=self._collection.count())

    def recreate_with_search_ef(self, ef: int) -> None:
        # Empties the collection and recreates it with this ef_search; load after calling. Set
        # at creation because Chroma 1.4.4 accepts collection.modify(ef_search) and reports it
        # applied, but queries keep the ef the collection was created with. Read back from the
        # server rather than trusted, so a setting it drops can't pass silently.
        self._search_ef = ef
        self.reset()

        applied = self.hnsw_configuration()
        if applied.get("ef_search") != ef:
            msg = f"Chroma kept hnsw configuration {applied} after setting ef_search={ef}"
            raise RuntimeError(msg)

    def hnsw_configuration(self) -> HNSWConfiguration:
        # As the server reports it now, not as this client last set it.
        return self._client.get_collection(COLLECTION_NAME).configuration.get("hnsw") or {}

    @override
    def reset(self) -> None:
        self._client.reset()
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME, metadata=self._metadata())
