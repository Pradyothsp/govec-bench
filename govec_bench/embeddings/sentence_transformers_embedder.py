from typing import cast, override

from sentence_transformers import SentenceTransformer

from govec_bench.embeddings.base import Embedder
from govec_bench.types import Vector

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformersEmbedder(Embedder, name="sentence-transformers"):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    @override
    def embed_texts(self, texts: list[str]) -> list[Vector]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return cast(list[Vector], embeddings.tolist())

    @override
    def embed_query(self, text: str) -> Vector:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return cast(Vector, embedding.tolist())
