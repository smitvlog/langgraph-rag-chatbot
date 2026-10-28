"""One in-memory vector store per chat session."""

from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .config import settings


class FastEmbedEmbeddings(Embeddings):
    """ONNX embeddings via fastembed. No PyTorch, so the app fits on a 512 MB server."""

    # fastembed's default batch of 256 chunks needs about 1 GB of RAM for a long document.
    BATCH_SIZE = 16

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts, batch_size=self.BATCH_SIZE)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    # Loaded once per process; the model is downloaded into the cache dir on first use.
    return FastEmbedEmbeddings(settings.embedding_model, settings.embedding_cache_dir)


class SessionStore:
    """Keeps each session's chunks apart, so users only ever search their own documents."""

    def __init__(self) -> None:
        self._stores: dict[str, InMemoryVectorStore] = {}
        self._files: dict[str, dict[str, int]] = {}

    def add(self, session_id: str, filename: str, chunks: list[Document]) -> None:
        if session_id not in self._stores:
            self._stores[session_id] = InMemoryVectorStore(get_embeddings())
        self._stores[session_id].add_documents(chunks)
        files = self._files.setdefault(session_id, {})
        files[filename] = files.get(filename, 0) + len(chunks)

    def search(self, session_id: str, query: str, k: int) -> list[tuple[Document, float]]:
        store = self._stores.get(session_id)
        return store.similarity_search_with_score(query, k=k) if store else []

    def files(self, session_id: str) -> dict[str, int]:
        return dict(self._files.get(session_id, {}))

    def clear(self, session_id: str) -> None:
        self._stores.pop(session_id, None)
        self._files.pop(session_id, None)
