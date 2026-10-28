"""One in-memory vector store per chat session."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # Loaded once per process; Hugging Face downloads and caches the model on first use.
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


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
