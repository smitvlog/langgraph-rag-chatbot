"""Settings, read from the environment (a local .env file is loaded too)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Chunks below this cosine similarity are treated as unrelated to the question.
    min_score: float = float(os.getenv("MIN_SCORE", "0.3"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))


settings = Settings()
