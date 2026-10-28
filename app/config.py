"""Settings, read from the environment (a local .env file is loaded too)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Inside the project, so a model downloaded at build time is still there at run time.
    embedding_cache_dir: str = os.getenv(
        "EMBEDDING_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "fastembed")
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Chunks given to the model for "summarise this" style questions.
    summary_chunks: int = int(os.getenv("SUMMARY_CHUNKS", "12"))
    # If even the best chunk scores below this cosine similarity, the question is unanswered.
    min_score: float = float(os.getenv("MIN_SCORE", "0.3"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    max_chunks_per_session: int = int(os.getenv("MAX_CHUNKS_PER_SESSION", "5000"))
    # Sessions live in memory, so idle ones are dropped and their number is capped.
    session_ttl_minutes: int = int(os.getenv("SESSION_TTL_MINUTES", "60"))
    max_sessions: int = int(os.getenv("MAX_SESSIONS", "100"))


settings = Settings()
