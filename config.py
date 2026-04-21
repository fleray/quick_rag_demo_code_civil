"""
config.py
─────────
Centralized configuration loader using Pydantic Settings.

Reads values from environment variables or a .env file.
Call `get_settings()` anywhere in the app to retrieve the singleton.

To add a new provider (e.g. Cohere, Mistral):
  1. Add the relevant API key field here.
  2. Swap the embedding / LLM classes in database.py and graph.py.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")

    # Model identifiers – swap these to change providers without touching logic
    llm_model: str = Field("gpt-4o-mini", description="Chat model name")
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model name"
    )

    # ── Couchbase connection ───────────────────────────────────────────────────
    couchbase_connection_string: str = Field(
        ..., description="Couchbase connection URI (couchbases://...)"
    )
    couchbase_username: str = Field(..., description="Couchbase username")
    couchbase_password: SecretStr = Field(..., description="Couchbase password")

    # ── Source data namespace ─────────────────────────────────────────────────
    couchbase_bucket: str = Field("code_civil")
    couchbase_scope: str = Field("livres_compiles")
    couchbase_collection: str = Field("law_articles")

    # ── Vector Search ─────────────────────────────────────────────────────────
    couchbase_index_name: str = Field("law_articles_vector_index")
    retriever_k: int = Field(5, description="Number of articles to retrieve per query")

    # ── Cache collections (same bucket / scope as source data) ────────────────
    couchbase_cache_collection: str = Field(
        "conv_cache", description="Collection for exact-match conversational cache"
    )
    couchbase_semantic_cache_collection: str = Field(
        "semantic_cache", description="Collection for semantic similarity cache"
    )
    couchbase_semantic_cache_index: str = Field(
        "semantic_cache_index",
        description="FTS vector index name on the semantic cache collection",
    )
    semantic_cache_score_threshold: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to consider a semantic cache hit",
    )

    # ── SDK timeouts ──────────────────────────────────────────────────────────
    couchbase_connect_timeout_seconds: int = Field(10)
    couchbase_kv_timeout_seconds: int = Field(10)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton (loaded once at startup)."""
    return Settings()  # type: ignore[call-arg]
