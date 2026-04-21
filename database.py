"""
database.py
───────────
Couchbase Capella connection layer.

All objects are cached as Streamlit resource singletons so the expensive
Cluster handshake and index warm-up happen only once per app session.

To swap the vector store backend:
  • Replace `CouchbaseSearchVectorStore` with your preferred LangChain
    VectorStore and update the corresponding FTS index.

To swap the embedding model:
  • Change `embedding_model` in your .env (or config.py) and rebuild the cache.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import streamlit as st
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from langchain_couchbase.cache import CouchbaseCache, CouchbaseSemanticCache
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import OpenAIEmbeddings

from config import Settings, get_settings


# ─────────────────────────────────────────────────────────────────────────────
# Cluster singleton
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔗 Connexion à Couchbase Capella…")
def get_cluster() -> tuple[Cluster, bool, str]:
    """
    Establish and return an authenticated Couchbase Cluster connection.

    Returns
    -------
    (cluster, ok, message)
        cluster  – the connected Cluster object (or None on failure)
        ok       – True if the connection succeeded
        message  – human-readable status string
    """
    cfg: Settings = get_settings()
    try:
        auth = PasswordAuthenticator(
            cfg.couchbase_username,
            cfg.couchbase_password.get_secret_value(),
        )
        timeout_opts = ClusterTimeoutOptions(
            connect_timeout=timedelta(seconds=cfg.couchbase_connect_timeout_seconds),
            kv_timeout=timedelta(seconds=cfg.couchbase_kv_timeout_seconds),
        )
        cluster = Cluster(
            cfg.couchbase_connection_string,
            ClusterOptions(auth, timeout_options=timeout_opts),
        )
        cluster.wait_until_ready(
            timedelta(seconds=cfg.couchbase_connect_timeout_seconds)
        )
        return cluster, True, "Connecté à Couchbase Capella"
    except Exception as exc:  # noqa: BLE001
        return None, False, f"Erreur de connexion : {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings singleton
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    """Return the cached OpenAI embeddings instance."""
    cfg = get_settings()
    return OpenAIEmbeddings(
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key.get_secret_value(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vector store singleton
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="📚 Chargement de la base vectorielle…")
def get_vector_store() -> Optional[CouchbaseSearchVectorStore]:
    """
    Create the CouchbaseSearchVectorStore pointed at law_articles.

    Key field mapping (matches the ingestion schema):
      • text_key      = "content"   (the article text)
      • embedding_key = "vec"       (the pre-computed embedding vector)

    Returns None if the Couchbase connection is unavailable.
    """
    cluster, ok, _ = get_cluster()
    if not ok or cluster is None:
        return None

    cfg = get_settings()
    return CouchbaseSearchVectorStore(
        cluster=cluster,
        bucket_name=cfg.couchbase_bucket,
        scope_name=cfg.couchbase_scope,
        collection_name=cfg.couchbase_collection,
        embedding=get_embeddings(),
        index_name=cfg.couchbase_index_name,
        text_key="content",     # Matches your stored field 'content'
        embedding_key="vec",   # Matches your stored field 'vec'
        scoped_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cache objects  (not cached with @st.cache_resource so they can be toggled)
# ─────────────────────────────────────────────────────────────────────────────

def get_conversational_cache() -> Optional[CouchbaseCache]:
    """
    Return a CouchbaseCache instance for exact-match LLM caching.

    Uses a dedicated collection (conv_cache) in the same scope as the
    law_articles data so no cross-bucket credentials are needed.
    """
    cluster, ok, _ = get_cluster()
    if not ok or cluster is None:
        return None

    cfg = get_settings()
    return CouchbaseCache(
        cluster=cluster,
        bucket_name=cfg.couchbase_bucket,
        scope_name=cfg.couchbase_scope,
        collection_name=cfg.couchbase_cache_collection,
        ttl=timedelta(days=7),
    )


def get_semantic_cache() -> Optional[CouchbaseSemanticCache]:
    """
    Return a CouchbaseSemanticCache instance for similarity-based LLM caching.

    Requires:
      • semantic_cache collection in Couchbase
      • semantic_cache_index FTS vector index (see indexes/semantic_cache_index.json)
    """
    cluster, ok, _ = get_cluster()
    if not ok or cluster is None:
        return None

    cfg = get_settings()
    return CouchbaseSemanticCache(
        cluster=cluster,
        embedding=get_embeddings(),
        bucket_name=cfg.couchbase_bucket,
        scope_name=cfg.couchbase_scope,
        collection_name=cfg.couchbase_semantic_cache_collection,
        index_name=cfg.couchbase_semantic_cache_index,
        score_threshold=cfg.semantic_cache_score_threshold,
        ttl=timedelta(days=7),
    )
