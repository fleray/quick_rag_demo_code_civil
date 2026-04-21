"""
app.py
──────
Streamlit entry point for the AI Legal Assistant – Code Civil.

Run:
    streamlit run app.py

Features
────────
• Dark navy / gold legal theme (see .streamlit/config.toml)
• LangGraph RAG pipeline with step-by-step status display
• Sidebar with Couchbase connection status + legal references panel
• Two independent caching toggles: Semantic Cache & Conversational Cache
• Full chat history maintained in Streamlit session state
"""

from __future__ import annotations
from typing import Any, Optional
import time

import streamlit as st
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage

import traceback
from config import get_settings
from database import (
    get_cluster,
    get_conversational_cache,
    get_semantic_cache,
    get_vector_store,
)
from graph import RAGState, build_rag_graph

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Assistant Juridique – Code Civil",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Assistant IA pour le Code Civil français · Propulsé par LangGraph & Couchbase Capella",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – inject additional polish on top of config.toml theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Google Font ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Root overrides ─────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header ─────────────────────────────────────────────────── */
    .legal-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1A2D45 60%, #0D2137 100%);
        border: 1px solid rgba(201, 168, 76, 0.3);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(201,168,76,0.15);
    }
    .legal-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #C9A84C;
        margin: 0 0 4px 0;
        letter-spacing: 0.5px;
    }
    .legal-header p {
        color: rgba(232, 232, 232, 0.65);
        font-size: 0.9rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── Sidebar section titles ──────────────────────────────────────── */
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #C9A84C;
        margin: 16px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(201, 168, 76, 0.25);
    }

    /* ── Status dot ─────────────────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .status-dot.ok  { background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }
    .status-dot.err { background: #e74c3c; box-shadow: 0 0 6px #e74c3c; }

    /* ── Source card in sidebar ──────────────────────────────────────── */
    .source-card {
        background: rgba(201,168,76,0.06);
        border: 1px solid rgba(201,168,76,0.18);
        border-left: 3px solid #C9A84C;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 10px;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .source-card .article-ref {
        font-weight: 600;
        color: #C9A84C;
        font-size: 0.78rem;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .source-card .article-snippet {
        color: rgba(232,232,232,0.72);
        font-size: 0.78rem;
    }

    /* ── Chat message tweaks ─────────────────────────────────────────── */
    .stChatMessage {
        border-radius: 10px !important;
    }

    /* ── Divider ─────────────────────────────────────────────────────── */
    hr { border-color: rgba(201,168,76,0.15) !important; }

    /* ── Cache badge ─────────────────────────────────────────────────── */
    .cache-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-left: 6px;
        vertical-align: middle;
    }
    .cache-badge.on  { background: rgba(46,204,113,0.18); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
    .cache-badge.off { background: rgba(200,200,200,0.08); color: #888; border: 1px solid rgba(200,200,200,0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role", "content"}
if "sources" not in st.session_state:
    st.session_state.sources = []           # list of source dicts from last query
if "last_cache_mode" not in st.session_state:
    st.session_state.last_cache_mode = None
if "cache_hit" not in st.session_state:
    st.session_state.cache_hit = False

# ─────────────────────────────────────────────────────────────────────────────
# Cache Monitor Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CacheMonitor:
    """Wraps a LangChain cache to track hits/misses in Streamlit session state."""
    def __init__(self, inner_cache):
        self.inner_cache = inner_cache

    def lookup(self, prompt: str, llm_string: str) -> Any:
        result = self.inner_cache.lookup(prompt, llm_string)
        st.session_state.cache_hit = (result is not None)
        return result

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        self.inner_cache.update(prompt, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        self.inner_cache.clear(**kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure (cached singletons)
# ─────────────────────────────────────────────────────────────────────────────

settings = get_settings()
_, cb_ok, cb_msg = get_cluster()
vector_store = get_vector_store()

# Build / retrieve the compiled LangGraph graph (only when vector_store is ready)
rag_graph = build_rag_graph(vector_store, settings) if vector_store else None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <a href="/" target="_self" style="text-decoration:none;">
            <div style="text-align:center; padding: 8px 0 16px 0; cursor:pointer;">
                <div style="font-size: 2.8rem; line-height:1;">⚖️</div>
                <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                            font-weight:700; color:#C9A84C; margin-top:4px;">
                    Code Civil IA
                </div>
                <div style="font-size:0.72rem; color:rgba(232,232,232,0.45);
                            letter-spacing:1px; text-transform:uppercase; margin-top:2px;">
                    Assistant Juridique
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Couchbase status ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Base de données</div>', unsafe_allow_html=True)
    dot_class = "ok" if cb_ok else "err"
    st.markdown(
        f'<span class="status-dot {dot_class}"></span>'
        f'<span style="font-size:0.82rem; color:rgba(232,232,232,0.75);">{cb_msg}</span>',
        unsafe_allow_html=True,
    )
    if cb_ok:
        st.markdown(
            f'<div style="font-size:0.72rem; color:rgba(232,232,232,0.4); margin-top:4px;">'
            f'{settings.couchbase_bucket} › {settings.couchbase_scope} › {settings.couchbase_collection}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Caching controls ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Paramètres de cache</div>', unsafe_allow_html=True)

    use_semantic = st.checkbox(
        "Cache Sémantique",
        value=True,
        help=(
            "CouchbaseSemanticCache : retrouve les réponses aux questions similaires "
            "sans appeler le LLM. Utilise une recherche vectorielle sur la collection "
            f"'{settings.couchbase_semantic_cache_collection}'."
        ),
        key="use_semantic_cache",
    )
    use_conv = st.checkbox(
        "Cache Conversationnel",
        value=False,
        help=(
            "CouchbaseCache : correspondance exacte sur la clé de requête. "
            f"Stocké dans la collection '{settings.couchbase_cache_collection}'. "
            "Activé uniquement si le cache sémantique est désactivé."
        ),
        key="use_conv_cache",
    )

    # Apply cache (semantic takes priority; conv is only used if semantic off)
    cache_mode = "none"
    if use_semantic:
        cache_mode = "semantic"
    elif use_conv:
        cache_mode = "conv"

    if cache_mode != st.session_state.last_cache_mode:
        if cache_mode == "semantic":
            sc = get_semantic_cache()
            if sc:
                set_llm_cache(CacheMonitor(sc))
            else:
                st.warning("⚠️ Cache sémantique indisponible (vérifiez Couchbase).")
        elif cache_mode == "conv":
            cc = get_conversational_cache()
            if cc:
                set_llm_cache(CacheMonitor(cc))
            else:
                st.warning("⚠️ Cache conversationnel indisponible.")
        else:
            set_llm_cache(None)
        st.session_state.last_cache_mode = cache_mode

    # Cache badge
    badge_label = {"semantic": "Sémantique", "conv": "Conversationnel", "none": "Désactivé"}[cache_mode]
    badge_class = "on" if cache_mode != "none" else "off"
    st.markdown(
        f'<div style="margin-top:6px; font-size:0.78rem; color:rgba(232,232,232,0.55);">'
        f'Mode actif : <span class="cache-badge {badge_class}">{badge_label}</span></div>',
        unsafe_allow_html=True,
    )

    if use_semantic and use_conv:
        st.caption("ℹ️ Le cache sémantique a priorité sur le conversationnel.")

    st.divider()

    # ── Legal references panel ─────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Références légales</div>', unsafe_allow_html=True)

    if not st.session_state.sources:
        st.markdown(
            '<div style="font-size:0.78rem; color:rgba(232,232,232,0.35); '
            'text-align:center; padding:12px 0; font-style:italic;">'
            'Les articles utilisés apparaîtront ici après votre première question.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:0.75rem; color:rgba(232,232,232,0.45); margin-bottom:8px;">'
            f'{len(st.session_state.sources)} article(s) consulté(s)</div>',
            unsafe_allow_html=True,
        )
        for src in st.session_state.sources:
            with st.expander(f"📜 {src['readable'][:45]}…" if len(src['readable']) > 45 else f"📜 {src['readable']}", expanded=False):
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="article-ref">{src["readable"]}</div>'
                    f'<div class="article-snippet">{src["snippet"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Chat controls ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Conversation</div>', unsafe_allow_html=True)

    if st.button("🗑️ Effacer la conversation", use_container_width=True, key="clear_chat_btn"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

    st.caption(f"💬 {len(st.session_state.messages)} message(s)")

    st.divider()
    st.markdown(
        '<div style="font-size:0.65rem; color:rgba(232,232,232,0.25); text-align:center;">'
        'Propulsé par LangGraph · Couchbase Capella · OpenAI'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="legal-header">
        <h1>⚖️ Assistant Juridique – Code Civil Français</h1>
        <p>
            Posez vos questions sur le droit civil français.
            Les réponses sont basées exclusivement sur les articles du Code Civil indexés dans Couchbase Capella.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Error state if no vector store ─────────────────────────────────────────────
if not cb_ok or vector_store is None:
    st.error(
        "🔴 Impossible de se connecter à Couchbase Capella. "
        "Vérifiez votre fichier `.env` et les paramètres de connexion dans la barre latérale."
    )
    st.info(
        "**Pour configurer l'application :**\n"
        "1. Copiez `.env.template` → `.env`\n"
        "2. Renseignez vos identifiants Couchbase et votre clé OpenAI\n"
        "3. Relancez l'application avec `streamlit run app.py`"
    )
    st.stop()

# ── Render existing chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚖️" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            indicators = []
            if "cache_info" in msg and msg["cache_info"]:
                indicators.append(msg["cache_info"])
            if "duration" in msg and msg["duration"]:
                indicators.append(f"⏱️ *{msg['duration']:.2f}s*")
            
            if indicators:
                st.markdown(f"<small>{' · '.join(indicators)}</small>", unsafe_allow_html=True)

# ── Example prompts (shown only when chat is empty) ───────────────────────────
if not st.session_state.messages:
    st.markdown(
        '<div style="text-align:center; color:rgba(232,232,232,0.35); '
        'font-size:0.82rem; margin: 12px 0 8px 0;">Exemples de questions</div>',
        unsafe_allow_html=True,
    )
    example_questions = [
        ("👨‍👩‍👧 Famille", "Quels sont les droits et obligations des époux dans le mariage selon le Code Civil ?"),
        ("🏠 Propriété", "Comment fonctionne le droit de propriété et quelles en sont les limites selon le Code Civil ?"),
        ("📜 Contrats", "Quelles sont les conditions de validité d'un contrat selon le Code Civil ?"),
        ("🛡️ Responsabilité", "Quelles sont les conditions pour engager la responsabilité civile extra-contractuelle ?"),
        ("⚱️ Successions", "Comment se déroule la dévolution légale d'une succession en l'absence de testament ?"),
        ("🆔 Personnes", "Quelles sont les règles relatives au changement de prénom dans le Code Civil ?"),
        ("🏢 Biens", "Quelle est la distinction entre les biens meubles et les biens immeubles ?"),
        ("🤝 Obligations", "Quels sont les différents types d'obligations prévus par le Code Civil ?"),
        ("💍 Mariage", "Quelles sont les causes de divorce reconnues par le Code Civil français ?"),
    ]
    
    # Render examples in a 3x3 grid
    for i in range(0, len(example_questions), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(example_questions):
                label, question = example_questions[i + j]
                if cols[j].button(label, key=f"example_{i+j}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    placeholder="Posez votre question sur le Code Civil français…",
    key="chat_input",
)

if user_input:
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.cache_hit = False  # Reset for new query

    with st.chat_message("user", avatar="🧑‍⚖️"):
        st.markdown(user_input)

    from graph import prune_chat_history

    # ── Build LangChain chat history from session ──────────────────────────────
    raw_history = []
    for msg in st.session_state.messages[:-1]:  # exclude the just-added user msg
        if msg["role"] == "user":
            raw_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            raw_history.append(AIMessage(content=msg["content"]))
    
    # Filter history by relevance using the Judge (to avoid topic-bleed)
    if raw_history:
        with st.spinner("🧠 Filtrage de l'historique..."):
            lc_history = prune_chat_history(
                question=user_input,
                history=raw_history,
                llm_model=settings.llm_model,
                openai_api_key=settings.openai_api_key.get_secret_value()
            )
    else:
        lc_history = []

    # ── Invoke the LangGraph RAG pipeline ─────────────────────────────────────
    with st.chat_message("assistant", avatar="⚖️"):
        final_state: RAGState | None = None
        answer = ""
        start_time = time.time()  # Start the clock

        with st.status("🔍 Recherche en cours…", expanded=True) as status:
            try:
                for event in rag_graph.stream(
                    {
                        "question": user_input,
                        "chat_history": lc_history,
                        "documents": [],
                        "sources": [],
                        "quality_ok": None,
                        "answer": "",
                    },
                    stream_mode="values",
                ):
                    # Step 1 – Documents retrieved
                    if event.get("documents") and not event.get("answer"):
                        n = len(event["documents"])
                        status.update(
                            label=f"📚 {n} article(s) récupéré(s) · Génération de la réponse…",
                            state="running",
                        )

                    # Step 2 – Final answer received
                    if event.get("answer"):
                        final_state = event
                        status.update(label="✅ Réponse prête", state="complete", expanded=False)

            except Exception:
                status.update(label="❌ Erreur lors du traitement", state="error")
                st.error(f"Une erreur s'est produite :\n{traceback.format_exc()}")
                st.stop()

        # ── Display the answer ─────────────────────────────────────────────────
        if final_state and final_state.get("answer"):
            answer = final_state["answer"]
            duration = time.time() - start_time  # Calculate duration
            st.markdown(answer)

            # Determine cache indicator
            indicators = []
            cache_info = ""
            if st.session_state.last_cache_mode != "none":
                cache_icon = "💾" if st.session_state.cache_hit else "🐟"
                cache_label = "Cache" if st.session_state.cache_hit else "Dory mode"
                cache_info = f"{cache_icon} *{cache_label}*"
                indicators.append(cache_info)
            
            indicators.append(f"⏱️ *{duration:.2f}s*")
            st.markdown(f"<small>{' · '.join(indicators)}</small>", unsafe_allow_html=True)

            # Update sources in sidebar state
            st.session_state.sources = final_state.get("sources", [])

            # Append assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "cache_info": cache_info,
                "duration": duration
            })
        else:
            fallback = "Une erreur inattendue s'est produite. Veuillez réessayer."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})

    # Rerun to refresh the sidebar sources panel immediately
    st.rerun()
