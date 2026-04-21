"""
graph.py
────────
LangGraph RAG workflow for the Code Civil assistant.

Graph topology
──────────────
  [retrieve] ──► [generate] ──► END

Nodes
─────
• retrieve       – Queries the Couchbase vector store, fills `documents` and `sources`.
• generate       – Assembles context, runs the ChatOpenAI chain with chat history.

State
─────
All nodes read from / write to `RAGState` (TypedDict).  Chat history is injected
as input from the Streamlit session state and is not mutated inside the graph.
"""

from __future__ import annotations

import traceback
from typing import List, TypedDict, Optional

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """Mutable state passed between all LangGraph nodes."""

    question: str
    """The user's current query."""

    chat_history: List[BaseMessage]
    """Previous conversation turns (injected from Streamlit session state)."""

    documents: List[Document]
    """Articles retrieved from the Couchbase vector store."""

    sources: List[dict]
    """Simplified metadata list for the UI sidebar."""

    answer: str
    """The final AI-generated response."""


# ─────────────────────────────────────────────────────────────────────────────
# Graph implementation
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_graph(_vector_store: CouchbaseSearchVectorStore, _settings: Settings) -> StateGraph:
    """Consolidates nodes and assembly into a compiled LangGraph."""

    # ── Expert LLM (uses global cache) ──────────────────────────────────────
    llm = ChatOpenAI(
        model=_settings.llm_model,
        temperature=0.1,
        api_key=_settings.openai_api_key.get_secret_value(),
        streaming=False,
    )

    # ── Retriever ─────────────────────────────────────────────────────────────
    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": _settings.retriever_k},
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def retrieve_node(state: RAGState) -> dict:
        """Query the vector store and prepare citation metadata."""
        question = state["question"]
        docs: List[Document] = retriever.invoke(question)

        sources = []
        for doc in docs:
            # In LangChain 0.3, CouchbaseSearchVectorStore puts the doc ID in doc.id
            doc_id = doc.id or doc.metadata.get("id") or doc.metadata.get("doc_id") or "Référence inconnue"
            
            readable = str(doc_id).replace("_", " - ") if doc_id else "Référence inconnue"

            snippet = doc.page_content
            if len(snippet) > 350:
                snippet = snippet[:350].rsplit(" ", 1)[0] + "…"

            sources.append(
                {
                    "doc_id": doc_id,
                    "readable": readable,
                    "snippet": snippet,
                    "full_text": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        return {"documents": docs, "sources": sources}

    # ─────────────────────────────────────────────────────────────────────────

    _generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Tu es un assistant juridique expert en droit civil français (Code Civil).

INSTRUCTIONS :
1. Base-toi EXCLUSIVEMENT sur les articles du Code Civil fournis ci-dessous.
2. Cite toujours les articles pertinents dans ta réponse (ex: \"Selon l'article 1382 du Code Civil…\").
3. Si plusieurs articles sont pertinents, synthétise-les de manière cohérente.
4. Adopte un ton professionnel mais accessible, pédagogue et précis.
5. Si la réponse exacte n'est pas dans les articles fournis, indique-le clairement
   et suggère des pistes de recherche complémentaires.
6. Structure ta réponse avec des paragraphes clairs. Utilise des listes si utile.
7. GESTION DU CHANGEMENT DE SUJET : Si la nouvelle question porte sur un sujet différent 
   de l'historique de la conversation, ignore l'historique et concentre-toi uniquement 
   sur le nouveau contexte fourni. Ne mélange pas les sujets.

═══════════════════════════════════════
ARTICLES DU CODE CIVIL PERTINENTS :
{context}
═══════════════════════════════════════""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    _generate_chain = _generate_prompt | llm | StrOutputParser()

    def generate_node(state: RAGState) -> dict:
        """Generate the final answer from retrieved articles and chat history."""
        docs = state["documents"]
        context_parts = []
        for doc in docs:
            doc_id = doc.metadata.get("id", "Article Code Civil")
            readable = doc_id.replace("_", " ")
            context_parts.append(f"▸ {readable}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        answer = _generate_chain.invoke(
            {
                "context": context,
                "question": state["question"],
                "chat_history": state.get("chat_history", []),
            }
        )
        return {"answer": answer}

    # ─────────────────────────────────────────────────────────────────────────

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def prune_chat_history(
    question: str, 
    history: List[BaseMessage], 
    llm_model: str, 
    openai_api_key: str
) -> List[BaseMessage]:
    """
    Uses the Judge LLM to filter out messages from the history 
    that are not relevant to the current question.
    """
    if not history:
        return []

    # Initialize a fast, deterministic judge for pruning
    judge = ChatOpenAI(
        model=llm_model,
        temperature=0.0,
        api_key=openai_api_key,
        cache=False
    )

    # Prepare a condensed view of history
    history_text = ""
    for i, msg in enumerate(history):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content)[:200] # truncate for speed
        history_text += f"[{i}] {role}: {content}\n"

    prune_prompt = f"""Tu es un expert en gestion de contexte. 
Question actuelle : {question}

Voici l'historique de la conversation :
{history_text}

Tâche : Identifie les indices des messages (ex: 0, 2, 3) qui sont PERTINENTS pour comprendre ou répondre à la question actuelle. 
Ignore les messages hors-sujet ou traitant d'un domaine juridique totalement différent.

Réponds UNIQUEMENT avec une liste d'indices séparés par des virgules (ex: 0, 1, 4). Si rien n'est pertinent, réponds "NONE"."""

    try:
        response = judge.invoke(prune_prompt)
        text = response.content.strip().upper()
        if text == "NONE":
            return []
        
        indices = [int(i.strip()) for i in text.split(",") if i.strip().isdigit()]
        return [history[i] for i in indices if i < len(history)]
    except Exception:
        # Fallback to full history if pruning fails
        return history
