"""
MoM Agent — LangGraph graph definition.

Pipeline:
    ingest_node
        ↓  (conditional: error → END)
    clean_merge_node
        ↓
    chunk_node
        ↓
    extract_node
        ↓
    synthesise_node
        ↓
    doc_writer_node
        ↓
    notify_outcome_node   (no-op if OUTCOME_AGENT_URL not set)
        ↓
    END
"""

from __future__ import annotations

import logging
import os

import httpx
from langgraph.graph import StateGraph, END

from mom_agent.state import MoMState
from mom_agent.nodes.ingest       import ingest_node
from mom_agent.nodes.clean_merge  import clean_merge_node
from mom_agent.nodes.chunk        import chunk_node
from mom_agent.nodes.extract      import extract_node
from mom_agent.nodes.synthesise   import synthesise_node
from mom_agent.nodes.doc_writer   import doc_writer_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notify Outcome Agent node
# ---------------------------------------------------------------------------

def notify_outcome_node(state: dict) -> dict:
    """
    POST the completed MoM blob path to the Outcome Agent's /files endpoint.
    No-op if OUTCOME_AGENT_URL is not configured — safe to leave unset for PoC.
    """
    if state.get("error"):
        logger.warning("notify_outcome_node: upstream error, skipping notification.")
        return state

    outcome_url = os.getenv("OUTCOME_AGENT_URL", "").strip()
    if not outcome_url:
        logger.info("notify_outcome_node: OUTCOME_AGENT_URL not set — skipping.")
        return state

    payload = {
        "session_id": state.get("session_id"),
        "blob_path":  state.get("docx_path"),
        "filename":   state.get("docx_filename"),
        "company":    state.get("company"),
    }

    try:
        response = httpx.post(
            f"{outcome_url.rstrip('/')}/files",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        logger.info(
            "notify_outcome_node: Outcome Agent notified — session_id=%s blob_path=%s",
            payload["session_id"],
            payload["blob_path"],
        )
    except Exception as exc:
        # Non-fatal — MoM doc is already saved; notification failure should not
        # block the TA from downloading their document.
        logger.warning("notify_outcome_node: failed to notify Outcome Agent: %s", exc)

    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_ingest(state: dict) -> str:
    return END if state.get("error") else "clean_merge_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(MoMState)

    graph.add_node("ingest_node",          ingest_node)
    graph.add_node("clean_merge_node",     clean_merge_node)
    graph.add_node("chunk_node",           chunk_node)
    graph.add_node("extract_node",         extract_node)
    graph.add_node("synthesise_node",      synthesise_node)
    graph.add_node("doc_writer_node",      doc_writer_node)
    graph.add_node("notify_outcome_node",  notify_outcome_node)

    graph.set_entry_point("ingest_node")
    graph.add_conditional_edges("ingest_node", _route_after_ingest)
    graph.add_edge("clean_merge_node",    "chunk_node")
    graph.add_edge("chunk_node",          "extract_node")
    graph.add_edge("extract_node",        "synthesise_node")
    graph.add_edge("synthesise_node",     "doc_writer_node")
    graph.add_edge("doc_writer_node",     "notify_outcome_node")
    graph.add_edge("notify_outcome_node", END)

    return graph.compile()