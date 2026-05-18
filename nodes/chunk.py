"""
MoM Agent — Chunk node.

Responsibilities:
  1. Pass through immediately if state["error"] is set.
  2. Read state["merged_text"] produced by clean_merge_node.
  3. Split into chunks of ~2000 tokens with 200-token overlap using tiktoken.
     Overlap ensures that action items and decisions that straddle a chunk
     boundary are captured in at least one chunk.
  4. Token counting uses cl100k_base encoding (compatible with both GPT-4
     and llama-family models via Groq).
  5. Write to state:
       merged_chunks (List[str]) — list of text segments ready for extract_node
"""

from __future__ import annotations

import logging
from typing import List

import tiktoken

from mom_agent.state import MoMState

logger = logging.getLogger(__name__)

# ─── Chunking constants ────────────────────────────────────────────────────────

CHUNK_SIZE    = 2000   # target tokens per chunk
OVERLAP_SIZE  = 200    # token overlap between consecutive chunks
ENCODING_NAME = "cl100k_base"

# Module-level encoder — initialised once to avoid repeated load cost
_ENC = tiktoken.get_encoding(ENCODING_NAME)


# ─── Node entry point ─────────────────────────────────────────────────────────

def chunk_node(state: MoMState) -> MoMState:
    """
    LangGraph node: splits merged_text into overlapping token-bounded chunks.
    Returns updated state with merged_chunks populated.
    """
    logger.info("[chunk] session_id=%s", state["session_id"])

    if state.get("error"):
        logger.warning("[chunk] Skipping — error already set: %s", state["error"])
        return state

    merged_text: str = state.get("merged_text", "") or ""
    if not merged_text.strip():
        error_msg = "chunk_node: merged_text is empty — nothing to chunk."
        logger.error("[chunk] %s", error_msg)
        return {**state, "error": error_msg}

    chunks = _chunk_text(merged_text, _ENC, CHUNK_SIZE, OVERLAP_SIZE)

    logger.info(
        "[chunk] Split into %d chunks (target=%d tokens, overlap=%d tokens).",
        len(chunks), CHUNK_SIZE, OVERLAP_SIZE,
    )
    for i, chunk in enumerate(chunks):
        logger.debug("[chunk] Chunk %d: %d tokens, %d chars", i + 1, len(_ENC.encode(chunk)), len(chunk))

    return {**state, "merged_chunks": chunks}


# ─── Chunking logic ────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    chunk_size: int,
    overlap_size: int,
) -> List[str]:
    """
    Tokenise the full text, then slice into chunks of chunk_size tokens
    with overlap_size token overlap. Decodes each slice back to a string.
    """
    if not text.strip():
        return []

    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap_size

    return chunks