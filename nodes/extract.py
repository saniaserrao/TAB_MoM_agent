"""
MoM Agent — Extract node.

Responsibilities:
  1. Pass through immediately if state["error"] is set.
  2. For each chunk in state["merged_chunks"], make one Groq chat_completion call
     using the system + user prompts from prompts/extraction.py.
  3. Parse the JSON response from the LLM.
  4. Accumulate partial extraction dicts across all chunks into state.

LangGraph map pattern:
  extract_node is called once per chunk in the pipeline. The results from all
  chunk calls are accumulated in state["chunk_extractions"] (a list of dicts).
  synthesise_node reads this list and reduces/deduplicates across chunks.

State fields written:
  chunk_extractions  (List[dict]) — one raw extraction dict per chunk
  engagement_type_detected (str | None) — majority-vote across chunks

On LLM or parse failure for a chunk:
  Log a warning and append None to chunk_extractions for that chunk position.
  Do NOT set state["error"] — partial extraction is acceptable.
  Only set state["error"] if ALL chunks fail.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from mom_agent.state import MoMState
from mom_agent.prompts.extraction import system_prompt, user_prompt_template

logger = logging.getLogger(__name__)


# ─── Node entry point ─────────────────────────────────────────────────────────

def extract_node(state: MoMState) -> MoMState:
    """
    LangGraph node: runs LLM extraction over each merged chunk.
    Returns updated state with chunk_extractions and engagement_type_detected.
    """
    logger.info("[extract] session_id=%s", state["session_id"])

    # Pass through immediately if a prior node set an error
    if state.get("error"):
        logger.warning("[extract] Skipping — error already set: %s", state["error"])
        return state

    chunks: list = state.get("merged_chunks") or []
    if not chunks:
        error_msg = "extract_node: merged_chunks is empty — nothing to extract from."
        logger.error("[extract] %s", error_msg)
        return {**state, "error": error_msg}

    speaker_map: dict = state.get("speaker_map") or {}
    if not speaker_map:
        logger.warning("[extract] speaker_map is empty — speaker attribution will be 'unknown'.")

    chunk_extractions: list = []
    failed_count = 0

    for i, chunk in enumerate(chunks):
        logger.info("[extract] Processing chunk %d/%d (%d chars).", i + 1, len(chunks), len(chunk))
        try:
            extraction = _extract_chunk(chunk, speaker_map)
            chunk_extractions.append(extraction)
            logger.debug("[extract] Chunk %d extracted successfully.", i + 1)
        except Exception as exc:
            logger.warning("[extract] Chunk %d extraction failed: %s", i + 1, exc)
            chunk_extractions.append(None)
            failed_count += 1

    if failed_count == len(chunks):
        error_msg = "extract_node: all chunks failed LLM extraction."
        logger.error("[extract] %s", error_msg)
        return {**state, "error": error_msg, "chunk_extractions": chunk_extractions}

    logger.info(
        "[extract] Extraction complete: %d/%d chunks succeeded.",
        len(chunks) - failed_count, len(chunks),
    )

    # Determine engagement_type_detected by majority vote across successful chunks
    engagement_type = _majority_vote_engagement_type(chunk_extractions)
    logger.info("[extract] Detected engagement type: %s", engagement_type)

    return {
        **state,
        "chunk_extractions": chunk_extractions,
        "engagement_type_detected": engagement_type,
    }


# ─── Single-chunk extraction ───────────────────────────────────────────────────

def _extract_chunk(chunk: str, speaker_map: dict) -> dict:
    """
    Call the Groq LLM for one chunk and return the parsed extraction dict.
    Raises on LLM error or JSON parse failure.
    """
    from shared.groq_client import chat_completion  # imported here to keep module testable

    user_prompt = user_prompt_template(chunk, speaker_map)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    response = chat_completion(messages, model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # Extract text content from response
    raw_text = _get_text_from_response(response)

    return _parse_json_response(raw_text)


def _get_text_from_response(response) -> str:
    """
    Extract the text string from a Groq chat completion response.
    Handles both object-style (response.choices[0].message.content)
    and dict-style responses.
    """
    try:
        # Standard Groq SDK response object
        return response.choices[0].message.content
    except AttributeError:
        pass

    # Dict-style fallback
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Cannot extract text from LLM response: {response!r}") from exc


def _parse_json_response(raw_text: str) -> dict:
    """
    Parse the LLM's text response as JSON.

    The system prompt instructs the LLM to return only valid JSON. In practice,
    some models occasionally wrap output in markdown fences — strip them first.
    """
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw response (first 500 chars): {raw_text[:500]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM response parsed to {type(parsed).__name__}, expected dict. "
            f"Raw: {raw_text[:200]}"
        )

    # Ensure all expected keys exist (with empty defaults) so downstream
    # synthesise_node can safely iterate without KeyError
    defaults = {
        "engagement_type_detected": None,
        "summary_fragment": "",
        "discussion_paragraphs": [],
        "problem_statements": [],
        "action_items": [],
        "client_queries": [],
        "key_decisions": [],
        "flexible_tags": [],
    }
    for key, default in defaults.items():
        parsed.setdefault(key, default)

    return parsed


# ─── Engagement type voting ────────────────────────────────────────────────────

def _majority_vote_engagement_type(chunk_extractions: list) -> Optional[str]:
    """
    Return the most commonly detected engagement type across all chunks.
    Ignores None entries (failed chunks) and null values from LLM.
    Returns None if no type was detected in any chunk.
    """
    from collections import Counter

    votes: list[str] = []
    for extraction in chunk_extractions:
        if extraction is None:
            continue
        et = extraction.get("engagement_type_detected")
        if et:
            votes.append(et)

    if not votes:
        return None

    most_common, _ = Counter(votes).most_common(1)[0]
    return most_common