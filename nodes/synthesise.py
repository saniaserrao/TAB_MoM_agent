from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from shared.groq_client import chat_completion

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert meeting minutes writer for Microsoft Innovation Hub (MIH).
You receive structured extractions from overlapping chunks of a single meeting transcript.
Your output will be read by people who did NOT attend the meeting — they must get a full,
accurate understanding of what was discussed, decided, and assigned, with zero fluff.

CRITICAL RULES:
- Return ONLY valid JSON. No preamble, no explanation, no markdown fences.
- The JSON must match the exact schema in the user message.

AGENDA ITEMS (problem_statements field):
- Each entry is ONE distinct agenda topic discussed in the meeting.
- A typical meeting has 3-7 items. HARD LIMIT: 8 maximum.
- Group tightly related sub-points into one item.
- The "content" field is a SHORT TOPIC TITLE (4-7 words), Title Case.
  GOOD: "FedRAMP Compliance and Reviewer Roulette"
  GOOD: "Engineering MR Rate Metrics"
  BAD:  "The team discussed MR rates and how to improve them"

DISCUSSIONS (flexible_tags with type="discussion"):
- Write one "discussion" tag per agenda item.
- Set "speaker" to the EXACT topic title from problem_statements so it can be matched.
- SUBSTANCE RULES — the discussion narrative MUST:
  * Be 4-8 sentences of dense, informative prose. Never fewer than 4 sentences.
  * Name specific people when they raised a point, proposed an idea, or raised a concern.
    Use passive voice only when the speaker is genuinely unknown.
    GOOD: "Wayne confirmed that Anti-Abuse has officially migrated into the Data Science section."
    GOOD: "Thomas raised the FedRAMP deviation requirement, noting that any CVE-linked vulnerability
           in container or dependency scanning needs a documented remediation plan even for false positives."
    BAD:  "The team discussed security issues and audits."
    BAD:  "Updates were shared regarding the reorganization."
  * Explain the WHAT and WHY — not just that something was discussed, but what was actually said,
    what the problem or context is, and what direction was taken or proposed.
  * Include specific numbers, names, tools, timelines, or constraints mentioned in the meeting.
  * A non-attendee reading this should understand the topic as well as someone who was in the room.
  * NEVER start with "The team discussed", "Updates were shared", "The group talked about",
    or any other vague filler opener. Start with a specific person, decision, or concrete fact.

ACTION ITEMS:
- Deduplicate by (description, owner_name). Keep the entry with the most complete deadline.
- Do NOT repeat a decision as an action item.
- description must be a concrete, specific task — not a vague restatement.

KEY DECISIONS:
- Only genuine decisions, agreements, or confirmed outcomes. Not restatements of problems.
- Be specific: include what was decided, not just that a decision was made.

DEDUPLICATION:
- If the same point appears in multiple chunks due to overlap, keep only one instance.
- If engagement_type differs across chunks, use the most common value.
"""

_USER_PROMPT_TEMPLATE = """Consolidate the chunk extractions below into the output schema.
Target 3-7 agenda items. HARD LIMIT: no more than 8 problem_statements.

CHUNK EXTRACTIONS:
{extractions_json}

Return a JSON object with exactly these fields:
{{
  "engagement_type_detected": "string or null",
  "summary": "4-6 sentence factual overview of the entire meeting. Name specific people and outcomes. Never use vague openers like 'The meeting covered...' — start with the most significant outcome or decision.",
  "problem_statements": [
    {{"type": "problem_statement", "content": "SHORT TOPIC TITLE 4-7 words", "speaker": "...", "timestamp": null}}
  ],
  "action_items": [
    {{"description": "specific, concrete task description", "owner_name": "...", "owner_role": "...", "deadline": "... or null"}}
  ],
  "client_queries": [
    {{"type": "client_query", "content": "...", "speaker": "client", "timestamp": null}}
  ],
  "key_decisions": [
    {{"type": "key_decision", "content": "specific decision or agreement reached", "speaker": "...", "timestamp": null}}
  ],
  "flexible_tags": [
    {{"type": "discussion", "content": "4-8 sentences of dense, specific, named narrative for this agenda item. Name people. Include specifics. No filler openers.", "speaker": "EXACT topic title from problem_statements"}},
    {{"type": "roi_signal", "content": "...", "speaker": "..."}},
    {{"type": "stakeholder_concern", "content": "...", "speaker": "..."}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_EXTRACTIONS_CHARS = 28_000  # safety cap before sending to LLM


def _build_extractions_payload(chunk_extractions: list) -> str:
    """
    Serialise chunk_extractions to JSON string, skipping None entries.
    Truncates if the serialised form exceeds the character cap to stay within
    the model context window.
    """
    valid = [c for c in chunk_extractions if c is not None]
    raw = json.dumps(valid, ensure_ascii=False, indent=2)
    if len(raw) > _MAX_EXTRACTIONS_CHARS:
        logger.warning(
            "chunk_extractions payload is %d chars — truncating to %d for LLM call.",
            len(raw),
            _MAX_EXTRACTIONS_CHARS,
        )
        raw = raw[:_MAX_EXTRACTIONS_CHARS] + "\n... (truncated)"
    return raw


def _parse_llm_response(text: str) -> Optional[dict]:
    """Strip markdown fences if present, then parse JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed in synthesise_node: %s", exc)
        return None


def _detect_engagement_type(chunk_extractions: list) -> Optional[str]:
    """Return the most common non-null engagement_type_detected across chunks."""
    counts: dict[str, int] = {}
    for chunk in chunk_extractions:
        if chunk is None:
            continue
        et = chunk.get("engagement_type_detected")
        if et:
            counts[et] = counts.get(et, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def synthesise_node(state: dict) -> dict:
    """
    LangGraph node — reduce and deduplicate all chunk_extractions into final
    MoMState extracted fields via a single Groq LLM call.
    """
    # Pass-through on upstream error
    if state.get("error"):
        logger.warning("synthesise_node: upstream error detected, passing through.")
        return state

    chunk_extractions: list = state.get("chunk_extractions") or []
    valid_chunks = [c for c in chunk_extractions if c is not None]

    if not valid_chunks:
        logger.error("synthesise_node: no valid chunk extractions to synthesise.")
        return {**state, "error": "synthesise_node: all chunk extractions failed or empty"}

    logger.info(
        "synthesise_node: synthesising %d valid chunks (of %d total).",
        len(valid_chunks),
        len(chunk_extractions),
    )

    # Pre-detect engagement_type from chunks as a fallback
    fallback_et = _detect_engagement_type(chunk_extractions)

    extractions_payload = _build_extractions_payload(chunk_extractions)
    user_prompt = _USER_PROMPT_TEMPLATE.format(extractions_json=extractions_payload)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = chat_completion(messages, model=GROQ_MODEL, max_tokens=4096, temperature=0.1)
        raw_text: str = response.choices[0].message.content
    except Exception as exc:
        logger.error("synthesise_node: Groq LLM call failed: %s", exc)
        return {**state, "error": f"synthesise_node: LLM call failed — {exc}"}

    parsed = _parse_llm_response(raw_text)
    if parsed is None:
        return {**state, "error": "synthesise_node: failed to parse LLM JSON response"}

    # Write all extracted fields back to state
    updates: dict[str, Any] = {
        "engagement_type_detected": parsed.get("engagement_type_detected") or fallback_et,
        "summary": parsed.get("summary", ""),
        "problem_statements": parsed.get("problem_statements") or [],
        "action_items": parsed.get("action_items") or [],
        "client_queries": parsed.get("client_queries") or [],
        "key_decisions": parsed.get("key_decisions") or [],
        "flexible_tags": parsed.get("flexible_tags") or [],
    }

    logger.info(
        "synthesise_node: done — %d problem statements, %d action items, "
        "%d client queries, %d key decisions, %d flexible tags.",
        len(updates["problem_statements"]),
        len(updates["action_items"]),
        len(updates["client_queries"]),
        len(updates["key_decisions"]),
        len(updates["flexible_tags"]),
    )

    return {**state, **updates}