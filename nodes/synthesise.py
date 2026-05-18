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

_SYSTEM_PROMPT = """You are an expert meeting minutes editor for Microsoft Innovation Hub (MIH).
You receive pre-extracted data from a meeting transcript, including discussion paragraphs
already written by a per-chunk extractor, grouped by topic key.

Your output will be read by people who did NOT attend. They need a complete, accurate
picture — zero fluff, specific details, named attribution.

CRITICAL RULES:
- Return ONLY valid JSON. No preamble, no explanation, no markdown fences.
- Match the exact schema in the user message.

AGENDA ITEMS (problem_statements):
- DERIVE TITLES DIRECTLY from the keys of discussion_paragraphs_by_topic.
  Do NOT invent new titles. Map each topic key → one agenda item.
  You may merge two closely related topic keys into one item (e.g. "Anti-Abuse
  Migration" + "Data Science Rename" → "Data Science Restructure"), but only if
  they are genuinely the same topic discussed from two angles.
- Title Case. 3-6 words. HARD LIMIT: 8 items.

DISCUSSIONS (flexible_tags type="discussion"):
- One per agenda item.
- "speaker" field = EXACTLY the problem_statement content string you wrote above.
- SOURCE: take all paragraphs under the matching topic key(s) from
  discussion_paragraphs_by_topic. Merge them into one cohesive narrative.
  Do NOT discard content. Do NOT paraphrase away specifics.
- LENGTH: 4-8 sentences. Never fewer than 4.
- Keep all named people, numbers, tool names, product names from source paragraphs.
- NEVER start with "The team discussed" or any vague filler.
  Start with a named person or a specific concrete fact.

ACTION ITEMS:
- Include ALL action items from structured_facts.action_items.
- Deduplicate by (description, owner_name) — keep most complete deadline.
- Descriptions must be specific: what exactly must be done, not just a category.

KEY DECISIONS:
- Include ALL decisions from structured_facts.key_decisions.
- Deduplicate. Confirmed outcomes only. Specific — what was decided/agreed/closed.

DEDUPLICATION:
- Same paragraph in multiple topic keys → keep the most complete version, drop shorter.
- Most common engagement_type across chunks wins.
"""

_USER_PROMPT_TEMPLATE = """Consolidate into final meeting minutes.
HARD LIMIT: 8 problem_statements maximum.

EXTRACTED DATA:
{extractions_json}

IMPORTANT: The problem_statements titles MUST come from the keys of
discussion_paragraphs_by_topic. Do not invent topic titles.
Each discussion flexible_tag's "speaker" field MUST exactly match
the corresponding problem_statement "content" string.

Return a JSON object with exactly these fields:
{{
  "engagement_type_detected": "string or null",
  "summary": "4-6 sentences. Name people and concrete outcomes. Lead with the most significant result. Never open with 'The meeting covered...'",
  "problem_statements": [
    {{"type": "problem_statement", "content": "Title from discussion_paragraphs_by_topic key", "speaker": "unknown", "timestamp": null}}
  ],
  "action_items": [
    {{"description": "specific task", "owner_name": "full name", "owner_role": "ta|msft|client|unknown", "deadline": "date or null"}}
  ],
  "client_queries": [
    {{"type": "client_query", "content": "exact question", "speaker": "client", "timestamp": null}}
  ],
  "key_decisions": [
    {{"type": "key_decision", "content": "specific decision made", "speaker": "unknown", "timestamp": null}}
  ],
  "flexible_tags": [
    {{"type": "discussion", "content": "4-8 sentences merged from discussion_paragraphs_by_topic for this topic. Named people. Specific details. No filler openers.", "speaker": "MUST EXACTLY MATCH the problem_statement content string for this topic"}},
    {{"type": "roi_signal", "content": "...", "speaker": "..."}},
    {{"type": "stakeholder_concern", "content": "...", "speaker": "..."}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_EXTRACTIONS_CHARS = 32_000


def _build_extractions_payload(chunk_extractions: list) -> str:
    """
    Consolidate all chunk extractions into a structured payload.

    Groups discussion_paragraphs by topic (deduplicated), then collects
    structured facts (actions, decisions, queries, problem statements).
    The synthesiser merges the pre-written paragraphs rather than writing
    from scratch with no raw material.
    """
    from collections import defaultdict

    valid = [c for c in chunk_extractions if c is not None]
    if not valid:
        return "{}"

    # ── 1. Group discussion_paragraphs by topic ────────────────────────────
    topic_map: dict[str, list[str]] = defaultdict(list)
    seen_para_keys: set[str] = set()

    for chunk in valid:
        for dp in (chunk.get("discussion_paragraphs") or []):
            topic     = (dp.get("topic") or "General").strip()
            paragraph = (dp.get("paragraph") or "").strip()
            if not paragraph:
                continue
            # Deduplicate on first 100 chars (overlapping chunks repeat content)
            key = paragraph[:100].lower()
            if key in seen_para_keys:
                continue
            seen_para_keys.add(key)
            topic_map[topic].append(paragraph)

    # ── 2. Collect structured facts ────────────────────────────────────────
    seen_actions:   set[str] = set()
    seen_decisions: set[str] = set()
    all_problems:   list = []
    all_actions:    list = []
    all_queries:    list = []
    all_decisions:  list = []
    all_flex:       list = []
    summary_frags:  list = []

    for chunk in valid:
        sf = (chunk.get("summary_fragment") or "").strip()
        if sf:
            summary_frags.append(sf)

        all_problems.extend(chunk.get("problem_statements") or [])

        for ai in (chunk.get("action_items") or []):
            key = f"{(ai.get('description',''))[:60]}|{ai.get('owner_name','')}"
            if key not in seen_actions:
                seen_actions.add(key)
                all_actions.append(ai)

        all_queries.extend(chunk.get("client_queries") or [])

        for kd in (chunk.get("key_decisions") or []):
            key = (kd.get("content") or "")[:80].lower()
            if key not in seen_decisions:
                seen_decisions.add(key)
                all_decisions.append(kd)

        all_flex.extend([
            t for t in (chunk.get("flexible_tags") or [])
            if t.get("type") != "discussion"
        ])

    payload = {
        "discussion_paragraphs_by_topic": dict(topic_map),
        "structured_facts": {
            "summary_fragments":  summary_frags,
            "problem_statements": all_problems,
            "action_items":       all_actions,
            "client_queries":     all_queries,
            "key_decisions":      all_decisions,
            "flexible_tags":      all_flex,
        },
    }

    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(raw) > _MAX_EXTRACTIONS_CHARS:
        logger.warning(
            "extractions payload is %d chars — truncating to %d.",
            len(raw), _MAX_EXTRACTIONS_CHARS,
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
        response = chat_completion(messages, model=GROQ_MODEL, max_tokens=6000, temperature=0.1)
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