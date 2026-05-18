"""
MoM Agent — Clean & Merge node.

Responsibilities:
  1. Pass through immediately if state["error"] is set.
  2. Strip transcript artefacts: [Music], [Applause], [Laughter], filler tokens,
     and other non-speech noise markers common in auto-generated captions.
  3. Label each content block with source prefix:
       [TRANSCRIPT] — lines from raw_transcript
       [NOTES]      — lines from raw_notes
  4. Merge transcript and notes into a single text block (transcript first,
     notes appended after a clear separator).
  5. Derive speaker_map from the attendees block at the top of the merged text.
     Speaker map logic:
       - Section header contains "Microsoft Innovation Hub" → "ta"
       - Section header contains "Microsoft" but NOT "Innovation Hub" → "msft"
       - Any other organisation section header → "client"
     Stored as plain strings (not enum values) for JSON serialisation.
  6. Write to state:
       merged_text  (Optional[str]  — intermediate, consumed by chunk_node)
       speaker_map  (Optional[dict] — { "Full Name": "ta"|"msft"|"client"|"unknown" })

Note: merged_text is NOT in the original MoMState TypedDict schema.
The LangGraph state dict is open to additional keys — we add it here and
chunk_node reads it. synthesise_node does not need it.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from mom_agent.state import MoMState

logger = logging.getLogger(__name__)

# ─── Artefact patterns to strip from transcript ───────────────────────────────

# Bracketed noise markers: [Music], [Applause], [Laughter], [Inaudible], etc.
_BRACKETED_NOISE_RE = re.compile(r"\[[^\]]{1,40}\]")

# Filler / disfluency tokens (standalone words on a line or within a line)
_FILLER_TOKENS = {
    "um", "uh", "er", "ah", "hmm", "mm", "mhm", "uh-huh",
    "umm", "uhh", "err",
}

# Lines that are purely punctuation or whitespace after cleaning
_EMPTY_OR_PUNCT_RE = re.compile(r"^[\s\W]+$")


# ─── Node entry point ─────────────────────────────────────────────────────────

def clean_merge_node(state: MoMState) -> MoMState:
    """
    LangGraph node: cleans transcript/notes, builds speaker_map, merges text.
    Returns updated state with merged_text and speaker_map populated.
    """
    logger.info("[clean_merge] session_id=%s", state["session_id"])

    # Pass through immediately if a prior node set an error
    if state.get("error"):
        logger.warning("[clean_merge] Skipping — error already set: %s", state["error"])
        return state

    updates: dict = {}

    # ── Clean transcript ──────────────────────────────────────────────────────
    raw_transcript: Optional[str] = state.get("raw_transcript")
    if raw_transcript:
        cleaned_transcript = _clean_text(raw_transcript)
        labelled_transcript = _label_lines(cleaned_transcript, "[TRANSCRIPT]")
        logger.info(
            "[clean_merge] Transcript cleaned: %d → %d chars",
            len(raw_transcript), len(labelled_transcript),
        )
    else:
        labelled_transcript = ""
        logger.warning("[clean_merge] No raw_transcript in state.")

    # ── Clean notes ───────────────────────────────────────────────────────────
    raw_notes: Optional[str] = state.get("raw_notes")
    if raw_notes:
        cleaned_notes = _clean_text(raw_notes)
        labelled_notes = _label_lines(cleaned_notes, "[NOTES]")
        logger.info(
            "[clean_merge] Notes cleaned: %d → %d chars",
            len(raw_notes), len(labelled_notes),
        )
    else:
        labelled_notes = ""
        logger.debug("[clean_merge] No raw_notes in state — notes block will be empty.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    parts = [p for p in [labelled_transcript, labelled_notes] if p.strip()]
    if not parts:
        updates["error"] = "clean_merge_node: no content to merge — both transcript and notes are empty."
        logger.error("[clean_merge] %s", updates["error"])
        return {**state, **updates}

    if labelled_transcript and labelled_notes:
        merged_text = labelled_transcript + "\n\n--- NOTES ---\n\n" + labelled_notes
    else:
        merged_text = parts[0]

    logger.info("[clean_merge] Merged text: %d chars total.", len(merged_text))

    # ── Derive speaker_map ────────────────────────────────────────────────────
    # Use the raw notes first (structured MoM docs have clean attendee blocks),
    # fall back to raw transcript.
    source_for_speaker_map = raw_notes or raw_transcript or ""
    speaker_map = _build_speaker_map(source_for_speaker_map)
    logger.info("[clean_merge] Speaker map: %d entries derived.", len(speaker_map))

    updates["merged_text"] = merged_text
    updates["speaker_map"] = speaker_map

    return {**state, **updates}


# ─── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Strip transcript artefacts from raw text.
    Handles both VTT-sourced transcripts and plain text notes.
    """
    lines = text.splitlines()
    cleaned: list[str] = []

    for line in lines:
        # Strip bracketed noise markers: [Music], [Applause], [Laughter], etc.
        line = _BRACKETED_NOISE_RE.sub("", line)

        # Strip standalone filler tokens (whole-word match, case-insensitive)
        words = line.split()
        words = [
            w for w in words
            if w.lower().strip(",.!?;:") not in _FILLER_TOKENS
        ]
        line = " ".join(words)

        # Drop lines that are purely punctuation / whitespace after cleaning
        if _EMPTY_OR_PUNCT_RE.match(line) or not line.strip():
            continue

        cleaned.append(line.strip())

    return "\n".join(cleaned)


def _label_lines(text: str, label: str) -> str:
    """
    Prepend each non-empty line with a source label.
    e.g. "[TRANSCRIPT] Vikram Nair opened the session..."
    """
    labelled: list[str] = []
    for line in text.splitlines():
        if line.strip():
            labelled.append(f"{label} {line.strip()}")
    return "\n".join(labelled)


# ─── Speaker map derivation ────────────────────────────────────────────────────

# Matches a bullet point attendee entry, e.g.:
#   • Arvind Raman   – Principal Tech Strategist, Microsoft Innovation Hub
#   - Vikram Nair    — CDO, Grasim Textiles
_ATTENDEE_LINE_RE = re.compile(
    r"^[\s•\-\*]+([A-Z][a-zA-Z\-'\.]+(?:\s+[A-Z][a-zA-Z\-'\.]+)+)\s*[–—\-]+",
)


def _build_speaker_map(text: str) -> dict:
    """
    Parse the attendees block from transcript or notes text and return a
    speaker_map dict: { "Full Name": "ta" | "msft" | "client" | "unknown" }

    Section header logic:
      - Contains "Microsoft Innovation Hub"  → "ta"
      - Contains "Microsoft" (not MIH)       → "msft"
      - Any other organisation name          → "client"

    Falls back to "unknown" if name is found outside a recognised section.
    """
    speaker_map: dict = {}
    current_role: str = "unknown"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section headers (lines that are NOT attendee bullet points)
        role = _classify_section_header(stripped)
        if role is not None:
            current_role = role
            continue

        # Try to match an attendee bullet line
        match = _ATTENDEE_LINE_RE.match(line)
        if match:
            full_name = match.group(1).strip()
            speaker_map[full_name] = current_role

    return speaker_map


def _classify_section_header(line: str) -> Optional[str]:
    line = line.strip()

    if not line:
        return None

    if len(line) > 120:
        return None

    if re.match(r"^[•\-\*]", line):
        return None

    line_upper = line.upper()

    if "MICROSOFT INNOVATION HUB" in line_upper:
        return "ta"

    if "MICROSOFT" in line_upper:
        return "msft"

    if re.match(r"^[A-Z][A-Za-z]", line):
        return "client"

    return None