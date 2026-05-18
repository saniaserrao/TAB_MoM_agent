"""
MoM Agent — State schema.
All LangGraph nodes read from and write to MoMState.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from typing_extensions import TypedDict


# ─── Enums ────────────────────────────────────────────────────────────────────

class InputMode(str, Enum):
    POST_SESSION = "post_session"   # YouTube URL (PoC) / Teams recording (prod)
    LIVE         = "live"           # Microphone capture via browser mic → server


class Speaker(str, Enum):
    TA      = "ta"       # Microsoft Innovation Hub facilitators / TAs
    MSFT    = "msft"     # Other Microsoft personnel (SAs, specialists, AMs)
    CLIENT  = "client"   # External / customer team
    UNKNOWN = "unknown"


# ─── TypedDicts ───────────────────────────────────────────────────────────────

class Participant(TypedDict):
    """A meeting participant entered manually via the UI."""
    name:        str   # Full name e.g. "Eric Johnson"
    designation: str   # Job title e.g. "CTO"
    role:        str   # Speaker enum value: ta | msft | client | unknown


class ActionItem(TypedDict):
    """A task extracted from the session with owner attribution."""
    description: str
    owner_name:  str            # Full name as it appears in the transcript
    owner_role:  str            # Speaker enum value: ta | msft | client | unknown
    deadline:    Optional[str]  # ISO date string or natural language e.g. "06/05/2025"


class Tag(TypedDict):
    """
    A single extracted tag — used for problem_statements, client_queries,
    key_decisions, and flexible_tags.
    """
    type:      str            # Tag category e.g. "problem_statement", "roi_signal"
    content:   str            # Extracted text content
    speaker:   str            # Speaker enum value
    timestamp: Optional[str]  # From transcript if available, else None


# ─── MoMState ─────────────────────────────────────────────────────────────────

class MoMState(TypedDict):
    """
    Full state schema for the MoM Agent LangGraph graph.
    Every node reads from and writes to this TypedDict.
    Optional fields start as None and are populated by their respective nodes.
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    session_id:      str
    input_mode:      str            # InputMode enum value stored as string
    source_url:      Optional[str]  # YouTube URL (PoC) / Teams recording URL (prod)
    company:         Optional[str]  # Company name — used for blob path and doc title
    engagement_type: Optional[str]  # TA-provided; if None, LLM detects in extract_node
    participants:    Optional[list] # List[Participant] — entered via UI

    # ── Intermediate ──────────────────────────────────────────────────────────
    audio_chunks:     Optional[list] # List[str] — tmp file paths for LIVE mode chunks
    raw_transcript:   Optional[str]  # Raw text from yt-dlp .vtt or Whisper
    speaker_map:      Optional[dict] # { "Full Name": "ta"|"msft"|"client"|"unknown" }
    merged_text:      Optional[str]  # clean_merge_node output — full cleaned+merged text
    merged_chunks:    Optional[list] # List[str] — 2000-token chunks with overlap
    chunk_extractions: Optional[list] # List[dict|None] — per-chunk LLM extraction results

    # ── Extracted (populated by extract_node + synthesise_node) ───────────────
    engagement_type_detected: Optional[str]
    summary:                  Optional[str]        # 3-5 sentence session overview
    problem_statements:       Optional[list]       # List[Tag]
    action_items:             Optional[list]       # List[ActionItem]
    client_queries:           Optional[list]       # List[Tag]
    key_decisions:            Optional[list]       # List[Tag]
    flexible_tags:            Optional[list]       # List[Tag] — engagement-type extras

    # ── Output ────────────────────────────────────────────────────────────────
    docx_path:     Optional[str]   # Blob path: <Company>/sessions/<session_id>/MoM_<session_id>.docx
    docx_filename: Optional[str]   # Filename only: MoM_<session_id>.docx

    # ── Control ───────────────────────────────────────────────────────────────
    error: Optional[str]           # Set by any node on failure; triggers END edge