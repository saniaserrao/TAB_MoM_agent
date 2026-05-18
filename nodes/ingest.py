"""
MoM Agent — Ingest node.

Responsibilities:
  1. POST_SESSION mode
     a. YouTube URL → yt-dlp pulls .vtt auto-transcript (no Whisper call needed)
     b. Falls back to Groq Whisper if no .vtt available
  2. LIVE mode
     → Reads accumulated audio chunk file paths already written to state by the
       API audio-chunk endpoint. Calls transcription.transcribe() per chunk and
       concatenates into raw_transcript.

Speaker map is derived from participants list in state (set by UI, not inferred).

On any failure: sets state["error"] and returns early.
The graph's conditional edge on ingest_node routes to END if error is set.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from mom_agent.state import InputMode, MoMState
from mom_agent.utils.transcription import transcribe

logger = logging.getLogger(__name__)


# ─── Node entry point ─────────────────────────────────────────────────────────

def ingest_node(state: MoMState) -> MoMState:
    """
    LangGraph node: ingests transcript into state and builds speaker_map
    from the participants list supplied by the UI.
    """
    logger.info("[ingest] session_id=%s | mode=%s", state["session_id"], state["input_mode"])

    updates: dict = {}

    # ── Build speaker_map from participants (UI-supplied, no LLM inference) ───
    participants = state.get("participants") or []
    speaker_map: dict[str, str] = {}
    for p in participants:
        name = (p.get("name") or "").strip()
        role = (p.get("role") or "unknown").strip()
        if name:
            speaker_map[name] = role
    updates["speaker_map"] = speaker_map
    logger.info("[ingest] speaker_map built — %d participants.", len(speaker_map))

    # ── Transcript ingestion ──────────────────────────────────────────────────
    try:
        if state["input_mode"] == InputMode.POST_SESSION or state["input_mode"] == "post_session":
            if not state.get("source_url"):
                raise ValueError("source_url is required for POST_SESSION mode.")
            raw_transcript = _ingest_youtube(state["source_url"])

        elif state["input_mode"] == InputMode.LIVE or state["input_mode"] == "live":
            audio_chunks = state.get("audio_chunks") or []
            if not audio_chunks:
                raise ValueError("No audio_chunks found in state for LIVE mode.")
            raw_transcript = _ingest_live_chunks(audio_chunks)

        else:
            raise ValueError(f"Unknown input_mode: {state['input_mode']}")

        updates["raw_transcript"] = raw_transcript
        logger.info("[ingest] Transcript ingested — %d characters.", len(raw_transcript))

    except Exception as exc:
        logger.error("[ingest] Transcript ingestion failed: %s", exc)
        updates["error"] = f"Transcript ingestion failed: {exc}"
        return {**state, **updates}

    return {**state, **updates}


# ─── YouTube ingestion ────────────────────────────────────────────────────────

def _ingest_youtube(url: str) -> str:
    """
    Pull auto-generated transcript from a YouTube URL using yt-dlp.
    Downloads the .vtt subtitle file and parses it to plain text.
    Falls back to Whisper if no auto-captions are available.
    """
    logger.info("[ingest] Fetching YouTube transcript: %s", url)

    with tempfile.TemporaryDirectory() as tmpdir:
        vtt_path = _download_vtt(url, tmpdir)
        if vtt_path:
            logger.info("[ingest] .vtt transcript found: %s", vtt_path)
            return _parse_vtt(vtt_path)

        logger.warning("[ingest] No auto-captions found. Falling back to Whisper transcription.")
        audio_path = _download_audio(url, tmpdir)
        return transcribe(audio_path)


def _download_vtt(url: str, output_dir: str) -> Optional[str]:
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--skip-download",
        "--no-playlist",
        "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
        url,
    ]
    logger.debug("[ingest] yt-dlp command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.warning("[ingest] yt-dlp stderr: %s", result.stderr[:500])
    vtt_files = list(Path(output_dir).glob("*.vtt"))
    return str(vtt_files[0]) if vtt_files else None


def _download_audio(url: str, output_dir: str) -> str:
    audio_path = os.path.join(output_dir, "audio.mp3")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", audio_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp audio download failed: {result.stderr[:500]}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found after download: {audio_path}")
    return audio_path


def _parse_vtt(vtt_path: str) -> str:
    """Parse a WebVTT file into clean plain text, stripping timestamps and artefacts."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.splitlines()
    cleaned: list[str] = []
    prev_line = ""

    timestamp_re = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")
    tag_re        = re.compile(r"<[^>]+>")
    cue_id_re     = re.compile(r"^\d+$")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if timestamp_re.match(line):
            continue
        if cue_id_re.match(line):
            continue
        line = tag_re.sub("", line).strip()
        if not line:
            continue
        if line == prev_line:
            continue
        cleaned.append(line)
        prev_line = line

    return "\n".join(cleaned)


# ─── Live audio ingestion ─────────────────────────────────────────────────────

def _ingest_live_chunks(audio_chunks: list) -> str:
    """
    Transcribe a list of audio chunk file paths sent by the browser via the
    /audio-chunk API endpoint. Each chunk is ~30 seconds of audio.
    """
    transcripts: list[str] = []
    for i, chunk_path in enumerate(audio_chunks):
        logger.info("[ingest] Transcribing chunk %d/%d: %s", i + 1, len(audio_chunks), chunk_path)
        try:
            text = transcribe(chunk_path)
            transcripts.append(text)
        except Exception as exc:
            logger.warning("[ingest] Chunk %d transcription failed: %s", i + 1, exc)
            transcripts.append("")
    return "\n".join(t for t in transcripts if t)


# ─── Missing import fix ───────────────────────────────────────────────────────
from typing import Optional  # noqa: E402 — kept at bottom to avoid circular at top