"""
MoM Agent — FastAPI server.

Endpoints:
  POST /run                  — kick off post-session pipeline (YouTube URL)
  POST /audio-chunk          — receive a 30s audio blob from browser mic
  POST /run-live             — trigger pipeline after live recording ends
  GET  /status/{session_id}  — SSE stream of pipeline progress events
  GET  /download/{session_id} — download the generated MoM docx
  GET  /companies            — list companies in blob store
  GET  /sessions/{company}   — list sessions for a company
  GET  /                     — serve the frontend HTML

Run with:
  cd ta_buddy
  uvicorn mom_agent.api:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from mom_agent.graph import build_graph
from mom_agent.state import InputMode
from mom_agent.utils.blob import list_companies, list_sessions, make_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TA Buddy — MoM Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# session_store[session_id] = {
#   "state":    MoMState dict,
#   "status":   "running" | "done" | "error",
#   "progress": [ { "step": str, "status": "pending"|"running"|"done"|"error" } ],
#   "events":   asyncio.Queue  (SSE events)
#   "audio_chunks": [ tmp file paths ]  (live mode only)
# }
# ---------------------------------------------------------------------------

session_store: dict[str, dict] = {}

PIPELINE_STEPS = [
    "Ingesting",
    "Cleaning",
    "Chunking",
    "Extracting",
    "Synthesising",
    "Writing Document",
]

STEP_TO_NODE = {
    "Ingesting":        "ingest_node",
    "Cleaning":         "clean_merge_node",
    "Chunking":         "chunk_node",
    "Extracting":       "extract_node",
    "Synthesising":     "synthesise_node",
    "Writing Document": "doc_writer_node",
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ParticipantIn(BaseModel):
    name:        str
    designation: str
    role:        str  # ta | msft | client | unknown


class RunRequest(BaseModel):
    source_url:      str
    company:         str
    engagement_type: str
    participants:    list[ParticipantIn] = []


class RunLiveRequest(BaseModel):
    session_id:      str
    company:         str
    engagement_type: str
    participants:    list[ParticipantIn] = []


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------

def _make_progress(steps: list[str], current_idx: int,
                   current_status: str = "running") -> list[dict]:
    result = []
    for i, step in enumerate(steps):
        if i < current_idx:
            result.append({"step": step, "status": "done"})
        elif i == current_idx:
            result.append({"step": step, "status": current_status})
        else:
            result.append({"step": step, "status": "pending"})
    return result


async def _push_event(session_id: str, event: dict) -> None:
    entry = session_store.get(session_id)
    if entry:
        await entry["events"].put(json.dumps(event))


def _push_event_sync(session_id: str, event: dict) -> None:
    """Thread-safe push from the graph worker thread."""
    entry = session_store.get(session_id)
    if entry:
        loop = entry.get("loop")
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                entry["events"].put(json.dumps(event)), loop
            )


# ---------------------------------------------------------------------------
# Graph runner (runs in a background thread so it doesn't block FastAPI)
# ---------------------------------------------------------------------------

def _run_graph_thread(session_id: str, initial_state: dict) -> None:
    """
    Runs the compiled LangGraph graph in a thread.
    Emits SSE progress events by monkey-patching each node with a wrapper.
    """
    entry = session_store[session_id]

    # Wrap each node to emit step events
    from mom_agent.nodes import ingest, clean_merge, chunk, extract, synthesise, doc_writer
    from mom_agent import graph as graph_module

    node_map = {
        "ingest_node":      ingest.ingest_node,
        "clean_merge_node": clean_merge.clean_merge_node,
        "chunk_node":       chunk.chunk_node,
        "extract_node":     extract.extract_node,
        "synthesise_node":  synthesise.synthesise_node,
        "doc_writer_node":  doc_writer.doc_writer_node,
    }

    step_names = list(STEP_TO_NODE.keys())

    def _make_wrapper(step_name: str, original_fn):
        step_idx = step_names.index(step_name)

        def wrapper(state):
            _push_event_sync(session_id, {
                "type":     "progress",
                "progress": _make_progress(step_names, step_idx, "running"),
                "step":     step_name,
            })
            result = original_fn(state)
            status = "error" if result.get("error") else "done"
            _push_event_sync(session_id, {
                "type":     "progress",
                "progress": _make_progress(step_names, step_idx, status),
                "step":     step_name,
            })
            return result
        return wrapper

    # Build a fresh graph with wrapped nodes
    from langgraph.graph import StateGraph, END
    from mom_agent.state import MoMState
    from mom_agent.graph import _route_after_ingest, notify_outcome_node

    g = StateGraph(MoMState)
    for node_id, fn in node_map.items():
        step_name = next(k for k, v in STEP_TO_NODE.items() if v == node_id)
        g.add_node(node_id, _make_wrapper(step_name, fn))
    g.add_node("notify_outcome_node", notify_outcome_node)

    g.set_entry_point("ingest_node")
    g.add_conditional_edges("ingest_node", _route_after_ingest)
    g.add_edge("clean_merge_node",    "chunk_node")
    g.add_edge("chunk_node",          "extract_node")
    g.add_edge("extract_node",        "synthesise_node")
    g.add_edge("synthesise_node",     "doc_writer_node")
    g.add_edge("doc_writer_node",     "notify_outcome_node")
    g.add_edge("notify_outcome_node", END)
    compiled = g.compile()

    try:
        final_state = compiled.invoke(initial_state)
        entry["state"] = final_state

        if final_state.get("error"):
            entry["status"] = "error"
            _push_event_sync(session_id, {
                "type":  "error",
                "error": final_state["error"],
            })
        else:
            entry["status"] = "done"
            _push_event_sync(session_id, {
                "type":         "done",
                "session_id":   session_id,
                "docx_filename": final_state.get("docx_filename"),
                "docx_path":    final_state.get("docx_path"),
                "summary":      final_state.get("summary", ""),
                "engagement_type": final_state.get("engagement_type_detected", ""),
                "action_items_count": len(final_state.get("action_items") or []),
                "decisions_count":    len(final_state.get("key_decisions") or []),
                "agenda_items_count": len(final_state.get("problem_statements") or []),
            })
    except Exception as exc:
        logger.exception("Graph run failed for session %s", session_id)
        entry["status"] = "error"
        _push_event_sync(session_id, {
            "type":  "error",
            "error": str(exc),
        })
    finally:
        # Sentinel to close the SSE stream
        _push_event_sync(session_id, {"type": "close"})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>MoM Agent frontend not found</h1>", status_code=404)


@app.post("/run")
async def run_post_session(req: RunRequest, request: Request):
    """Kick off the post-session pipeline with a YouTube / Teams URL."""
    session_id = make_session_id(req.company, req.engagement_type)

    loop = asyncio.get_event_loop()
    session_store[session_id] = {
        "status":       "running",
        "events":       asyncio.Queue(),
        "loop":         loop,
        "audio_chunks": [],
        "state":        {},
    }

    initial_state = {
        "session_id":      session_id,
        "input_mode":      InputMode.POST_SESSION,
        "source_url":      req.source_url,
        "company":         req.company,
        "engagement_type": req.engagement_type,
        "participants":    [p.dict() for p in req.participants],
        "audio_chunks":       None,
        "speaker_map":        None,
        "raw_transcript":     None,
        "merged_text":        None,
        "merged_chunks":      None,
        "chunk_extractions":  None,
        "engagement_type_detected": None,
        "summary":            None,
        "problem_statements": None,
        "action_items":       None,
        "client_queries":     None,
        "key_decisions":      None,
        "flexible_tags":      None,
        "docx_path":          None,
        "docx_filename":      None,
        "error":              None,
    }

    thread = threading.Thread(
        target=_run_graph_thread,
        args=(session_id, initial_state),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id}


@app.post("/audio-chunk")
async def receive_audio_chunk(session_id: str, file: UploadFile = File(...)):
    """
    Receive a 30-second audio blob from the browser MediaRecorder.
    Saves to a temp file and accumulates in session store.
    """
    if session_id not in session_store:
        # Create a live session entry if not yet created
        session_store[session_id] = {
            "status":       "recording",
            "events":       asyncio.Queue(),
            "loop":         asyncio.get_event_loop(),
            "audio_chunks": [],
            "state":        {},
        }

    content   = await file.read()
    tmp       = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp.write(content)
    tmp.close()

    session_store[session_id]["audio_chunks"].append(tmp.name)
    logger.info("audio-chunk: session=%s chunk=%d saved to %s",
                session_id, len(session_store[session_id]["audio_chunks"]), tmp.name)

    return {"chunk_index": len(session_store[session_id]["audio_chunks"])}


@app.post("/run-live")
async def run_live(req: RunLiveRequest):
    """
    Trigger the full pipeline after the TA stops live recording.
    Audio chunks must already have been posted via /audio-chunk.
    """
    session_id = req.session_id
    entry      = session_store.get(session_id)

    if not entry:
        raise HTTPException(status_code=404, detail="Session not found. Post audio chunks first.")

    audio_chunks = entry.get("audio_chunks", [])
    if not audio_chunks:
        raise HTTPException(status_code=400, detail="No audio chunks received for this session.")

    entry["status"] = "running"
    entry["loop"]   = asyncio.get_event_loop()

    initial_state = {
        "session_id":      session_id,
        "input_mode":      InputMode.LIVE,
        "source_url":      None,
        "company":         req.company,
        "engagement_type": req.engagement_type,
        "participants":    [p.dict() for p in req.participants],
        "audio_chunks":    audio_chunks,
        "speaker_map":     None,
        "raw_transcript":  None,
        "merged_text":     None,
        "merged_chunks":   None,
        "chunk_extractions": None,
        "engagement_type_detected": None,
        "summary":            None,
        "problem_statements": None,
        "action_items":       None,
        "client_queries":     None,
        "key_decisions":      None,
        "flexible_tags":      None,
        "docx_path":          None,
        "docx_filename":      None,
        "error":              None,
    }

    thread = threading.Thread(
        target=_run_graph_thread,
        args=(session_id, initial_state),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id}


@app.get("/status/{session_id}")
async def stream_status(session_id: str):
    """
    SSE endpoint — streams progress events to the frontend.
    Events: { type: "progress"|"done"|"error"|"close", ... }
    """
    entry = session_store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found.")

    async def event_generator():
        queue: asyncio.Queue = entry["events"]
        while True:
            try:
                raw = await asyncio.wait_for(queue.get(), timeout=120.0)
                data = json.loads(raw)
                yield f"data: {raw}\n\n"
                if data.get("type") == "close":
                    break
            except asyncio.TimeoutError:
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/download/{session_id}")
async def download_mom(session_id: str):
    """
    Download the generated MoM docx for a completed session.
    Looks in the local fallback outputs/ directory (or Azure in production).
    """
    entry = session_store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found.")
    if entry.get("status") != "done":
        raise HTTPException(status_code=400, detail="Session not yet complete.")

    final_state  = entry.get("state", {})
    docx_path    = final_state.get("docx_path", "")
    docx_filename = final_state.get("docx_filename", "MoM.docx")

    # Local fallback path
    local_path = Path("outputs") / docx_path
    if not local_path.exists():
        # Try Azure download
        try:
            from mom_agent.utils.blob import download_blob
            tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
            tmp.close()
            download_blob(docx_path, tmp.name)
            local_path = Path(tmp.name)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"File not found: {exc}")

    return FileResponse(
        path=str(local_path),
        filename=docx_filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.post("/parse-participants")
async def parse_participants(request: Request):
    """
    Parse a free-text block of participant names/designations into a structured list.
    Accepts: { "text": "<raw paste>", "company": "<client company name>" }
    Returns: { "participants": [ { "name": str, "designation": str, "role": str } ] }
    Role values: ta | msft | client | unknown
    """
    import re as _re
    import json as _json
    body = await request.json()
    raw_text = (body.get("text") or "").strip()
    company  = (body.get("company") or "").strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="text field is required.")

    try:
        from shared.groq_client import chat_completion

        company_clause = f"The client company is '{company}'." if company else ""
        system_lines = [
            "You are a participant list parser for meeting minutes.",
            "Extract every person's name, job designation/title, and organisational role from the text.",
            company_clause,
            "Role classification rules:",
            "  'ta'      = Microsoft Innovation Hub facilitators or Technical Architects",
            "  'msft'    = Other Microsoft personnel (Solution Architects, Specialists, AMs, etc.)",
            "  'client'  = People from the external/customer organisation",
            "  'unknown' = Cannot be determined from the text",
            "Return ONLY a valid JSON object with this exact schema:",
            '{ "participants": [ { "name": "<full name>", "designation": "<job title>", "role": "<ta|msft|client|unknown>" } ] }',
            "No preamble, no markdown fences, no explanation. Just the JSON object.",
        ]
        system_prompt_text = "\n".join(line for line in system_lines if line)
        user_content = f"Parse this participant list:\n\n{raw_text}"

        response = chat_completion(
            [
                {"role": "system", "content": system_prompt_text},
                {"role": "user",   "content": user_content},
            ],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            max_tokens=1000,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if the model adds them
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.IGNORECASE)
        raw = _re.sub(r"\s*```$", "", raw.strip()).strip()
        parsed = _json.loads(raw)
        if "participants" not in parsed:
            raise ValueError("LLM response missing 'participants' key")
        return parsed
    except Exception as exc:
        logger.error("parse-participants failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Participant parsing failed: {exc}")


@app.get("/companies")
async def get_companies():
    """List all company names in the blob store."""
    return {"companies": list_companies()}


@app.get("/sessions/{company}")
async def get_sessions(company: str):
    """List all sessions for a given company."""
    return {"sessions": list_sessions(company)}