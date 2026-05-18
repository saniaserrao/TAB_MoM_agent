from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONTAINER_NAME     = os.getenv("AZURE_BLOB_CONTAINER", "ta-buddy-shared")
_LOCAL_FALLBACK_DIR = Path("outputs")


def _connection_string() -> str:
    """Read at call time — allows env var to be set after import / monkeypatching in tests."""
    return os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")


def _is_azure_configured() -> bool:
    return bool(_connection_string().strip())


def _ensure_local_dir(local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Blob path helpers
# ---------------------------------------------------------------------------

def make_session_blob_path(company: str, session_id: str, filename: str) -> str:
    """
    Canonical blob path for a session file.
    Pattern: <Company>/sessions/<session_id>/<filename>
    e.g.  ICICI_Lombard/sessions/ICICI_Lombard_BEW_17May2026/MoM_ICICI_Lombard_BEW_17May2026.docx
    """
    safe_company = company.strip().replace(" ", "_")
    return f"{safe_company}/sessions/{session_id}/{filename}"


def make_session_id(company: str, engagement_type: str, date_str: Optional[str] = None) -> str:
    """
    Build a canonical session ID.
    Pattern: <Company>_<EngagementType>_<DDMmmYYYY>
    e.g.  ICICI_Lombard_BEW_17May2026
    If date_str is None, today's date is used.
    """
    from datetime import date as _date
    if date_str is None:
        date_str = _date.today().strftime("%d%b%Y")
    safe_company = company.strip().replace(" ", "_")
    safe_et      = engagement_type.strip().replace(" ", "_")
    return f"{safe_company}_{safe_et}_{date_str}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_blob(local_path: str, blob_path: str) -> None:
    """
    Upload a local file to Azure Blob Storage at blob_path inside the
    configured container.

    Local fallback: if AZURE_STORAGE_CONNECTION_STRING is not set, copies
    the file into outputs/<blob_path> so the pipeline works without Azure.

    Raises on failure — does not swallow exceptions.
    """
    if not _is_azure_configured():
        fallback_dest = _LOCAL_FALLBACK_DIR / blob_path
        fallback_dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(local_path, fallback_dest)
        logger.warning(
            "upload_blob: Azure not configured. Written to local fallback: %s",
            fallback_dest,
        )
        return

    from azure.storage.blob import BlobServiceClient  # type: ignore
    service   = BlobServiceClient.from_connection_string(_connection_string())
    container = service.get_container_client(_CONTAINER_NAME)
    with open(local_path, "rb") as data:
        container.upload_blob(name=blob_path, data=data, overwrite=True)
    logger.info("upload_blob: '%s' → blob '%s' in '%s'.", local_path, blob_path, _CONTAINER_NAME)


def download_blob(blob_path: str, local_path: str) -> None:
    """
    Download a blob from Azure Blob Storage to local_path.
    Local fallback mirrors upload_blob behaviour.
    Raises on failure.
    """
    if not _is_azure_configured():
        fallback_src = _LOCAL_FALLBACK_DIR / blob_path
        if not fallback_src.exists():
            raise FileNotFoundError(
                f"download_blob: Azure not configured and local fallback not found: {fallback_src}"
            )
        import shutil
        _ensure_local_dir(local_path)
        shutil.copy2(fallback_src, local_path)
        logger.warning(
            "download_blob: Azure not configured. Copied from local fallback: %s → %s",
            fallback_src,
            local_path,
        )
        return

    from azure.storage.blob import BlobServiceClient  # type: ignore
    service      = BlobServiceClient.from_connection_string(_connection_string())
    container    = service.get_container_client(_CONTAINER_NAME)
    blob_client  = container.get_blob_client(blob_path)
    _ensure_local_dir(local_path)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    logger.info("download_blob: blob '%s' → '%s'.", blob_path, local_path)


def list_companies() -> list[str]:
    """
    Return all top-level company folder names in the container.
    These are virtual directories — blobs with a '/' in their name.
    Local fallback: list subdirectories of outputs/.

    Returns sorted list of company names e.g. ["ICICI_Lombard", "Mother_Dairy", "Shell"].
    Reserved names excluded: "knowledge".
    """
    _RESERVED = {"knowledge"}

    if not _is_azure_configured():
        base = _LOCAL_FALLBACK_DIR
        if not base.exists():
            return []
        companies = [
            d.name for d in base.iterdir()
            if d.is_dir() and d.name not in _RESERVED
        ]
        return sorted(companies)

    from azure.storage.blob import BlobServiceClient  # type: ignore
    service   = BlobServiceClient.from_connection_string(_connection_string())
    container = service.get_container_client(_CONTAINER_NAME)

    companies: set[str] = set()
    for blob in container.list_blobs():
        parts = blob.name.split("/")
        if len(parts) >= 1 and parts[0] not in _RESERVED:
            companies.add(parts[0])
    return sorted(companies)


def list_sessions(company: str) -> list[dict]:
    """
    Return all session entries for a given company.
    Each entry: { "session_id": str, "blob_path": str, "filename": str }

    Looks under <company>/sessions/<session_id>/ for .docx files.
    Local fallback: scans outputs/<company>/sessions/.
    """
    safe_company = company.strip().replace(" ", "_")
    prefix       = f"{safe_company}/sessions/"

    if not _is_azure_configured():
        base = _LOCAL_FALLBACK_DIR / safe_company / "sessions"
        if not base.exists():
            return []
        sessions = []
        for session_dir in sorted(base.iterdir()):
            if not session_dir.is_dir():
                continue
            for f in session_dir.glob("*.docx"):
                sessions.append({
                    "session_id": session_dir.name,
                    "blob_path":  f"{safe_company}/sessions/{session_dir.name}/{f.name}",
                    "filename":   f.name,
                })
        return sessions

    from azure.storage.blob import BlobServiceClient  # type: ignore
    service   = BlobServiceClient.from_connection_string(_connection_string())
    container = service.get_container_client(_CONTAINER_NAME)

    sessions = []
    seen: set[str] = set()
    for blob in container.list_blobs(name_starts_with=prefix):
        parts = blob.name.split("/")
        # Structure: <company>/sessions/<session_id>/<filename>
        if len(parts) < 4:
            continue
        session_id = parts[2]
        filename   = parts[3]
        if not filename.endswith(".docx"):
            continue
        if session_id not in seen:
            seen.add(session_id)
        sessions.append({
            "session_id": session_id,
            "blob_path":  blob.name,
            "filename":   filename,
        })
    return sorted(sessions, key=lambda s: s["session_id"])