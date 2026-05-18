"""
MoM Agent — Transcription abstraction.
Backend is controlled by the TRANSCRIPTION_BACKEND env var.

  groq  → Groq Whisper whisper-large-v3  (PoC default)
  azure → Azure AI Speech                (production)

All nodes call transcribe(audio_path) and never reference the backend directly.
Swapping to Azure in production requires only flipping the env var.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TRANSCRIPTION_BACKEND = os.getenv("TRANSCRIPTION_BACKEND", "groq")


# ─── Public interface ─────────────────────────────────────────────────────────

def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to text.
    Routes to the backend specified by TRANSCRIPTION_BACKEND.

    Args:
        audio_path: Local path to audio file (.mp3, .wav, .m4a, .webm, etc.)

    Returns:
        Transcribed text as a single string.

    Raises:
        NotImplementedError: If TRANSCRIPTION_BACKEND is set to 'azure'
                             before the Azure backend is implemented.
        ValueError:          If TRANSCRIPTION_BACKEND is an unrecognised value.
    """
    logger.debug("Transcription backend: %s | file: %s", TRANSCRIPTION_BACKEND, audio_path)

    if TRANSCRIPTION_BACKEND == "groq":
        return _transcribe_groq(audio_path)
    elif TRANSCRIPTION_BACKEND == "azure":
        return _transcribe_azure(audio_path)
    else:
        raise ValueError(
            f"Unknown TRANSCRIPTION_BACKEND='{TRANSCRIPTION_BACKEND}'. "
            "Valid values: 'groq', 'azure'."
        )


# ─── Groq backend ─────────────────────────────────────────────────────────────

def _transcribe_groq(audio_path: str) -> str:
    """
    Transcribe using Groq Whisper (whisper-large-v3).
    Uses the shared Groq client for consistent key rotation.
    """
    # Import here to avoid circular import — shared client is at project root
    try:
        from shared.groq_client import get_client
    except ImportError:
        # Fallback for running mom_agent in isolation during development
        import groq as groq_sdk
        api_key = os.getenv("GROQ_API_KEY_1", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY_1 must be set for Groq transcription.")
        client = groq_sdk.Groq(api_key=api_key)
        return _call_groq_transcription(client, audio_path)

    client = get_client()
    return _call_groq_transcription(client, audio_path)


def _call_groq_transcription(client, audio_path: str) -> str:
    """Execute the Groq audio transcription API call."""
    logger.info("Sending audio to Groq Whisper: %s", audio_path)
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="text",
        )
    # response is a plain string when response_format="text"
    return response if isinstance(response, str) else response.text


# ─── Azure backend (production) ───────────────────────────────────────────────

def _transcribe_azure(audio_path: str) -> str:
    """
    Transcribe using Azure AI Speech.
    Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env.
    Speaker diarization is available at this layer — returns attributed transcript.

    TODO: Implement when full MAF migrates to Azure models.
          Install: pip install azure-cognitiveservices-speech
    """
    raise NotImplementedError(
        "Azure Speech transcription backend is not yet implemented. "
        "Set TRANSCRIPTION_BACKEND=groq for PoC development. "
        "Implement _transcribe_azure() when migrating to production Azure models."
    )