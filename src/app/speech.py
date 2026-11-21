"""Speech adapters for completed recordings and replies in the Gradio browser UI."""

from __future__ import annotations

import os
import sys
import wave
from functools import lru_cache
from pathlib import Path
from threading import Event

import numpy as np
import requests

# The team's standalone TTS package uses top-level voice and evaluation imports.
TTS_SRC = str(Path(__file__).resolve().parents[2] / "Text_to_speech" / "src")
if TTS_SRC not in sys.path:
    sys.path.insert(0, TTS_SRC)

from voice.backends import ElevenLabsBackend, SystemVoiceBackend
from voice.config import load_settings
from voice.tts import TTSEngine, TTSError


class SpeechError(RuntimeError):
    """A speech failure that can be shown beside the browser controls."""


def transcribe(audio_path: str) -> str:
    """Send the WAV prepared by gr.Audio to Deepgram and return the final transcript."""
    if not audio_path:
        raise SpeechError("Record a question first.")
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise SpeechError("Set DEEPGRAM_API_KEY to use the microphone, or type your question.")
    try:
        with open(audio_path, "rb") as audio:
            with wave.open(audio, "rb") as wav:
                if not wav.readframes(1):
                    raise SpeechError("This recording is empty. Try recording again.")
            audio.seek(0)
            response = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
                params={"model": "nova-3", "language": "en", "smart_format": "true"},
                data=audio,
                timeout=(5, 60),
            )
    except requests.RequestException as exc:
        raise SpeechError("Deepgram could not be reached. Retry or type your question.") from exc
    except (wave.Error, EOFError, OSError) as exc:
        raise SpeechError("The recording could not be read as a WAV file. Try recording again.") from exc

    if response.status_code in (401, 403):
        raise SpeechError("Deepgram rejected the speech request. Check DEEPGRAM_API_KEY and its permissions.")
    if response.status_code == 429:
        raise SpeechError("Deepgram is limiting requests. Try again shortly or type your question.")
    if not response.ok:
        raise SpeechError(f"Deepgram could not transcribe this recording (HTTP {response.status_code}).")
    try:
        transcript = response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SpeechError("Deepgram returned an unreadable transcript. Try recording again.") from exc
    if not isinstance(transcript, str) or not transcript.strip():
        raise SpeechError("No speech was detected. Check your microphone and record again.")
    return transcript.strip()


@lru_cache(maxsize=1)
def _system_backend(rate_wpm: int, timeout_s: float) -> SystemVoiceBackend:
    # System voices run on a dedicated worker. Share its job queue, not a turn.
    return SystemVoiceBackend(rate_wpm=rate_wpm, timeout_s=timeout_s)


def synthesize(text: str, *, cancel: Event | None = None) -> tuple[int, np.ndarray]:
    """Render a reply as mono PCM. Each call owns its provider session and turn."""
    if not text or not text.strip():
        raise SpeechError("There is no reply to read aloud yet.")
    settings = load_settings(overrides={"cache": {"mode": "off"}})
    with requests.Session() as session:
        engine = TTSEngine(
            settings=settings,
            backends={
                "elevenlabs": ElevenLabsBackend(**settings["elevenlabs"], session=session),
                "system": _system_backend(settings["offline"]["rate_wpm"], settings["offline"]["timeout_s"]),
            },
            metrics_log=False,
        )
        try:
            rate, samples = engine.synthesize_to_numpy(text, cancel=cancel)
        except TTSError as exc:
            raise SpeechError(
                "Speech playback is unavailable. Check the configured system voice "
                "or ELEVENLABS_API_KEY and its text-to-speech access."
            ) from exc
    if cancel is not None and cancel.is_set():
        raise SpeechError("Speech was stopped.")
    if samples.size == 0 or (engine.last_metrics and engine.last_metrics.errors):
        raise SpeechError("The speech service could not finish this reply. Try playback again.")
    return rate, samples
