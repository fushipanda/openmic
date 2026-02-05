"""Realtime transcription using ElevenLabs Scribe WebSocket."""

import asyncio
import base64
import os
from typing import Callable

from elevenlabs.client import ElevenLabs


class RealtimeTranscriber:
    """Handles realtime transcription via ElevenLabs Scribe WebSocket."""

    def __init__(
        self,
        on_partial: Callable[[str], None] | None = None,
        on_committed: Callable[[str], None] | None = None,
    ) -> None:
        self.on_partial = on_partial
        self.on_committed = on_committed
        self._client: ElevenLabs | None = None
        self._connection = None
        self._running = False

    async def connect(self) -> None:
        """Establish WebSocket connection to Scribe realtime API."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        self._client = ElevenLabs(api_key=api_key)
        self._running = True

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        self._connection = None

    def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Send an audio chunk to the transcription service.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz mono)
        """
        if not self._running or self._client is None:
            return

        # ElevenLabs expects base64 encoded audio for the websocket
        # This will be used when streaming is implemented
        _ = base64.b64encode(audio_bytes).decode("utf-8")

    @property
    def is_connected(self) -> bool:
        return self._running and self._client is not None


class BatchTranscriber:
    """Handles batch transcription with diarization via ElevenLabs Scribe."""

    def __init__(self) -> None:
        self._client: ElevenLabs | None = None

    def _get_client(self) -> ElevenLabs:
        if self._client is None:
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY environment variable not set")
            self._client = ElevenLabs(api_key=api_key)
        return self._client

    def transcribe_file(
        self,
        audio_path: str,
        num_speakers: int = 10,
    ) -> dict:
        """Transcribe an audio file with speaker diarization.

        Args:
            audio_path: Path to the audio file
            num_speakers: Maximum number of speakers to detect

        Returns:
            Transcription result with speaker labels
        """
        client = self._get_client()

        with open(audio_path, "rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                diarize=True,
                num_speakers=num_speakers,
            )

        return result
