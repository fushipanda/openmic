"""Realtime transcription using ElevenLabs Scribe WebSocket."""

import asyncio
import base64
import json
import os
import queue
from typing import Callable

import websockets

from elevenlabs.client import ElevenLabs


class RealtimeTranscriber:
    """Handles realtime transcription via ElevenLabs Scribe WebSocket."""

    WEBSOCKET_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?commit_strategy=vad"
    SAMPLE_RATE = 16000

    def __init__(
        self,
        on_partial: Callable[[str], None] | None = None,
        on_committed: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self.on_partial = on_partial
        self.on_committed = on_committed
        self.on_error = on_error
        self.on_debug = on_debug
        self.verbose = False
        self._ws = None
        self._running = False
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._send_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._chunks_sent = 0
        self._max_peak = 0

    async def connect(self) -> None:
        """Establish WebSocket connection to Scribe realtime API."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        self._running = True
        self._loop = asyncio.get_event_loop()

        # Connect with API key header
        headers = {"xi-api-key": api_key}
        self._ws = await websockets.connect(
            self.WEBSOCKET_URL,
            additional_headers=headers,
            subprotocols=["chat"],
        )

        # Start send and receive tasks
        self._send_task = asyncio.create_task(self._send_audio_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._running = False

        # Cancel tasks
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Send an audio chunk to the transcription service.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz mono)
        """
        if not self._running:
            return

        # Queue the audio for async sending
        self._audio_queue.put(audio_bytes)

    async def _send_audio_loop(self) -> None:
        """Continuously send queued audio chunks to the WebSocket."""
        self._chunks_sent = 0
        self._max_peak = 0
        while self._running and self._ws:
            try:
                try:
                    audio_bytes = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._audio_queue.get(timeout=0.1)
                    )
                except queue.Empty:
                    continue

                # Track peak amplitude for verbose mode
                if self.verbose and self.on_debug:
                    import numpy as np
                    samples = np.frombuffer(audio_bytes, dtype=np.int16)
                    peak = int(np.max(np.abs(samples)))
                    if peak > self._max_peak:
                        self._max_peak = peak

                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": audio_b64,
                    "sample_rate": self.SAMPLE_RATE,
                }
                await self._ws.send(json.dumps(message))
                self._chunks_sent += 1

                if self.verbose and self.on_debug and self._chunks_sent % 50 == 0:
                    self.on_debug(f"chunks={self._chunks_sent} peak={self._max_peak}")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Send error: {e}")
                continue

    async def _receive_loop(self) -> None:
        """Continuously receive transcription events from the WebSocket."""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                event = json.loads(message)
                message_type = event.get("message_type", "")

                if self.verbose and self.on_debug:
                    text_preview = event.get("text", "")[:80]
                    self.on_debug(f"ws recv: {message_type} [{text_preview}]")

                if message_type == "partial_transcript":
                    text = event.get("text", "")
                    if text and self.on_partial:
                        self.on_partial(text)

                elif message_type == "committed_transcript":
                    text = event.get("text", "")
                    if text and self.on_committed:
                        self.on_committed(text)

                elif message_type == "session_started":
                    if self.on_error:
                        self.on_error("Connected to ElevenLabs Scribe")

                elif message_type in ("auth_error", "quota_exceeded", "rate_limited", "input_error"):
                    error_msg = event.get("message", message_type)
                    if self.on_error:
                        self.on_error(f"ElevenLabs: {error_msg}")

            except websockets.exceptions.ConnectionClosed:
                if self.on_error:
                    self.on_error("WebSocket connection closed")
                break
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Receive error: {e}")
                continue

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None


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

    @staticmethod
    def parse_diarized_result(result) -> list[dict]:
        """Parse the diarized transcription result into speaker-labeled segments.

        Args:
            result: The transcription result from ElevenLabs API

        Returns:
            List of dicts with 'speaker', 'text', 'start', 'end' keys
        """
        segments = []

        # ElevenLabs returns words with speaker info in the response
        if hasattr(result, "words") and result.words:
            current_speaker = None
            current_text = []
            current_start = None

            for word in result.words:
                speaker = getattr(word, "speaker_id", None) or "Speaker"
                text = getattr(word, "text", "")
                start = getattr(word, "start", 0)
                end = getattr(word, "end", 0)

                if speaker != current_speaker:
                    if current_text:
                        segments.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_text),
                            "start": current_start,
                            "end": end,
                        })
                    current_speaker = speaker
                    current_text = [text]
                    current_start = start
                else:
                    current_text.append(text)

            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "start": current_start,
                    "end": getattr(result.words[-1], "end", 0) if result.words else 0,
                })
        elif hasattr(result, "text"):
            # Fallback if no word-level info
            segments.append({
                "speaker": "Speaker",
                "text": result.text,
                "start": 0,
                "end": 0,
            })

        return segments
