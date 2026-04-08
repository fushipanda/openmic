"""Local transcription backend using pywhispercpp (whisper.cpp). No cloud API calls."""

import asyncio
import os
import queue
from typing import Callable

import numpy as np

# Shared whisper params to suppress TUI-breaking output and tune for speech
_WHISPER_PARAMS = {
    "language": b"en",
    "print_progress": False,
    "print_realtime": False,
    "print_timestamps": False,
    "no_speech_thold": 0.3,
}


def _get_whisper_model():
    """Load whisper.cpp model via pywhispercpp. Auto-downloads if needed."""
    from pywhispercpp.model import Model

    model_size = os.environ.get("WHISPER_MODEL", "small.en")
    # Use False to skip stderr redirect — Textual owns stderr so
    # the dup-based redirect in pywhispercpp crashes with fd=-1.
    return Model(model_size, n_threads=os.cpu_count() or 4, redirect_whispercpp_logs_to=False)


class LocalRealtimeTranscriber:
    """Realtime-ish transcription using whisper.cpp locally.

    Accumulates audio chunks and transcribes periodically (~10 seconds
    of audio). Not as smooth as a WebSocket stream but keeps data local.
    """

    SAMPLE_RATE = 16000
    CHUNK_INTERVAL_SECS = 10

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
        self._running = False
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._transcribe_task: asyncio.Task | None = None
        self._model = None
        self._chunks_sent = 0
        self._max_peak = 0

    def _get_model(self):
        if self._model is None:
            model_size = os.environ.get("WHISPER_MODEL", "small.en")
            if self.on_error:
                self.on_error(f"Loading whisper model ({model_size})...")
            self._model = _get_whisper_model()
            if self.on_error:
                self.on_error("Local transcription ready")
        return self._model

    async def connect(self) -> None:
        """Start the local transcription loop."""
        self._running = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._get_model)
        self._transcribe_task = asyncio.create_task(self._transcribe_loop())

    async def disconnect(self) -> None:
        """Stop the transcription loop."""
        self._running = False
        if self._transcribe_task:
            self._transcribe_task.cancel()
            try:
                await self._transcribe_task
            except asyncio.CancelledError:
                pass
            self._transcribe_task = None
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def send_audio_chunk(self, audio_bytes: bytes) -> None:
        if not self._running:
            return
        self._audio_queue.put(audio_bytes)
        self._chunks_sent += 1

    async def _transcribe_loop(self) -> None:
        buffer = bytearray()
        bytes_per_interval = self.SAMPLE_RATE * 2 * self.CHUNK_INTERVAL_SECS

        while self._running:
            try:
                try:
                    while True:
                        chunk = self._audio_queue.get_nowait()
                        buffer.extend(chunk)
                except queue.Empty:
                    pass

                if len(buffer) >= bytes_per_interval:
                    audio_data = bytes(buffer)
                    buffer.clear()

                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        None, self._transcribe_audio, audio_data
                    )
                    if text and text.strip():
                        if self.on_committed:
                            self.on_committed(text.strip())

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Transcription error: {e}")
                await asyncio.sleep(0.5)

        # Flush remaining buffer
        if buffer:
            try:
                text = self._transcribe_audio(bytes(buffer))
                if text and text.strip() and self.on_committed:
                    self.on_committed(text.strip())
            except Exception:
                pass

    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        model = self._get_model()
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments = model.transcribe(samples, **_WHISPER_PARAMS)
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    @property
    def is_connected(self) -> bool:
        return self._running


class LocalBatchTranscriber:
    """Batch transcription using whisper.cpp. No diarisation."""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _get_whisper_model()
        return self._model

    def transcribe_file(self, audio_path: str, num_speakers: int = 10) -> "LocalResult":
        model = self._get_model()
        segments = model.transcribe(audio_path, **_WHISPER_PARAMS)
        words = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                words.append(_Word(
                    text=text,
                    start=seg.t0 / 100.0,
                    end=seg.t1 / 100.0,
                    speaker_id="Speaker",
                ))
        return LocalResult(words=words)

    @staticmethod
    def parse_diarized_result(result) -> list[dict]:
        from openmic.transcribe import BatchTranscriber
        return BatchTranscriber.parse_diarized_result(result)


class _Word:
    """Minimal word-like object matching ElevenLabs result structure."""
    def __init__(self, text: str, start: float, end: float, speaker_id: str):
        self.text = text
        self.start = start
        self.end = end
        self.speaker_id = speaker_id


class LocalResult:
    """Minimal result object matching ElevenLabs transcription result."""
    def __init__(self, words: list[_Word]):
        self.words = words
        self.text = " ".join(w.text for w in words)
