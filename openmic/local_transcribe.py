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

# webrtcvad frame constants — must be exactly 10, 20, or 30ms at 16kHz
_VAD_FRAME_MS      = 30
_VAD_FRAME_SAMPLES = 16000 * _VAD_FRAME_MS // 1000   # 480 samples
_VAD_FRAME_BYTES   = _VAD_FRAME_SAMPLES * 2           # 960 bytes (int16)
_BYTES_PER_SAMPLE  = 2
_MAX_SPEECH_BYTES  = 16000 * _BYTES_PER_SAMPLE * 30  # 30s safety ceiling
_MAX_IDLE_BYTES    = 16000 * _BYTES_PER_SAMPLE * 35  # 35s before idle trim


def _get_whisper_model():
    """Load whisper.cpp model via pywhispercpp. Auto-downloads if needed."""
    from pywhispercpp.model import Model

    model_size = os.environ.get("WHISPER_MODEL", "small.en")
    return Model(model_size, n_threads=os.cpu_count() or 4, redirect_whispercpp_logs_to=os.devnull)


def _try_load_webrtcvad(aggressiveness: int = 2):
    """Lazy import of webrtcvad. Returns a configured Vad instance or None.

    aggressiveness: 0–3 (0 = least aggressive, 3 = most aggressive silence filtering).
    Returns None if webrtcvad-wheels is not installed — caller falls back to fixed-interval loop.
    """
    try:
        import webrtcvad
        return webrtcvad.Vad(aggressiveness)
    except Exception:
        return None


class LocalRealtimeTranscriber:
    """Realtime-ish transcription using whisper.cpp locally.

    Accumulates audio chunks and transcribes periodically (~10 seconds
    of audio). Not as smooth as a WebSocket stream but keeps data local.
    """

    SAMPLE_RATE = 16000
    CHUNK_INTERVAL_SECS = int(os.environ.get("WHISPER_CHUNK_INTERVAL", "10"))

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
        """Start the local transcription loop, using VAD if webrtcvad is available."""
        self._running = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._get_model)

        vad_enabled = os.environ.get("WHISPER_VAD_ENABLED", "true").lower() != "false"

        if vad_enabled:
            aggressiveness = int(os.environ.get("WHISPER_VAD_AGGRESSIVENESS", "2"))
            vad = _try_load_webrtcvad(aggressiveness)
        else:
            vad = None

        if vad is not None:
            silence_ms = int(os.environ.get("WHISPER_VAD_SILENCE_MS", "600"))
            self._transcribe_task = asyncio.create_task(
                self._vad_transcribe_loop(vad, silence_ms)
            )
        else:
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

    async def _vad_transcribe_loop(self, vad, silence_ms: int) -> None:
        """Transcription loop gated by webrtcvad speech-boundary detection.

        Accumulates audio in a rolling buffer and processes it in 30ms VAD frames.
        Flushes a speech segment to Whisper once silence_ms of consecutive silence
        is detected after speech, rather than waiting for a fixed time interval.
        """
        silence_threshold = max(1, silence_ms // _VAD_FRAME_MS)

        rolling_buffer    = bytearray()
        bytes_processed   = 0
        speech_start_byte = None   # None = currently in silence
        silence_frames    = 0

        while self._running:
            try:
                # Drain audio queue into rolling buffer
                try:
                    while True:
                        rolling_buffer.extend(self._audio_queue.get_nowait())
                except queue.Empty:
                    pass

                # Process all complete 30ms VAD frames available
                while (len(rolling_buffer) - bytes_processed) >= _VAD_FRAME_BYTES:
                    frame = rolling_buffer[bytes_processed : bytes_processed + _VAD_FRAME_BYTES]
                    bytes_processed += _VAD_FRAME_BYTES

                    is_speech = vad.is_speech(bytes(frame), sample_rate=self.SAMPLE_RATE)

                    if is_speech:
                        silence_frames = 0
                        if speech_start_byte is None:
                            speech_start_byte = bytes_processed - _VAD_FRAME_BYTES
                    else:
                        if speech_start_byte is not None:
                            silence_frames += 1
                            if silence_frames >= silence_threshold:
                                # Speech ended — extract segment without trailing silence
                                speech_end = bytes_processed - (silence_frames * _VAD_FRAME_BYTES)
                                segment = bytes(rolling_buffer[speech_start_byte : speech_end])
                                speech_start_byte = None
                                silence_frames    = 0
                                del rolling_buffer[:bytes_processed]
                                bytes_processed = 0

                                if segment:
                                    loop = asyncio.get_event_loop()
                                    text = await loop.run_in_executor(
                                        None, self._transcribe_audio, segment
                                    )
                                    if text and text.strip() and self.on_committed:
                                        self.on_committed(text.strip())
                                continue  # buffer trimmed — restart inner while

                # 30s safety ceiling: force-flush very long unbroken speech
                if speech_start_byte is not None:
                    if (bytes_processed - speech_start_byte) >= _MAX_SPEECH_BYTES:
                        segment = bytes(rolling_buffer[speech_start_byte : bytes_processed])
                        speech_start_byte = bytes_processed
                        silence_frames    = 0
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(None, self._transcribe_audio, segment)
                        if text and text.strip() and self.on_committed:
                            self.on_committed(text.strip())

                # Idle trim: prevent unbounded buffer growth during silence
                if speech_start_byte is None and len(rolling_buffer) > _MAX_IDLE_BYTES:
                    keep = 16000 * _BYTES_PER_SAMPLE * 5  # retain 5s tail
                    trim = len(rolling_buffer) - keep
                    del rolling_buffer[:trim]
                    bytes_processed = max(0, bytes_processed - trim)

                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.on_error:
                    self.on_error(f"VAD transcription error: {e}")
                await asyncio.sleep(0.5)

        # Shutdown flush — transcribe any remaining buffered speech
        if speech_start_byte is not None and bytes_processed > speech_start_byte:
            try:
                segment = bytes(rolling_buffer[speech_start_byte : bytes_processed])
                text = self._transcribe_audio(segment)
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
        """Parse a transcription result into speaker-labeled segments.

        Iterates result.words (each with speaker_id, text, start, end attributes),
        groups consecutive words by speaker, and returns a list of segment dicts.
        All local whisper segments currently carry speaker_id="Speaker".
        """
        segments = []

        if hasattr(result, "words") and result.words:
            current_speaker = None
            current_text    = []
            current_start   = None

            for word in result.words:
                speaker = getattr(word, "speaker_id", None) or "Speaker"
                text    = getattr(word, "text", "")
                start   = getattr(word, "start", 0)
                end     = getattr(word, "end", 0)

                if speaker != current_speaker:
                    if current_text:
                        segments.append({
                            "speaker": current_speaker,
                            "text":    " ".join(current_text),
                            "start":   current_start,
                            "end":     end,
                        })
                    current_speaker = speaker
                    current_text    = [text]
                    current_start   = start
                else:
                    current_text.append(text)

            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text":    " ".join(current_text),
                    "start":   current_start,
                    "end":     getattr(result.words[-1], "end", 0) if result.words else 0,
                })

        elif hasattr(result, "text"):
            segments.append({
                "speaker": "Speaker",
                "text":    result.text,
                "start":   0,
                "end":     0,
            })

        return segments


class _Word:
    """Minimal word-like object for transcription results."""
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
