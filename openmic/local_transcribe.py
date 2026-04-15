"""Local transcription backend using faster-whisper (CTranslate2). No cloud API calls."""

import asyncio
import os
import queue
from typing import Callable

import numpy as np

# webrtcvad frame constants — must be exactly 10, 20, or 30ms at 16kHz
_VAD_FRAME_MS      = 30
_VAD_FRAME_SAMPLES = 16000 * _VAD_FRAME_MS // 1000   # 480 samples
_VAD_FRAME_BYTES   = _VAD_FRAME_SAMPLES * 2           # 960 bytes (int16)
_BYTES_PER_SAMPLE  = 2
_MAX_SPEECH_BYTES  = 16000 * _BYTES_PER_SAMPLE * 30  # 30s safety ceiling
_MAX_IDLE_BYTES    = 16000 * _BYTES_PER_SAMPLE * 35  # 35s before idle trim
# RMS energy gate: frames below this level are treated as silence before webrtcvad sees them.
# Tunes out mic self-noise / fan hiss that fools webrtcvad. Range 0–32768 (16-bit audio).
_VAD_ENERGY_DEFAULT = 200


def _get_whisper_model():
    """Load faster-whisper model via CTranslate2. Auto-downloads if needed. Uses GPU when available."""
    from faster_whisper import WhisperModel

    model_size = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        if device != "cpu" and ("cuda" in str(e).lower() or "cublas" in str(e).lower() or "libcu" in str(e).lower()):
            # CUDA runtime missing — fall back to CPU automatically
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        raise


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
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self.on_partial = on_partial
        self.on_committed = on_committed
        self.on_error = on_error
        self.on_debug = on_debug
        self.on_ready = on_ready
        self.verbose = False
        self._running = False
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._transcribe_task: asyncio.Task | None = None
        self._model = None
        self._chunks_sent = 0
        self._max_peak = 0

    def _get_model(self):
        if self._model is None:
            self._model = _get_whisper_model()
        return self._model

    def _dbg(self, msg: str) -> None:
        if self.on_debug:
            self.on_debug(msg)

    async def connect(self) -> None:
        """Start the local transcription loop, using VAD if webrtcvad is available."""
        self._running = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._get_model)
        if self.on_ready:
            self.on_ready()

        vad_enabled = os.environ.get("WHISPER_VAD_ENABLED", "true").lower() != "false"

        if vad_enabled:
            aggressiveness = int(os.environ.get("WHISPER_VAD_AGGRESSIVENESS", "2"))
            vad = _try_load_webrtcvad(aggressiveness)
        else:
            vad = None

        if vad is not None:
            silence_ms = int(os.environ.get("WHISPER_VAD_SILENCE_MS", "600"))
            self._dbg(f"VAD loop: aggressiveness={aggressiveness}, silence_ms={silence_ms}")
            self._transcribe_task = asyncio.create_task(
                self._vad_transcribe_loop(vad, silence_ms)
            )
        else:
            self._dbg("Fixed-interval loop (VAD disabled), interval=10s")
            self._transcribe_task = asyncio.create_task(self._transcribe_loop())

    async def disconnect(self) -> None:
        """Stop the transcription loop and explicitly free the whisper model."""
        self._running = False
        if self._transcribe_task:
            try:
                # Give the loop up to 15s to exit naturally and run its shutdown flush
                await asyncio.wait_for(self._transcribe_task, timeout=15.0)
            except asyncio.TimeoutError:
                self._transcribe_task.cancel()
                try:
                    await self._transcribe_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._transcribe_task = None
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        # Explicitly release the whisper.cpp C-level context so whisper_free() is
        # called immediately rather than waiting for Python GC.
        self._model = None

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
                    duration_s = len(buffer) / (self.SAMPLE_RATE * 2)
                    self._dbg(f"Fixed flush: {duration_s:.1f}s buffered → whisper")
                    audio_data = bytes(buffer)
                    buffer.clear()

                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        None, self._transcribe_audio, audio_data
                    )
                    self._dbg(f"Whisper returned: {repr(text[:80]) if text else '(empty)'}")
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
                duration_s = len(buffer) / (self.SAMPLE_RATE * 2)
                self._dbg(f"Shutdown flush: {duration_s:.1f}s remaining → whisper")
                text = self._transcribe_audio(bytes(buffer))
                self._dbg(f"Whisper returned: {repr(text[:80]) if text else '(empty)'}")
                if text and text.strip() and self.on_committed:
                    self.on_committed(text.strip())
            except Exception:
                pass

    async def _calibrate_noise_floor(self) -> int:
        """Sample ~1s of audio to measure ambient noise floor, return a threshold above it.

        Collects up to 33 frames (≈1s), computes per-frame RMS, uses the 90th percentile
        as the noise floor estimate, then sets threshold = max(200, floor * 2).
        Logs the result so the user can override with WHISPER_VAD_ENERGY_THRESHOLD.
        """
        _CALIB_FRAMES = 33  # ~1 second at 30ms per frame
        rms_values: list[float] = []
        deadline = asyncio.get_event_loop().time() + 1.5  # wait at most 1.5s

        calib_buffer = bytearray()
        while len(rms_values) < _CALIB_FRAMES and asyncio.get_event_loop().time() < deadline and self._running:
            try:
                while True:
                    calib_buffer.extend(self._audio_queue.get_nowait())
            except queue.Empty:
                pass

            while len(calib_buffer) >= _VAD_FRAME_BYTES:
                frame = calib_buffer[:_VAD_FRAME_BYTES]
                del calib_buffer[:_VAD_FRAME_BYTES]
                arr = np.frombuffer(bytes(frame), dtype=np.int16).astype(np.float32)
                rms_values.append(float(np.sqrt(np.mean(arr ** 2))))
                if len(rms_values) >= _CALIB_FRAMES:
                    break

            await asyncio.sleep(0.03)

        if rms_values:
            rms_values.sort()
            # Use median as noise floor estimate — more robust than p90 to transient bumps
            median_rms = rms_values[len(rms_values) // 2]
            # 1.5x gives headroom over noise while staying below typical speech (3-5x floor)
            # Cap at 800 so we never accidentally silence real speech
            threshold = min(800, max(_VAD_ENERGY_DEFAULT, int(median_rms * 1.5)))
        else:
            median_rms = 0.0
            threshold = _VAD_ENERGY_DEFAULT

        self._dbg(
            f"VAD noise calibration: floor≈{median_rms:.0f} RMS → threshold={threshold} "
            f"(set WHISPER_VAD_ENERGY_THRESHOLD to override)"
        )
        # Put calibration audio back so it isn't lost
        if calib_buffer:
            self._audio_queue.put(bytes(calib_buffer))
        return threshold

    async def _vad_transcribe_loop(self, vad, silence_ms: int) -> None:
        """Transcription loop gated by webrtcvad speech-boundary detection.

        Accumulates audio in a rolling buffer and processes it in 30ms VAD frames.
        Flushes a speech segment to Whisper once silence_ms of consecutive silence
        is detected after speech, rather than waiting for a fixed time interval.
        """
        silence_threshold = max(1, silence_ms // _VAD_FRAME_MS)

        # Auto-calibrate noise floor from the first second of audio unless overridden
        manual_threshold = os.environ.get("WHISPER_VAD_ENERGY_THRESHOLD")
        if manual_threshold:
            energy_threshold = int(manual_threshold)
            self._dbg(f"VAD energy threshold: {energy_threshold} (manual)")
        else:
            energy_threshold = await self._calibrate_noise_floor()

        rolling_buffer    = bytearray()
        bytes_processed   = 0
        speech_start_byte = None   # None = currently in silence
        silence_frames    = 0
        _dbg_frames_total  = 0
        _dbg_frames_speech = 0
        _dbg_last_report   = 0    # frames since last periodic status

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

                    # RMS energy gate — reject frames too quiet to be speech
                    frame_np = np.frombuffer(bytes(frame), dtype=np.int16)
                    rms = np.sqrt(np.mean(frame_np.astype(np.float32) ** 2))
                    if rms < energy_threshold:
                        is_speech = False
                    else:
                        is_speech = vad.is_speech(bytes(frame), sample_rate=self.SAMPLE_RATE)
                    _dbg_frames_total += 1
                    if is_speech:
                        _dbg_frames_speech += 1

                    # Periodic status every ~3s (100 frames × 30ms)
                    if _dbg_frames_total - _dbg_last_report >= 100:
                        pct = 100 * _dbg_frames_speech / _dbg_frames_total
                        self._dbg(
                            f"VAD: {_dbg_frames_total} frames, "
                            f"{_dbg_frames_speech} speech ({pct:.0f}%), "
                            f"in_speech={'yes' if speech_start_byte is not None else 'no'}"
                        )
                        _dbg_last_report = _dbg_frames_total

                    if is_speech:
                        silence_frames = 0
                        if speech_start_byte is None:
                            speech_start_byte = bytes_processed - _VAD_FRAME_BYTES
                            self._dbg(f"VAD: speech start detected at frame {_dbg_frames_total}")
                    else:
                        if speech_start_byte is not None:
                            silence_frames += 1
                            if silence_frames >= silence_threshold:
                                # Speech ended — extract segment without trailing silence
                                speech_end = bytes_processed - (silence_frames * _VAD_FRAME_BYTES)
                                segment = bytes(rolling_buffer[speech_start_byte : speech_end])
                                duration_s = len(segment) / (self.SAMPLE_RATE * _BYTES_PER_SAMPLE)
                                self._dbg(f"VAD: speech end → flushing {duration_s:.2f}s to whisper")
                                speech_start_byte = None
                                silence_frames    = 0
                                del rolling_buffer[:bytes_processed]
                                bytes_processed = 0

                                if segment:
                                    loop = asyncio.get_event_loop()
                                    text = await loop.run_in_executor(
                                        None, self._transcribe_audio, segment
                                    )
                                    self._dbg(f"Whisper returned: {repr(text[:80]) if text else '(empty)'}")
                                    if text and text.strip() and self.on_committed:
                                        self.on_committed(text.strip())
                                continue  # buffer trimmed — restart inner while

                # 30s safety ceiling: force-flush very long unbroken speech
                if speech_start_byte is not None:
                    if (bytes_processed - speech_start_byte) >= _MAX_SPEECH_BYTES:
                        segment = bytes(rolling_buffer[speech_start_byte : bytes_processed])
                        self._dbg(f"VAD: 30s ceiling — force-flushing to whisper")
                        speech_start_byte = bytes_processed
                        silence_frames    = 0
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(None, self._transcribe_audio, segment)
                        self._dbg(f"Whisper returned: {repr(text[:80]) if text else '(empty)'}")
                        if text and text.strip() and self.on_committed:
                            self.on_committed(text.strip())

                # Idle trim: prevent unbounded buffer growth during silence
                if speech_start_byte is None and len(rolling_buffer) > _MAX_IDLE_BYTES:
                    keep = 16000 * _BYTES_PER_SAMPLE * 5  # retain 5s tail
                    trim = len(rolling_buffer) - keep
                    self._dbg(f"VAD: idle trim — discarding {trim / (self.SAMPLE_RATE * _BYTES_PER_SAMPLE):.1f}s of silence")
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
                duration_s = len(segment) / (self.SAMPLE_RATE * _BYTES_PER_SAMPLE)
                self._dbg(f"VAD shutdown flush: {duration_s:.2f}s remaining → whisper")
                text = self._transcribe_audio(segment)
                self._dbg(f"Whisper returned: {repr(text[:80]) if text else '(empty)'}")
                if text and text.strip() and self.on_committed:
                    self.on_committed(text.strip())
            except Exception:
                pass

    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        model = self._get_model()
        device = getattr(getattr(model, 'model', None), 'device', '?')
        self._dbg(f"_transcribe_audio: {len(audio_bytes) / (self.SAMPLE_RATE * 2):.2f}s, device={device}")
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, _ = model.transcribe(
                samples, language="en", beam_size=5,
                no_speech_threshold=0.3, vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        except Exception as e:
            err = str(e).lower()
            if "cuda" in err or "cublas" in err or "libcu" in err:
                # CUDA runtime unavailable — reinitialise on CPU and retry once
                from faster_whisper import WhisperModel
                model_size = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
                self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments, _ = self._model.transcribe(
                    samples, language="en", beam_size=5,
                    no_speech_threshold=0.3, vad_filter=False,
                )
                return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            raise

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
        try:
            raw_segments, _ = model.transcribe(audio_path, language="en", beam_size=5, no_speech_threshold=0.3, vad_filter=True)
        except Exception as e:
            err = str(e).lower()
            if "cuda" in err or "cublas" in err or "libcu" in err:
                from faster_whisper import WhisperModel
                model_size = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
                self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
                raw_segments, _ = self._model.transcribe(audio_path, language="en", beam_size=5, no_speech_threshold=0.3, vad_filter=True)
            else:
                raise
        words = []
        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                words.append(_Word(
                    text=text,
                    start=seg.start,
                    end=seg.end,
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
    """Transcription result with a word list and full text."""
    def __init__(self, words: list[_Word]):
        self.words = words
        self.text = " ".join(w.text for w in words)
