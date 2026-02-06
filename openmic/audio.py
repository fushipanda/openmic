"""Audio capture module using sounddevice."""

import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable

import sounddevice as sd
import numpy as np


def _find_input_device() -> int | None:
    """Find a working input device, preferring pipewire."""
    devices = sd.query_devices()
    # Prefer pipewire, then any device with input channels
    for priority_name in ("pipewire",):
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and priority_name in d["name"].lower():
                return i
    return None


class AudioRecorder:
    """Records audio from microphone to WAV file."""

    SAMPLE_RATE = 16000  # 16kHz for speech recognition
    CHANNELS = 1
    DTYPE = np.int16

    def __init__(
        self,
        output_dir: Path | None = None,
        on_audio_chunk: Callable[[bytes], None] | None = None,
    ) -> None:
        self.output_dir = output_dir or Path(".")
        self.on_audio_chunk = on_audio_chunk
        self._recording = False
        self._paused = False
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._current_file: Path | None = None
        self._device = _find_input_device()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def current_file(self) -> Path | None:
        return self._current_file

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Called for each audio block from the microphone."""
        if status:
            pass  # Could log status flags if needed

        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

        if self.on_audio_chunk is not None:
            self.on_audio_chunk(indata.tobytes())

    def start(self, filename: str | None = None) -> Path:
        """Start recording audio to a WAV file.

        Args:
            filename: Optional filename. If not provided, uses timestamp.

        Returns:
            Path to the output WAV file.
        """
        if self._recording:
            raise RuntimeError("Already recording")

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"recording_{timestamp}.wav"

        self._current_file = self.output_dir / filename
        self._frames = []
        self._recording = True

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

        return self._current_file

    def pause(self) -> None:
        """Pause recording. Stops mic capture but keeps frames and file handle."""
        if not self._recording or self._paused:
            return
        self._paused = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def resume(self) -> None:
        """Resume recording after pause. Restarts mic capture, appends to same file."""
        if not self._recording or not self._paused:
            return
        self._paused = False
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> Path | None:
        """Stop recording and save the WAV file.

        Returns:
            Path to the saved WAV file, or None if not recording.
        """
        if not self._recording:
            return None

        self._recording = False
        self._paused = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            frames = self._frames.copy()
            self._frames = []

        if not frames or self._current_file is None:
            return None

        audio_data = np.concatenate(frames)
        self._save_wav(self._current_file, audio_data)

        saved_file = self._current_file
        self._current_file = None
        return saved_file

    def _save_wav(self, path: Path, data: np.ndarray) -> None:
        """Save audio data to a WAV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(data.tobytes())
