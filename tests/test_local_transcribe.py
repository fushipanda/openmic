"""Tests for local_transcribe.py: VAD-gated and fixed-interval transcription."""

import asyncio
import queue
from unittest.mock import MagicMock, patch

import pytest

from openmic.local_transcribe import (
    _VAD_FRAME_BYTES,
    _MAX_SPEECH_BYTES,
    LocalBatchTranscriber,
    LocalRealtimeTranscriber,
    LocalResult,
    _Word,
    _try_load_webrtcvad,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class FakeSegment:
    def __init__(self, text, start=0.0, end=1.0):
        self.text  = text
        self.start = start
        self.end   = end


class FakeModel:
    def __init__(self, return_text="hello world"):
        self.return_text = return_text
        self.call_count  = 0
        self.last_audio  = None

    def transcribe(self, audio, **kwargs):
        self.call_count += 1
        self.last_audio = audio
        return [FakeSegment(self.return_text)], None


class FakeVad:
    """Programmable fake webrtcvad.Vad.

    Set `speech_frames` to a list of bools before the test runs.
    Each call to is_speech() pops the next value; defaults to False when exhausted.
    """
    def __init__(self, aggressiveness=2):
        self.speech_frames = []
        self._call_idx     = 0

    def is_speech(self, frame, sample_rate):
        if self._call_idx < len(self.speech_frames):
            result = self.speech_frames[self._call_idx]
        else:
            result = False
        self._call_idx += 1
        return result


def _make_transcriber(fake_model=None, committed=None):
    """Return (transcriber, committed_list) with model pre-loaded to avoid real whisper."""
    if committed is None:
        committed = []
    if fake_model is None:
        fake_model = FakeModel()
    t = LocalRealtimeTranscriber(
        on_committed=lambda text: committed.append(text),
        on_error=lambda msg: None,
    )
    t._model = fake_model  # bypass _get_model() which calls faster-whisper
    return t, committed


def _make_audio(n_frames: int) -> bytes:
    """Return n_frames worth of 16-bit PCM audio with enough energy to pass the RMS gate."""
    import numpy as np
    n_samples = (_VAD_FRAME_BYTES * n_frames) // 2
    samples = np.full(n_samples, 1000, dtype=np.int16)
    return samples.tobytes()


def _run_vad_loop(transcriber, vad, silence_ms, audio_chunks, run_seconds=0.3):
    """Helper: run _vad_transcribe_loop briefly then cancel."""
    async def _run():
        transcriber._running = True
        for chunk in audio_chunks:
            transcriber._audio_queue.put(chunk)
        # Bypass noise-floor calibration in tests — return fixed threshold immediately
        async def _fixed_threshold():
            return 0
        transcriber._calibrate_noise_floor = _fixed_threshold
        task = asyncio.create_task(
            transcriber._vad_transcribe_loop(vad, silence_ms)
        )
        await asyncio.sleep(run_seconds)
        transcriber._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# TestLocalResult
# ---------------------------------------------------------------------------

class TestLocalResult:
    def test_text_joins_word_texts(self):
        words = [_Word("Hello", 0.0, 0.5, "Speaker"), _Word("world.", 0.5, 1.0, "Speaker")]
        r = LocalResult(words=words)
        assert r.text == "Hello world."

    def test_empty_words_gives_empty_text(self):
        r = LocalResult(words=[])
        assert r.text == ""
        assert r.words == []

    def test_word_attributes_preserved(self):
        w = _Word(text="test", start=1.0, end=2.0, speaker_id="Speaker 1")
        assert w.text      == "test"
        assert w.start     == 1.0
        assert w.end       == 2.0
        assert w.speaker_id == "Speaker 1"


# ---------------------------------------------------------------------------
# TestTryLoadWebrtcvad
# ---------------------------------------------------------------------------

class TestTryLoadWebrtcvad:
    def test_returns_none_when_import_fails(self):
        with patch("builtins.__import__", side_effect=ImportError("no webrtcvad")):
            result = _try_load_webrtcvad()
        assert result is None

    def test_returns_vad_instance_when_available(self):
        mock_vad_instance = MagicMock()
        mock_webrtcvad    = MagicMock()
        mock_webrtcvad.Vad.return_value = mock_vad_instance

        with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
            result = _try_load_webrtcvad(aggressiveness=3)

        mock_webrtcvad.Vad.assert_called_once_with(3)
        assert result is mock_vad_instance


# ---------------------------------------------------------------------------
# TestLocalBatchTranscriber
# ---------------------------------------------------------------------------

class TestLocalBatchTranscriber:
    def _make_batch(self, return_text="batch result"):
        bt = LocalBatchTranscriber()
        bt._model = FakeModel(return_text=return_text)
        return bt

    def test_transcribe_file_returns_local_result(self):
        bt     = self._make_batch("transcribed text")
        result = bt.transcribe_file("fake.wav")
        assert isinstance(result, LocalResult)
        assert result.text == "transcribed text"
        assert len(result.words) == 1

    def test_transcribe_file_filters_empty_segments(self):
        bt = self._make_batch()
        bt._model = FakeModel()
        # Override transcribe to return mix of empty and valid segments
        bt._model.transcribe = lambda audio, **kw: (
            [FakeSegment(""), FakeSegment("  "), FakeSegment("valid text", start=2.0, end=3.0)],
            None,
        )
        result = bt.transcribe_file("fake.wav")
        assert len(result.words) == 1
        assert result.words[0].text == "valid text"

    def test_parse_diarized_result_groups_same_speaker(self):
        """Consecutive words from the same speaker are merged into one segment."""
        result = MagicMock()
        result.words = [
            FakeSegment("Hello", start=0.0, end=0.5),
            FakeSegment("world", start=0.5, end=1.0),
        ]
        for w in result.words:
            w.speaker_id = "Speaker"

        segments = LocalBatchTranscriber.parse_diarized_result(result)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "Speaker"
        assert "Hello" in segments[0]["text"]
        assert "world" in segments[0]["text"]

    def test_parse_diarized_result_splits_on_speaker_change(self):
        """Words from different speakers produce separate segments."""
        result = MagicMock()
        w1 = FakeSegment("Hi", start=0.0, end=0.5); w1.speaker_id = "A"
        w2 = FakeSegment("Hey", start=0.5, end=1.0); w2.speaker_id = "B"
        result.words = [w1, w2]

        segments = LocalBatchTranscriber.parse_diarized_result(result)
        assert len(segments) == 2
        assert segments[0]["speaker"] == "A"
        assert segments[1]["speaker"] == "B"

    def test_parse_diarized_result_empty_words(self):
        """Empty word list produces empty segments."""
        from types import SimpleNamespace
        result = SimpleNamespace(words=[])  # no .text attribute — avoids fallback branch
        assert LocalBatchTranscriber.parse_diarized_result(result) == []


# ---------------------------------------------------------------------------
# TestRealtimeTranscriberLifecycle
# ---------------------------------------------------------------------------

class TestRealtimeTranscriberLifecycle:
    def test_send_audio_chunk_ignored_when_not_running(self):
        t, _ = _make_transcriber()
        t.send_audio_chunk(bytes(1024))
        assert t._audio_queue.empty()

    def test_connect_sets_running_flag(self, monkeypatch):
        monkeypatch.setenv("WHISPER_VAD_ENABLED", "false")
        t, _ = _make_transcriber()

        async def _run():
            await t.connect()
            assert t._running is True
            await t.disconnect()

        asyncio.run(_run())

    def test_disconnect_clears_running_and_task(self, monkeypatch):
        monkeypatch.setenv("WHISPER_VAD_ENABLED", "false")
        t, _ = _make_transcriber()

        async def _run():
            await t.connect()
            await t.disconnect()
            assert t._running is False
            assert t._transcribe_task is None

        asyncio.run(_run())

    def test_is_connected_reflects_running_state(self, monkeypatch):
        monkeypatch.setenv("WHISPER_VAD_ENABLED", "false")
        t, _ = _make_transcriber()
        assert t.is_connected is False

        async def _run():
            await t.connect()
            assert t.is_connected is True
            await t.disconnect()
            assert t.is_connected is False

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# TestFixedIntervalFallback
# ---------------------------------------------------------------------------

class TestFixedIntervalFallback:
    def test_fallback_fires_on_committed_when_buffer_full(self, monkeypatch):
        """When VAD is unavailable the fixed-interval loop should still fire."""
        monkeypatch.setenv("WHISPER_CHUNK_INTERVAL", "1")
        committed   = []
        fake_model  = FakeModel(return_text="fixed interval text")

        with patch("openmic.local_transcribe._try_load_webrtcvad", return_value=None):
            t = LocalRealtimeTranscriber(
                on_committed=lambda text: committed.append(text),
                on_error=lambda msg: None,
            )
            t._model = fake_model
            t.CHUNK_INTERVAL_SECS = 1  # shadow class attr for this instance

            audio_data = bytes(16000 * 2)  # 1s of audio at 16kHz int16

            async def _run():
                await t.connect()
                t.send_audio_chunk(audio_data)
                await asyncio.sleep(0.5)
                await t.disconnect()

            asyncio.run(_run())

        assert len(committed) >= 1
        assert "fixed interval text" in committed[0]

    def test_vad_disabled_by_env_var_uses_fixed_loop(self, monkeypatch):
        """WHISPER_VAD_ENABLED=false must skip VAD even if webrtcvad is importable."""
        monkeypatch.setenv("WHISPER_VAD_ENABLED", "false")

        mock_vad = MagicMock()
        with patch("openmic.local_transcribe._try_load_webrtcvad", return_value=mock_vad) as mock_fn:
            t, _ = _make_transcriber()

            async def _run():
                await t.connect()
                await t.disconnect()

            asyncio.run(_run())

        mock_fn.assert_not_called()


# ---------------------------------------------------------------------------
# TestVADTranscribeLoop
# ---------------------------------------------------------------------------

class TestVADTranscribeLoop:
    def test_speech_then_silence_fires_on_committed(self):
        """5 speech frames followed by enough silence should produce one commit."""
        t, committed = _make_transcriber(FakeModel("spoken phrase"))
        vad = FakeVad()
        # 5 speech frames, then 20 silence frames (600ms / 30ms = 20 frames threshold)
        vad.speech_frames = [True] * 5 + [False] * 20

        _run_vad_loop(t, vad, silence_ms=600, audio_chunks=[_make_audio(25)])

        assert len(committed) == 1
        assert "spoken phrase" in committed[0]

    def test_silence_only_produces_no_commit(self):
        """Pure silence should never call on_committed."""
        t, committed = _make_transcriber()
        vad = FakeVad()
        vad.speech_frames = [False] * 50

        _run_vad_loop(t, vad, silence_ms=600, audio_chunks=[_make_audio(50)])

        assert committed == []

    def test_multiple_phrases_produce_multiple_commits(self):
        """Two speech-then-silence cycles should produce two commits."""
        t, committed = _make_transcriber(FakeModel("phrase"))
        vad = FakeVad()
        # phrase 1: 5 speech + 20 silence; phrase 2: 5 speech + 20 silence
        vad.speech_frames = [True] * 5 + [False] * 20 + [True] * 5 + [False] * 20

        _run_vad_loop(t, vad, silence_ms=600, audio_chunks=[_make_audio(50)], run_seconds=0.5)

        assert len(committed) == 2

    def test_trailing_silence_excluded_from_segment(self):
        """The audio passed to _transcribe_audio must not include the trailing silence."""
        captured = []

        def fake_transcribe_audio(audio_bytes):
            captured.append(audio_bytes)
            return "text"

        t, _ = _make_transcriber()
        t._transcribe_audio = fake_transcribe_audio

        vad = FakeVad()
        n_speech  = 5
        n_silence = 20
        vad.speech_frames = [True] * n_speech + [False] * n_silence

        _run_vad_loop(t, vad, silence_ms=600, audio_chunks=[_make_audio(n_speech + n_silence)])

        assert len(captured) == 1
        expected_speech_bytes = n_speech * _VAD_FRAME_BYTES
        assert len(captured[0]) == expected_speech_bytes

    def test_30s_force_flush_fires_on_committed(self):
        """Continuous speech beyond 30s must be force-flushed."""
        t, committed = _make_transcriber(FakeModel("long speech"))
        vad = FakeVad()
        # All speech frames — VAD never sees silence, so force-flush must fire
        n_frames_needed = (_MAX_SPEECH_BYTES // _VAD_FRAME_BYTES) + 10
        vad.speech_frames = [True] * n_frames_needed

        _run_vad_loop(
            t,
            vad,
            silence_ms=600,
            audio_chunks=[_make_audio(n_frames_needed)],
            run_seconds=0.5,
        )

        assert len(committed) >= 1

    def test_aggressiveness_env_var_passed_to_vad(self, monkeypatch):
        """WHISPER_VAD_AGGRESSIVENESS must be forwarded to _try_load_webrtcvad."""
        monkeypatch.setenv("WHISPER_VAD_AGGRESSIVENESS", "3")
        monkeypatch.setenv("WHISPER_VAD_SILENCE_MS", "600")

        captured_args = {}
        fake_vad = FakeVad()

        def fake_try_load(aggressiveness=2):
            captured_args["aggressiveness"] = aggressiveness
            return fake_vad

        with patch("openmic.local_transcribe._try_load_webrtcvad", side_effect=fake_try_load):
            t, _ = _make_transcriber()

            async def _run():
                await t.connect()
                await t.disconnect()

            asyncio.run(_run())

        assert captured_args.get("aggressiveness") == 3
