"""Unit tests for transcribe.py: parse_diarized_result with sample responses."""

from types import SimpleNamespace

import pytest

from openmic.transcribe import BatchTranscriber


class TestParseDiarizedResult:
    """Tests for BatchTranscriber.parse_diarized_result static method."""

    def test_multiple_speakers(self):
        """Words from different speakers are grouped into separate segments."""
        words = [
            SimpleNamespace(speaker_id="Speaker 1", text="Hello", start=0.0, end=0.5),
            SimpleNamespace(speaker_id="Speaker 1", text="everyone.", start=0.5, end=1.0),
            SimpleNamespace(speaker_id="Speaker 2", text="Hi", start=1.2, end=1.5),
            SimpleNamespace(speaker_id="Speaker 2", text="there.", start=1.5, end=2.0),
            SimpleNamespace(speaker_id="Speaker 1", text="Let's", start=2.5, end=3.0),
            SimpleNamespace(speaker_id="Speaker 1", text="begin.", start=3.0, end=3.5),
        ]
        result = SimpleNamespace(words=words)

        segments = BatchTranscriber.parse_diarized_result(result)

        assert len(segments) == 3
        assert segments[0] == {
            "speaker": "Speaker 1",
            "text": "Hello everyone.",
            "start": 0.0,
            "end": 1.5,  # end attr of first word from next speaker ("Hi")
        }
        assert segments[1] == {
            "speaker": "Speaker 2",
            "text": "Hi there.",
            "start": 1.2,
            "end": 3.0,  # end attr of first word from next speaker ("Let's")
        }
        assert segments[2] == {
            "speaker": "Speaker 1",
            "text": "Let's begin.",
            "start": 2.5,
            "end": 3.5,
        }

    def test_single_speaker(self):
        """All words from one speaker produce a single segment."""
        words = [
            SimpleNamespace(speaker_id="Speaker 1", text="Just", start=0.0, end=0.3),
            SimpleNamespace(speaker_id="Speaker 1", text="me", start=0.3, end=0.5),
            SimpleNamespace(speaker_id="Speaker 1", text="talking.", start=0.5, end=1.0),
        ]
        result = SimpleNamespace(words=words)

        segments = BatchTranscriber.parse_diarized_result(result)

        assert len(segments) == 1
        assert segments[0]["speaker"] == "Speaker 1"
        assert segments[0]["text"] == "Just me talking."

    def test_empty_words(self):
        """Empty word list produces no segments."""
        result = SimpleNamespace(words=[])
        segments = BatchTranscriber.parse_diarized_result(result)
        assert segments == []

    def test_no_words_attribute_with_text_fallback(self):
        """Result with no words but with text attribute uses fallback."""
        result = SimpleNamespace(text="Fallback transcript text.")
        # Remove words attribute so hasattr returns False
        assert not hasattr(result, "words") or not result.words

        # Create a clean result without words
        result = type("Result", (), {"text": "Fallback transcript text."})()

        segments = BatchTranscriber.parse_diarized_result(result)

        assert len(segments) == 1
        assert segments[0]["speaker"] == "Speaker"
        assert segments[0]["text"] == "Fallback transcript text."
        assert segments[0]["start"] == 0
        assert segments[0]["end"] == 0

    def test_missing_speaker_id(self):
        """Words without speaker_id fall back to 'Speaker'."""
        words = [
            SimpleNamespace(text="No", start=0.0, end=0.3),
            SimpleNamespace(text="speaker.", start=0.3, end=0.6),
        ]
        result = SimpleNamespace(words=words)

        segments = BatchTranscriber.parse_diarized_result(result)

        assert len(segments) == 1
        assert segments[0]["speaker"] == "Speaker"
        assert segments[0]["text"] == "No speaker."

    def test_single_word(self):
        """A result with a single word produces one segment."""
        words = [
            SimpleNamespace(speaker_id="Speaker 1", text="Hello.", start=0.0, end=0.5),
        ]
        result = SimpleNamespace(words=words)

        segments = BatchTranscriber.parse_diarized_result(result)

        assert len(segments) == 1
        assert segments[0]["text"] == "Hello."

    def test_many_speaker_transitions(self):
        """Rapid speaker alternation produces correct segment count."""
        words = [
            SimpleNamespace(speaker_id="A", text="one", start=0.0, end=0.5),
            SimpleNamespace(speaker_id="B", text="two", start=0.5, end=1.0),
            SimpleNamespace(speaker_id="A", text="three", start=1.0, end=1.5),
            SimpleNamespace(speaker_id="B", text="four", start=1.5, end=2.0),
        ]
        result = SimpleNamespace(words=words)

        segments = BatchTranscriber.parse_diarized_result(result)

        assert len(segments) == 4
        assert [s["speaker"] for s in segments] == ["A", "B", "A", "B"]
        assert [s["text"] for s in segments] == ["one", "two", "three", "four"]

    def test_no_words_no_text(self):
        """Result with neither words nor text returns empty list."""
        result = type("Result", (), {})()

        segments = BatchTranscriber.parse_diarized_result(result)

        assert segments == []
