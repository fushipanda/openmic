"""Unit tests for storage.py: save/load transcripts and notes."""

from pathlib import Path
from unittest.mock import patch
import shutil

import pytest

from openmic.storage import (
    TRANSCRIPTS_DIR,
    NOTES_DIR,
    RECORDINGS_DIR,
    ensure_dirs,
    save_transcript,
    get_latest_transcript,
    list_transcripts,
    save_notes,
    format_transcript_title,
    list_recordings,
    delete_all_recordings,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirect storage dirs to a temp directory for each test."""
    test_transcripts = tmp_path / "transcripts"
    test_notes = tmp_path / "notes"
    test_recordings = tmp_path / "recordings"
    monkeypatch.setattr("openmic.storage.TRANSCRIPTS_DIR", test_transcripts)
    monkeypatch.setattr("openmic.storage.NOTES_DIR", test_notes)
    monkeypatch.setattr("openmic.storage.RECORDINGS_DIR", test_recordings)
    yield test_transcripts, test_notes, test_recordings


class TestEnsureDirs:
    def test_creates_directories(self, isolated_storage):
        transcripts_dir, notes_dir, recordings_dir = isolated_storage
        assert not transcripts_dir.exists()
        assert not notes_dir.exists()
        assert not recordings_dir.exists()

        ensure_dirs()

        assert transcripts_dir.is_dir()
        assert notes_dir.is_dir()
        assert recordings_dir.is_dir()

    def test_idempotent(self, isolated_storage):
        ensure_dirs()
        ensure_dirs()  # should not raise


class TestSaveTranscript:
    def test_basic_save(self, isolated_storage):
        transcripts_dir, _, _ = isolated_storage
        segments = [
            {"speaker": "Speaker 1", "text": "Hello there."},
            {"speaker": "Speaker 2", "text": "Hi, how are you?"},
        ]

        path = save_transcript(segments)

        assert path.exists()
        assert path.parent == transcripts_dir
        assert path.suffix == ".md"

        content = path.read_text()
        assert "# Meeting Transcript" in content
        assert "**Speaker 1:** Hello there." in content
        assert "**Speaker 2:** Hi, how are you?" in content

    def test_save_with_session_name(self, isolated_storage):
        segments = [{"speaker": "Speaker", "text": "Test"}]
        path = save_transcript(segments, session_name="standup")

        assert "standup" in path.name

    def test_save_without_session_name(self, isolated_storage):
        segments = [{"speaker": "Speaker", "text": "Test"}]
        path = save_transcript(segments)

        # Should be YYYY-MM-DD_HH-MM.md without extra suffix
        parts = path.stem.split("_")
        assert len(parts) == 2  # date and time

    def test_missing_speaker_key(self, isolated_storage):
        segments = [{"text": "No speaker here."}]
        path = save_transcript(segments)
        content = path.read_text()
        assert "**Speaker:** No speaker here." in content

    def test_empty_segments(self, isolated_storage):
        path = save_transcript([])
        content = path.read_text()
        assert "# Meeting Transcript" in content


class TestGetLatestTranscript:
    def test_no_transcripts(self, isolated_storage):
        assert get_latest_transcript() is None

    def test_returns_latest(self, isolated_storage):
        transcripts_dir, _, _ = isolated_storage
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        (transcripts_dir / "2025-01-01_10-00.md").write_text("old")
        (transcripts_dir / "2025-12-31_23-59.md").write_text("new")

        latest = get_latest_transcript()
        assert latest.name == "2025-12-31_23-59.md"

    def test_single_transcript(self, isolated_storage):
        transcripts_dir, _, _ = isolated_storage
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        (transcripts_dir / "2025-06-15_14-30.md").write_text("only one")

        latest = get_latest_transcript()
        assert latest.name == "2025-06-15_14-30.md"


class TestListTranscripts:
    def test_empty(self, isolated_storage):
        assert list_transcripts() == []

    def test_sorted_newest_first(self, isolated_storage):
        transcripts_dir, _, _ = isolated_storage
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        (transcripts_dir / "2025-01-01_10-00.md").write_text("a")
        (transcripts_dir / "2025-06-15_14-30.md").write_text("b")
        (transcripts_dir / "2025-12-31_23-59.md").write_text("c")

        result = list_transcripts()
        assert len(result) == 3
        assert result[0].name == "2025-12-31_23-59.md"
        assert result[-1].name == "2025-01-01_10-00.md"


class TestSaveNotes:
    def test_saves_notes(self, isolated_storage):
        _, notes_dir, _ = isolated_storage
        transcript_path = Path("transcripts/2025-06-15_14-30.md")

        path = save_notes("# Notes content", transcript_path)

        assert path.exists()
        assert path.parent == notes_dir
        assert path.name == "2025-06-15_14-30_notes.md"
        assert path.read_text() == "# Notes content"

    def test_notes_filename_from_transcript(self, isolated_storage):
        transcript_path = Path("transcripts/2025-01-01_10-00_standup.md")
        path = save_notes("content", transcript_path)
        assert path.name == "2025-01-01_10-00_standup_notes.md"


class TestFormatTranscriptTitle:
    """FR-20: Friendly transcript title formatting."""

    def test_without_session_name(self):
        """Title without session name uses 'Meeting Transcript'."""
        title = format_transcript_title("2025-06-15_14-30")
        assert "Meeting Transcript" in title
        assert "Jun 15" in title
        assert "2025" in title
        assert "2:30 PM" in title

    def test_with_session_name(self):
        """Title with session name uses the session name instead of 'Meeting Transcript'."""
        title = format_transcript_title("2025-06-15_14-30", "standup")
        assert "standup" in title
        assert "Meeting Transcript" not in title
        assert "Jun 15" in title

    def test_session_name_underscores_to_spaces(self):
        """Underscores in session names are converted to spaces."""
        title = format_transcript_title("2025-01-01_10-00", "team_standup")
        assert "team standup" in title

    def test_ordinal_suffixes(self):
        """Day ordinal suffixes are correct (1st, 2nd, 3rd, 4th, etc.)."""
        assert "1st" in format_transcript_title("2025-01-01_10-00")
        assert "2nd" in format_transcript_title("2025-01-02_10-00")
        assert "3rd" in format_transcript_title("2025-01-03_10-00")
        assert "4th" in format_transcript_title("2025-01-04_10-00")
        assert "11th" in format_transcript_title("2025-01-11_10-00")
        assert "12th" in format_transcript_title("2025-01-12_10-00")
        assert "21st" in format_transcript_title("2025-01-21_10-00")

    def test_invalid_timestamp_fallback(self):
        """Invalid timestamps fall back to the raw string."""
        title = format_transcript_title("not-a-date")
        assert "not-a-date" in title


class TestListRecordings:
    def test_empty(self, isolated_storage):
        assert list_recordings() == []

    def test_returns_wav_files(self, isolated_storage):
        _, _, recordings_dir = isolated_storage
        recordings_dir.mkdir(parents=True, exist_ok=True)
        (recordings_dir / "2026-01-01_10-00.wav").write_bytes(b"\x00" * 100)
        (recordings_dir / "2026-01-02_14-30.wav").write_bytes(b"\x00" * 200)

        result = list_recordings()
        assert len(result) == 2
        assert result[0].name == "2026-01-02_14-30.wav"

    def test_ignores_non_wav(self, isolated_storage):
        _, _, recordings_dir = isolated_storage
        recordings_dir.mkdir(parents=True, exist_ok=True)
        (recordings_dir / "2026-01-01_10-00.wav").write_bytes(b"\x00" * 100)
        (recordings_dir / "notes.txt").write_text("not a wav")

        result = list_recordings()
        assert len(result) == 1


class TestDeleteAllRecordings:
    def test_delete_all(self, isolated_storage):
        _, _, recordings_dir = isolated_storage
        recordings_dir.mkdir(parents=True, exist_ok=True)
        (recordings_dir / "a.wav").write_bytes(b"\x00" * 1024)
        (recordings_dir / "b.wav").write_bytes(b"\x00" * 2048)

        count, total_bytes = delete_all_recordings()
        assert count == 2
        assert total_bytes == 3072
        assert list_recordings() == []

    def test_delete_empty(self, isolated_storage):
        count, total_bytes = delete_all_recordings()
        assert count == 0
        assert total_bytes == 0
