"""Unit tests for storage.py: save/load transcripts and notes."""

from pathlib import Path
from unittest.mock import patch
import shutil

import pytest

from openmic.storage import (
    TRANSCRIPTS_DIR,
    NOTES_DIR,
    ensure_dirs,
    save_transcript,
    get_latest_transcript,
    list_transcripts,
    save_notes,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirect storage dirs to a temp directory for each test."""
    test_transcripts = tmp_path / "transcripts"
    test_notes = tmp_path / "notes"
    monkeypatch.setattr("openmic.storage.TRANSCRIPTS_DIR", test_transcripts)
    monkeypatch.setattr("openmic.storage.NOTES_DIR", test_notes)
    yield test_transcripts, test_notes


class TestEnsureDirs:
    def test_creates_directories(self, isolated_storage):
        transcripts_dir, notes_dir = isolated_storage
        assert not transcripts_dir.exists()
        assert not notes_dir.exists()

        ensure_dirs()

        assert transcripts_dir.is_dir()
        assert notes_dir.is_dir()

    def test_idempotent(self, isolated_storage):
        ensure_dirs()
        ensure_dirs()  # should not raise


class TestSaveTranscript:
    def test_basic_save(self, isolated_storage):
        transcripts_dir, _ = isolated_storage
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
        transcripts_dir, _ = isolated_storage
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        (transcripts_dir / "2025-01-01_10-00.md").write_text("old")
        (transcripts_dir / "2025-12-31_23-59.md").write_text("new")

        latest = get_latest_transcript()
        assert latest.name == "2025-12-31_23-59.md"

    def test_single_transcript(self, isolated_storage):
        transcripts_dir, _ = isolated_storage
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        (transcripts_dir / "2025-06-15_14-30.md").write_text("only one")

        latest = get_latest_transcript()
        assert latest.name == "2025-06-15_14-30.md"


class TestListTranscripts:
    def test_empty(self, isolated_storage):
        assert list_transcripts() == []

    def test_sorted_newest_first(self, isolated_storage):
        transcripts_dir, _ = isolated_storage
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
        _, notes_dir = isolated_storage
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
