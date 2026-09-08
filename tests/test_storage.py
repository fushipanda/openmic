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




class TestDataDirLocation:
    """Data must resolve against the user, not the installed package.

    A non-editable install (pipx / uv tool / pip --user, i.e. every path in
    install.sh) puts storage.py inside site-packages. Deriving data dirs from
    __file__ therefore wrote transcripts into site-packages, where a tool
    upgrade deletes them.
    """

    def test_data_dir_not_derived_from_package_location(self):
        """Data dirs must not sit next to the installed package."""
        from openmic.storage import _data_dir
        import openmic.storage as storage_mod

        package_parent = Path(storage_mod.__file__).resolve().parent.parent
        assert _data_dir() != package_parent
        assert package_parent not in _data_dir().parents

    def test_default_is_xdg_local_share(self, monkeypatch):
        """With no overrides, data lives in ~/.local/share/openmic."""
        from openmic.storage import _data_dir

        monkeypatch.delenv("OPENMIC_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert _data_dir() == Path.home() / ".local" / "share" / "openmic"

    def test_xdg_data_home_is_honoured(self, monkeypatch, tmp_path):
        """XDG_DATA_HOME relocates the data dir."""
        from openmic.storage import _data_dir

        monkeypatch.delenv("OPENMIC_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert _data_dir() == tmp_path / "openmic"

    def test_openmic_data_dir_overrides_xdg(self, monkeypatch, tmp_path):
        """OPENMIC_DATA_DIR takes precedence over XDG_DATA_HOME."""
        from openmic.storage import _data_dir

        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("OPENMIC_DATA_DIR", str(tmp_path / "explicit"))
        assert _data_dir() == tmp_path / "explicit"

    def test_ensure_dirs_creates_missing_parents(self, monkeypatch, tmp_path):
        """A fresh install has no parent chain; ensure_dirs must build it."""
        nested = tmp_path / "no" / "such" / "tree"
        monkeypatch.setattr("openmic.storage.TRANSCRIPTS_DIR", nested / "transcripts")
        monkeypatch.setattr("openmic.storage.NOTES_DIR", nested / "notes")
        monkeypatch.setattr("openmic.storage.RECORDINGS_DIR", nested / "recordings")

        assert not nested.exists()
        ensure_dirs()
        assert (nested / "transcripts").is_dir()
        assert (nested / "notes").is_dir()
        assert (nested / "recordings").is_dir()
