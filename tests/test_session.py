"""Tests for openmic/session.py — append-only JSONL session storage."""

import json
import time
from pathlib import Path

import pytest

from openmic.session import (
    SESSIONS_DIR,
    append_notes,
    append_rename,
    append_title_update,
    append_transcript,
    create_session,
    display_title,
    get_session_meta,
    list_sessions,
    read_session,
    session_to_text,
)

_SEGMENTS = [
    {"speaker": "Speaker", "text": "Hello world", "start": 0.0, "end": 2.0},
    {"speaker": "Speaker", "text": "This is a test.", "start": 2.1, "end": 4.5},
]


@pytest.fixture(autouse=True)
def tmp_sessions_dir(tmp_path, monkeypatch):
    """Redirect SESSIONS_DIR to a temp directory for every test."""
    import openmic.session as sess_module
    monkeypatch.setattr(sess_module, "SESSIONS_DIR", tmp_path / "sessions")
    return tmp_path / "sessions"


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_named_session_creates_file(self, tmp_sessions_dir):
        path = create_session("standup")
        assert path.exists()
        assert path.suffix == ".jsonl"
        assert "standup" in path.name

    def test_unnamed_session_uses_timestamp(self, tmp_sessions_dir):
        path = create_session()
        assert path.exists()
        # Filename should match YYYY-MM-DD_HH-MM pattern
        stem = path.stem
        assert len(stem) == 16
        assert stem[4] == "-" and stem[7] == "-" and stem[10] == "_"

    def test_meta_entry_written(self, tmp_sessions_dir):
        path = create_session("weekly")
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "meta"
        assert entry["name"] == "weekly"
        assert "id" in entry
        assert "created" in entry

    def test_collision_appends_counter(self, tmp_sessions_dir):
        p1 = create_session("standup")
        p2 = create_session("standup")
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_name_sanitised(self, tmp_sessions_dir):
        path = create_session("hello world! / test")
        # Spaces → underscores, special chars stripped
        assert "/" not in path.name
        assert "!" not in path.name


# ---------------------------------------------------------------------------
# append_transcript
# ---------------------------------------------------------------------------

class TestAppendTranscript:
    def test_appends_transcript_entry(self, tmp_sessions_dir):
        path = create_session("test")
        append_transcript(path, _SEGMENTS, 4.5)
        lines = path.read_text().splitlines()
        assert len(lines) == 2  # meta + transcript
        entry = json.loads(lines[1])
        assert entry["type"] == "transcript"
        assert entry["duration_s"] == 4.5
        assert entry["segments"] == _SEGMENTS

    def test_transcript_has_id_and_timestamp(self, tmp_sessions_dir):
        path = create_session("test")
        append_transcript(path, _SEGMENTS, 10.0)
        entry = json.loads(path.read_text().splitlines()[1])
        assert "id" in entry
        assert "timestamp" in entry

    def test_multiple_recordings_append_in_order(self, tmp_sessions_dir):
        path = create_session("test")
        segs2 = [{"speaker": "Speaker", "text": "Second recording", "start": 0.0, "end": 3.0}]
        append_transcript(path, _SEGMENTS, 4.5)
        append_transcript(path, segs2, 3.0)
        lines = path.read_text().splitlines()
        # meta + 2 transcripts
        assert len(lines) == 3
        assert json.loads(lines[1])["segments"] == _SEGMENTS
        assert json.loads(lines[2])["segments"] == segs2


# ---------------------------------------------------------------------------
# append_notes
# ---------------------------------------------------------------------------

class TestAppendNotes:
    def test_appends_notes_entry(self, tmp_sessions_dir):
        path = create_session("test")
        append_notes(path, "# My Notes\n\nContent here.", "default")
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[1])
        assert entry["type"] == "notes"
        assert entry["template"] == "default"
        assert "# My Notes" in entry["content"]

    def test_notes_has_id_and_timestamp(self, tmp_sessions_dir):
        path = create_session("test")
        append_notes(path, "Notes content", "concise")
        entry = json.loads(path.read_text().splitlines()[1])
        assert "id" in entry
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# get_session_meta
# ---------------------------------------------------------------------------

class TestGetSessionMeta:
    def test_returns_meta_from_first_line(self, tmp_sessions_dir):
        path = create_session("mymeeting")
        meta = get_session_meta(path)
        assert meta["type"] == "meta"
        assert meta["name"] == "mymeeting"

    def test_fallback_on_bad_file(self, tmp_sessions_dir):
        tmp_sessions_dir.mkdir(exist_ok=True)
        bad = tmp_sessions_dir / "bad.jsonl"
        bad.write_text("not json\n")
        meta = get_session_meta(bad)
        assert meta["type"] == "meta"  # fallback dict returned


# ---------------------------------------------------------------------------
# read_session
# ---------------------------------------------------------------------------

class TestReadSession:
    def test_structure_keys(self, tmp_sessions_dir):
        path = create_session("test")
        data = read_session(path)
        assert "meta" in data
        assert "transcripts" in data
        assert "notes" in data

    def test_meta_populated(self, tmp_sessions_dir):
        path = create_session("mymeeting")
        data = read_session(path)
        assert data["meta"]["name"] == "mymeeting"

    def test_transcripts_populated(self, tmp_sessions_dir):
        path = create_session("test")
        append_transcript(path, _SEGMENTS, 4.5)
        data = read_session(path)
        assert len(data["transcripts"]) == 1
        assert data["transcripts"][0]["segments"] == _SEGMENTS

    def test_notes_populated(self, tmp_sessions_dir):
        path = create_session("test")
        append_notes(path, "Some notes", "default")
        data = read_session(path)
        assert len(data["notes"]) == 1
        assert data["notes"][0]["content"] == "Some notes"

    def test_multiple_recordings(self, tmp_sessions_dir):
        path = create_session("test")
        append_transcript(path, _SEGMENTS, 4.5)
        append_transcript(path, _SEGMENTS, 3.0)
        data = read_session(path)
        assert len(data["transcripts"]) == 2

    def test_missing_file_returns_empty(self, tmp_sessions_dir):
        missing = tmp_sessions_dir / "ghost.jsonl"
        data = read_session(missing)
        assert data["meta"] == {}
        assert data["transcripts"] == []
        assert data["notes"] == []
        assert data["autoTitle"] is None
        assert data["customTitle"] is None
        assert data["updatedAt"] is None
        assert data["lastTranscriptAt"] is None

    def test_invalid_lines_skipped(self, tmp_sessions_dir):
        path = create_session("test")
        with path.open("a") as f:
            f.write("not json at all\n")
        append_transcript(path, _SEGMENTS, 1.0)
        data = read_session(path)
        assert len(data["transcripts"]) == 1  # bad line skipped


# ---------------------------------------------------------------------------
# session_to_text
# ---------------------------------------------------------------------------

class TestSessionToText:
    def test_concatenates_segment_text(self, tmp_sessions_dir):
        path = create_session("test")
        append_transcript(path, _SEGMENTS, 4.5)
        text = session_to_text(path)
        assert "Hello world" in text
        assert "This is a test." in text

    def test_empty_session_returns_empty_string(self, tmp_sessions_dir):
        path = create_session("test")
        assert session_to_text(path) == ""

    def test_multiple_recordings_all_included(self, tmp_sessions_dir):
        path = create_session("test")
        segs2 = [{"speaker": "Speaker", "text": "Second chunk", "start": 0.0, "end": 2.0}]
        append_transcript(path, _SEGMENTS, 4.5)
        append_transcript(path, segs2, 2.0)
        text = session_to_text(path)
        assert "Hello world" in text
        assert "Second chunk" in text

    def test_speaker_prefix_included(self, tmp_sessions_dir):
        path = create_session("test")
        segs = [{"speaker": "Alice", "text": "Hi there", "start": 0.0, "end": 1.0}]
        append_transcript(path, segs, 1.0)
        text = session_to_text(path)
        assert text.startswith("Alice: Hi there")


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_returns_empty_when_no_sessions(self, tmp_sessions_dir):
        assert list_sessions() == []

    def test_lists_jsonl_files(self, tmp_sessions_dir):
        create_session("alpha")
        create_session("beta")
        sessions = list_sessions()
        assert len(sessions) == 2

    def test_sorted_newest_first(self, tmp_sessions_dir):
        p1 = create_session("first")
        time.sleep(0.05)
        p2 = create_session("second")
        sessions = list_sessions()
        assert sessions[0] == p2
        assert sessions[1] == p1

    def test_ignores_non_jsonl_files(self, tmp_sessions_dir):
        tmp_sessions_dir.mkdir(exist_ok=True)
        (tmp_sessions_dir / "stray.txt").write_text("not a session")
        create_session("real")
        sessions = list_sessions()
        assert len(sessions) == 1
        assert sessions[0].suffix == ".jsonl"


# ---------------------------------------------------------------------------
# create_session — slug field
# ---------------------------------------------------------------------------

class TestCreateSessionSlug:
    def test_slug_in_meta(self, tmp_sessions_dir):
        path = create_session("standup")
        meta = json.loads(path.read_text().splitlines()[0])
        assert "slug" in meta

    def test_slug_matches_filename_stem_for_named(self, tmp_sessions_dir):
        path = create_session("standup")
        meta = json.loads(path.read_text().splitlines()[0])
        assert meta["slug"] == path.stem

    def test_slug_is_timestamp_pattern_for_unnamed(self, tmp_sessions_dir):
        path = create_session()
        meta = json.loads(path.read_text().splitlines()[0])
        # YYYY-MM-DD_HH-MM
        assert len(meta["slug"]) == 16
        assert meta["slug"][4] == "-" and meta["slug"][7] == "-" and meta["slug"][10] == "_"

    def test_slug_unchanged_on_collision(self, tmp_sessions_dir):
        p1 = create_session("standup")
        p2 = create_session("standup")
        slug1 = json.loads(p1.read_text().splitlines()[0])["slug"]
        slug2 = json.loads(p2.read_text().splitlines()[0])["slug"]
        # Filenames differ due to collision suffix, slugs may match (base slug stored in meta)
        assert slug1 == "standup"
        assert slug2 == "standup"
        assert p1 != p2


# ---------------------------------------------------------------------------
# append_title_update
# ---------------------------------------------------------------------------

class TestAppendTitleUpdate:
    def test_appends_title_update_entry(self, tmp_sessions_dir):
        path = create_session("test")
        append_title_update(path, "Quarterly budget review", "anthropic/claude-3-5-haiku")
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[1])
        assert entry["type"] == "title_update"
        assert entry["autoTitle"] == "Quarterly budget review"
        assert entry["model"] == "anthropic/claude-3-5-haiku"

    def test_appends_title_update_has_id_and_timestamp(self, tmp_sessions_dir):
        path = create_session("test")
        append_title_update(path, "Sprint planning kick-off", "openai/gpt-4o-mini")
        entry = json.loads(path.read_text().splitlines()[1])
        assert "id" in entry
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# append_rename
# ---------------------------------------------------------------------------

class TestAppendRename:
    def test_appends_rename_entry(self, tmp_sessions_dir):
        path = create_session("test")
        append_rename(path, "My Custom Title")
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[1])
        assert entry["type"] == "rename"
        assert entry["customTitle"] == "My Custom Title"

    def test_appends_rename_has_id_and_timestamp(self, tmp_sessions_dir):
        path = create_session("test")
        append_rename(path, "Another Title")
        entry = json.loads(path.read_text().splitlines()[1])
        assert "id" in entry
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# display_title
# ---------------------------------------------------------------------------

class TestDisplayTitle:
    def _make_data(self, *, slug="my-session", auto_title=None, custom_title=None):
        return {
            "meta": {"slug": slug, "id": "abc-123"},
            "autoTitle": auto_title,
            "customTitle": custom_title,
        }

    def test_custom_over_auto(self):
        data = self._make_data(auto_title="AI Title", custom_title="Custom Title")
        assert display_title(data) == "Custom Title"

    def test_auto_over_slug(self):
        data = self._make_data(slug="my-session", auto_title="AI Title")
        assert display_title(data) == "AI Title"

    def test_fallback_to_slug(self):
        data = self._make_data(slug="my-session")
        assert display_title(data) == "my-session"

    def test_fallback_to_id_when_no_slug(self):
        data = {"meta": {"id": "abc-123"}, "autoTitle": None, "customTitle": None}
        assert display_title(data) == "abc-123"


# ---------------------------------------------------------------------------
# read_session — title and computed timestamp fields
# ---------------------------------------------------------------------------

class TestReadSessionTitleFields:
    def test_auto_title_from_title_update(self, tmp_sessions_dir):
        path = create_session("test")
        append_title_update(path, "Sprint review session", "anthropic/claude")
        data = read_session(path)
        assert data["autoTitle"] == "Sprint review session"

    def test_custom_title_from_rename(self, tmp_sessions_dir):
        path = create_session("test")
        append_rename(path, "My Custom Name")
        data = read_session(path)
        assert data["customTitle"] == "My Custom Name"

    def test_last_write_wins_for_auto_title(self, tmp_sessions_dir):
        path = create_session("test")
        append_title_update(path, "First title", "model-a")
        append_title_update(path, "Second title", "model-b")
        data = read_session(path)
        assert data["autoTitle"] == "Second title"

    def test_last_write_wins_for_rename(self, tmp_sessions_dir):
        path = create_session("test")
        append_rename(path, "First name")
        append_rename(path, "Second name")
        data = read_session(path)
        assert data["customTitle"] == "Second name"

    def test_updatedAt_is_mtime(self, tmp_sessions_dir):
        path = create_session("test")
        data = read_session(path)
        assert data["updatedAt"] == pytest.approx(path.stat().st_mtime, abs=1.0)

    def test_lastTranscriptAt_from_last_entry(self, tmp_sessions_dir):
        path = create_session("test")
        segs = [{"speaker": "Alice", "text": "Hello", "start": 0.0, "end": 1.0}]
        append_transcript(path, segs, 1.0)
        data = read_session(path)
        assert data["lastTranscriptAt"] == data["transcripts"][-1]["timestamp"]

    def test_no_transcripts_lastTranscriptAt_is_none(self, tmp_sessions_dir):
        path = create_session("test")
        data = read_session(path)
        assert data["lastTranscriptAt"] is None
