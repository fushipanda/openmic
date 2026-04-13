"""Tests for openmic/session.py — append-only JSONL session storage."""

import json
import time
from pathlib import Path

import pytest

from openmic.session import (
    SESSIONS_DIR,
    append_notes,
    append_transcript,
    create_session,
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
        assert data == {"meta": {}, "transcripts": [], "notes": []}

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
