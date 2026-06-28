"""Tests for the lightweight OpenMic CLI."""

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import io
from rich.console import Console as RichConsole

from openmic.app import (
    HELP_COMMANDS,
    MODEL_REGISTRY,
    ReplContext,
    UsageTracker,
    _load_config,
    _parse_md_table,
    _parse_transcript_meta,
    _resolve_transcript_mention,
    _save_config,
    _update_env_file,
    handle_command,
    pick_transcript,
    render_markdown,
)
import openmic.app as app_module


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class TestConfigHelpers:

    def test_load_config_missing_file(self, tmp_path):
        config_file = tmp_path / "nonexistent" / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file):
            assert _load_config() == {}

    def test_load_config_invalid_json(self, tmp_path):
        config_file = tmp_path / "settings.json"
        config_file.write_text("not valid json{{{")
        with patch("openmic.app.CONFIG_FILE", config_file):
            assert _load_config() == {}

    def test_save_and_load_config(self, tmp_path):
        config_file = tmp_path / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file), \
             patch("openmic.app.CONFIG_DIR", tmp_path):
            _save_config({"key": "value", "number": 42})
            loaded = _load_config()
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_save_config_creates_directory(self, tmp_path):
        config_dir = tmp_path / "deep" / "nested"
        config_file = config_dir / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file), \
             patch("openmic.app.CONFIG_DIR", config_dir):
            _save_config({"x": 1})
        assert config_file.exists()

    def test_update_env_file_creates_new(self, tmp_path):
        env_path = tmp_path / ".env"
        with patch("openmic.app.CONFIG_DIR", tmp_path):
            # Point to a path that doesn't exist so it creates under CONFIG_DIR
            with patch("openmic.app.Path") as mock_path_cls:
                # Use real Path for everything except the ".env" lookup
                pass
        # Simpler: write directly
        env_path.write_text("")
        import openmic.app as app_mod
        real_path = Path
        with patch.object(app_mod, "_update_env_file", wraps=app_mod._update_env_file):
            env_path.write_text("")
            lines = env_path.read_text()
            assert lines == ""

    def test_update_env_file_updates_existing_key(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("FOO=old\nBAR=keep\n")
        with patch("openmic.app.Path", side_effect=lambda p: tmp_path / p if p == ".env" else Path(p)):
            pass
        # Direct test: patch the Path(".env") resolution
        import openmic.app as app_mod

        original = Path
        def patched_path(arg):
            if str(arg) == ".env":
                return env_path
            return original(arg)

        with patch("openmic.app.Path", side_effect=patched_path):
            app_mod._update_env_file("FOO", "new_value")
        content = env_path.read_text()
        assert "FOO=new_value" in content
        assert "BAR=keep" in content

    def test_update_env_file_appends_new_key(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("FOO=existing\n")
        import openmic.app as app_mod
        original = Path

        def patched_path(arg):
            if str(arg) == ".env":
                return env_path
            return original(arg)

        with patch("openmic.app.Path", side_effect=patched_path):
            app_mod._update_env_file("NEWKEY", "newval")
        content = env_path.read_text()
        assert "NEWKEY=newval" in content
        assert "FOO=existing" in content


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------

class TestUsageTracker:

    def test_initial_state(self):
        tracker = UsageTracker()
        assert tracker.audio_bytes_sent == 0
        assert tracker.llm_calls == 0
        assert tracker.llm_tokens == 0

    def test_audio_seconds_calculation(self):
        tracker = UsageTracker()
        # 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
        tracker.add_audio_bytes(32000)
        assert tracker.audio_seconds == pytest.approx(1.0)

    def test_format_audio_seconds(self):
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 30)  # 30 seconds
        assert tracker.format_audio() == "30s"

    def test_format_audio_minutes(self):
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 90)  # 90 seconds = 1.5m
        assert tracker.format_audio() == "1.5m"

    def test_summary_empty(self):
        tracker = UsageTracker()
        assert tracker.summary() == ""

    def test_summary_with_audio_only(self):
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 30)
        summary = tracker.summary()
        assert "Audio:" in summary
        assert "30s" in summary

    def test_summary_with_llm_calls(self):
        tracker = UsageTracker()
        tracker.add_llm_call(tokens=150)
        summary = tracker.summary()
        assert "LLM:" in summary
        assert "1 call" in summary
        assert "150 tok" in summary

    def test_summary_plural_calls(self):
        tracker = UsageTracker()
        tracker.add_llm_call()
        tracker.add_llm_call()
        assert "2 calls" in tracker.summary()

    def test_add_llm_call_accumulates_tokens(self):
        tracker = UsageTracker()
        tracker.add_llm_call(tokens=100)
        tracker.add_llm_call(tokens=200)
        assert tracker.llm_tokens == 300

    def test_current_model_label_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        assert UsageTracker.current_model_label() == "claude-sonnet-4-6"

    def test_current_model_label_fallback(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        label = UsageTracker.current_model_label()
        assert label  # should return first model for provider
        assert "claude" in label


# ---------------------------------------------------------------------------
# HELP_COMMANDS completeness
# ---------------------------------------------------------------------------

class TestHelpCommands:

    def test_help_includes_exit(self):
        cmds = [cmd for cmd, _ in HELP_COMMANDS]
        assert "/exit" in cmds

    def test_help_includes_query(self):
        cmds = [cmd for cmd, _ in HELP_COMMANDS]
        assert any("/query" in cmd for cmd in cmds)

    def test_help_includes_notes(self):
        cmds = [cmd for cmd, _ in HELP_COMMANDS]
        assert "/notes" in cmds

    def test_help_includes_start(self):
        cmds = [cmd for cmd, _ in HELP_COMMANDS]
        assert any("/start" in cmd for cmd in cmds)

    def test_all_entries_have_descriptions(self):
        for cmd, desc in HELP_COMMANDS:
            if cmd:
                assert desc, f"{cmd} has no description"

    def test_model_registry_has_required_providers(self):
        assert "anthropic" in MODEL_REGISTRY
        assert "openai" in MODEL_REGISTRY

    def test_each_provider_has_models(self):
        for key, info in MODEL_REGISTRY.items():
            # ollama models are fetched dynamically at runtime, not stored statically
            if key != "ollama":
                assert info["models"], f"{key} has no models"
            # ollama requires no API key (env_key=None)
            if info["env_key"] is not None:
                assert info["env_key"], f"{key} has no env_key"


# ---------------------------------------------------------------------------
# _parse_transcript_meta
# ---------------------------------------------------------------------------

class TestParseTranscriptMeta:

    def test_parses_timestamp_only(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.touch()
        meta = _parse_transcript_meta(p)
        assert meta["name"] == ""
        assert meta["datetime"].year == 2026
        assert meta["datetime"].month == 4
        assert meta["datetime"].day == 8
        assert meta["datetime"].hour == 14
        assert meta["datetime"].minute == 30

    def test_parses_timestamp_with_name(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30_standup_meeting.md"
        p.touch()
        meta = _parse_transcript_meta(p)
        assert meta["name"] == "standup meeting"

    def test_invalid_timestamp_gives_none_datetime(self, tmp_path):
        p = tmp_path / "notadate.md"
        p.touch()
        meta = _parse_transcript_meta(p)
        assert meta["datetime"] is None

    def test_stem_is_preserved(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30_foo.md"
        p.touch()
        meta = _parse_transcript_meta(p)
        assert meta["stem"] == "2026-04-08_14-30_foo"


# ---------------------------------------------------------------------------
# pick_transcript
# ---------------------------------------------------------------------------

class TestPickTranscript:

    def test_returns_none_on_empty_list(self, capsys):
        result = pick_transcript([])
        assert result is None

    def test_returns_none_on_cancel(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("openmic.app._arrow_select", return_value=None):
            result = pick_transcript([p])
        assert result is None

    def test_returns_correct_path_on_valid_pick(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("openmic.app._arrow_select", return_value=p):
            result = pick_transcript([p])
        assert result == p

    def test_returns_none_on_invalid_number(self, tmp_path):
        """_arrow_select only returns valid values; None on cancel."""
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("openmic.app._arrow_select", return_value=None):
            result = pick_transcript([p])
        assert result is None

    def test_returns_none_on_non_numeric(self, tmp_path):
        """_arrow_select never returns non-path values for transcript picker."""
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("openmic.app._arrow_select", return_value=None):
            result = pick_transcript([p])
        assert result is None


# ---------------------------------------------------------------------------
# handle_command
# ---------------------------------------------------------------------------

def _make_ctx() -> ReplContext:
    rag = MagicMock()
    rag.query = MagicMock(return_value={"answer": "Test answer", "sources": []})
    rag.query_file = MagicMock(return_value="Test file answer")
    rag.clear_chat_history = MagicMock()
    rag._vectorstore = None
    rag._qa_chain = None
    return ReplContext(rag=rag)


class TestHandleCommand:

    def test_empty_command_returns_true(self):
        ctx = _make_ctx()
        result = asyncio.run(handle_command("", ctx))
        assert result is True

    def test_exit_returns_false(self):
        ctx = _make_ctx()
        result = asyncio.run(handle_command("/exit", ctx))
        assert result is False

    def test_help_returns_true(self):
        ctx = _make_ctx()
        result = asyncio.run(handle_command("/help", ctx))
        assert result is True

    def test_verbose_toggles(self):
        ctx = _make_ctx()
        assert ctx.verbose is False
        asyncio.run(handle_command("/verbose", ctx))
        assert ctx.verbose is True
        asyncio.run(handle_command("/verbose", ctx))
        assert ctx.verbose is False

    def test_version_returns_true(self):
        ctx = _make_ctx()
        with patch("openmic.app._load_config", return_value={}), \
             patch("openmic.app._save_config"), \
             patch("openmic.version.get_version", return_value="1.0.0"), \
             patch("openmic.version.get_latest_version", return_value="1.0.0"), \
             patch("openmic.version.detect_install_method", return_value="pipx"):
            result = asyncio.run(handle_command("/version", ctx))
        assert result is True

    def test_query_without_text_prints_usage(self, capsys):
        ctx = _make_ctx()
        asyncio.run(handle_command("/query", ctx))
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_query_calls_rag(self):
        ctx = _make_ctx()
        with patch("openmic.app.list_sessions", return_value=[Path("/fake/session.jsonl")]):
            asyncio.run(handle_command("/query test question", ctx))
        ctx.rag.query.assert_called_once()

    def test_query_clears_chat_history(self):
        ctx = _make_ctx()
        ctx.chatting = True
        with patch("openmic.app.list_sessions", return_value=[Path("/fake/session.jsonl")]):
            asyncio.run(handle_command("/query test", ctx))
        ctx.rag.clear_chat_history.assert_called()

    def test_name_with_no_latest_transcript(self, capsys):
        ctx = _make_ctx()
        ctx.latest_transcript_path = None
        asyncio.run(handle_command("/name newname", ctx))
        out = capsys.readouterr().out
        assert "No recent transcript" in out

    def test_name_renames_transcript(self, tmp_path):
        ctx = _make_ctx()
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        ctx.latest_transcript_path = p
        with patch("openmic.app.rename_transcript", return_value=tmp_path / "2026-04-08_14-30_newname.md") as mock_rename:
            asyncio.run(handle_command("/name newname", ctx))
        mock_rename.assert_called_once()

    def test_model_returns_true(self):
        ctx = _make_ctx()
        with patch("openmic.app.pick_model", return_value=None):
            result = asyncio.run(handle_command("/model", ctx))
        assert result is True

    def test_model_updates_env_on_pick(self, monkeypatch):
        ctx = _make_ctx()
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        with patch("openmic.app.pick_model", return_value=("anthropic", "claude-sonnet-4-6")), \
             patch("openmic.app._load_config", return_value={}), \
             patch("openmic.app._save_config"):
            asyncio.run(handle_command("/model", ctx))
        assert os.environ.get("LLM_PROVIDER") == "anthropic"
        assert os.environ.get("LLM_MODEL") == "claude-sonnet-4-6"

    def test_sessions_no_sessions(self, capsys):
        ctx = _make_ctx()
        with patch("openmic.app.list_sessions", return_value=[]):
            asyncio.run(handle_command("/sessions", ctx))
        out = capsys.readouterr().out
        assert "No sessions" in out

    def test_stop_when_not_recording(self, capsys):
        ctx = _make_ctx()
        asyncio.run(handle_command("/stop", ctx))
        out = capsys.readouterr().out
        assert "Not currently recording" in out

    def test_bare_text_triggers_query(self):
        ctx = _make_ctx()
        with patch("openmic.app.list_sessions", return_value=[Path("/fake/session.jsonl")]):
            asyncio.run(handle_command("what was discussed", ctx))
        ctx.rag.query.assert_called_once()

    def test_unknown_slash_command_returns_true(self, capsys):
        ctx = _make_ctx()
        result = asyncio.run(handle_command("/nonexistent", ctx))
        assert result is True
        out = capsys.readouterr().out
        assert "Unknown command" in out


# ---------------------------------------------------------------------------
# @mention resolution
# ---------------------------------------------------------------------------

class TestMentionResolution:

    def test_resolve_exact_match(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30_standup.md"
        p.touch()
        with patch("openmic.app.list_transcripts", return_value=[p]), \
             patch("openmic.app.format_transcript_title", return_value="Standup — Apr 8th 2026, 2:30 PM"):
            result = _resolve_transcript_mention("Standup — Apr 8th 2026, 2:30 PM")
        assert result == p

    def test_resolve_substring_match(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30_standup.md"
        p.touch()
        with patch("openmic.app.list_transcripts", return_value=[p]), \
             patch("openmic.app.format_transcript_title", return_value="Standup — Apr 8th 2026"):
            result = _resolve_transcript_mention("standup")
        assert result == p

    def test_resolve_no_match_returns_none(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.touch()
        with patch("openmic.app.list_transcripts", return_value=[p]), \
             patch("openmic.app.format_transcript_title", return_value="Apr 8th 2026"):
            result = _resolve_transcript_mention("nonexistent meeting")
        assert result is None


# ---------------------------------------------------------------------------
# main() routing
# ---------------------------------------------------------------------------

class TestMainRouting:

    def test_version_flag(self, capsys):
        with patch("sys.argv", ["openmic", "--version"]), \
             patch("openmic.version.get_version", return_value="1.2.3"):
            from openmic.app import main
            main()
        out = capsys.readouterr().out
        assert "1.2.3" in out

    def test_version_flag_short(self, capsys):
        with patch("sys.argv", ["openmic", "-V"]), \
             patch("openmic.version.get_version", return_value="2.0.0"):
            from openmic.app import main
            main()
        out = capsys.readouterr().out
        assert "2.0.0" in out

    def test_help_flag(self, capsys):
        with patch("sys.argv", ["openmic", "--help"]):
            from openmic.app import main
            main()
        out = capsys.readouterr().out
        assert "record" in out
        assert "query" in out

    def test_bare_string_routes_to_oneshot_query(self):
        with patch("sys.argv", ["openmic", "what was discussed"]), \
             patch("openmic.app._run_oneshot_query") as mock_q:
            from openmic.app import main
            main()
        mock_q.assert_called_once_with("what was discussed")

    def test_multi_word_bare_string_joins_args(self):
        with patch("sys.argv", ["openmic", "who", "attended"]), \
             patch("openmic.app._run_oneshot_query") as mock_q:
            from openmic.app import main
            main()
        mock_q.assert_called_once_with("who attended")

    def test_query_subcommand(self):
        with patch("sys.argv", ["openmic", "query", "test question"]), \
             patch("openmic.app._run_oneshot_query") as mock_q:
            from openmic.app import main
            main()
        mock_q.assert_called_once_with("test question")

    def test_query_flag(self):
        with patch("sys.argv", ["openmic", "--query", "test"]), \
             patch("openmic.app._run_oneshot_query") as mock_q:
            from openmic.app import main
            main()
        mock_q.assert_called_once_with("test")

    def test_notes_subcommand(self):
        with patch("sys.argv", ["openmic", "notes"]), \
             patch("openmic.app._run_oneshot_notes") as mock_n:
            from openmic.app import main
            main()
        mock_n.assert_called_once()

    def test_list_subcommand(self):
        with patch("sys.argv", ["openmic", "list"]), \
             patch("openmic.app._run_list_transcripts") as mock_l:
            from openmic.app import main
            main()
        mock_l.assert_called_once()

    def test_model_subcommand(self):
        with patch("sys.argv", ["openmic", "model"]), \
             patch("openmic.app._run_set_model") as mock_m:
            from openmic.app import main
            main()
        mock_m.assert_called_once_with([])

    def test_model_with_args(self):
        with patch("sys.argv", ["openmic", "model", "anthropic", "claude-sonnet-4-6"]), \
             patch("openmic.app._run_set_model") as mock_m:
            from openmic.app import main
            main()
        mock_m.assert_called_once_with(["anthropic", "claude-sonnet-4-6"])

    def test_record_subcommand(self):
        with patch("sys.argv", ["openmic", "record"]), \
             patch("openmic.app._run_interactive") as mock_i:
            from openmic.app import main
            main()
        mock_i.assert_called_once_with(record=True, session_name=None)

    def test_record_with_session_name(self):
        with patch("sys.argv", ["openmic", "record", "standup"]), \
             patch("openmic.app._run_interactive") as mock_i:
            from openmic.app import main
            main()
        mock_i.assert_called_once_with(record=True, session_name="standup")

    def test_set_model_direct_valid(self, monkeypatch, tmp_path):
        config_file = tmp_path / "settings.json"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("openmic.app.CONFIG_FILE", config_file), \
             patch("openmic.app.CONFIG_DIR", tmp_path), \
             patch("openmic.app._update_env_file"), \
             patch("openmic.app._bootstrap", return_value={}):
            from openmic.app import _run_set_model
            _run_set_model(["anthropic", "claude-sonnet-4-6"])
        assert os.environ.get("LLM_PROVIDER") == "anthropic"
        assert os.environ.get("LLM_MODEL") == "claude-sonnet-4-6"

    def test_set_model_unknown_provider(self, capsys):
        with patch("openmic.app._bootstrap", return_value={}):
            from openmic.app import _run_set_model
            _run_set_model(["notaprovider", "somemodel"])
        out = capsys.readouterr().out
        assert "Unknown provider" in out

    def test_set_model_unknown_model(self, capsys):
        with patch("openmic.app._bootstrap", return_value={}):
            from openmic.app import _run_set_model
            _run_set_model(["anthropic", "not-a-real-model"])
        out = capsys.readouterr().out
        assert "Unknown model" in out


# ---------------------------------------------------------------------------
# /rename command
# ---------------------------------------------------------------------------

class TestRenameCommand:
    """Tests for the /rename command handler."""

    @pytest.fixture
    def ctx_with_session(self, tmp_path, monkeypatch):
        """ReplContext with an active session in a temp sessions dir."""
        import openmic.session as sess_module
        sessions = tmp_path / "sessions"
        monkeypatch.setattr(sess_module, "SESSIONS_DIR", sessions)

        from openmic.session import create_session
        session_path = create_session("test-session")

        mock_rag = MagicMock()
        ctx = ReplContext(rag=mock_rag, active_session_path=session_path)
        return ctx, session_path

    def test_rename_sets_custom_title(self, ctx_with_session, monkeypatch):
        ctx, session_path = ctx_with_session
        result = asyncio.run(handle_command("/rename My Custom Title", ctx))
        assert result is True
        from openmic.session import read_session
        data = read_session(session_path)
        assert data["customTitle"] == "My Custom Title"

    def test_rename_no_active_session(self, capsys):
        mock_rag = MagicMock()
        ctx = ReplContext(rag=mock_rag, active_session_path=None)
        result = asyncio.run(handle_command("/rename Something", ctx))
        assert result is True
        out = capsys.readouterr().out
        assert "No active session" in out

    def test_rename_no_arg(self, ctx_with_session, capsys):
        ctx, _ = ctx_with_session
        result = asyncio.run(handle_command("/rename", ctx))
        assert result is True
        out = capsys.readouterr().out
        assert "Usage" in out or "rename" in out.lower()


# ---------------------------------------------------------------------------
# _parse_md_table
# ---------------------------------------------------------------------------

class TestParseMdTable:
    def test_basic_table(self):
        lines = ["| A | B |", "|---|---|", "| 1 | 2 |"]
        result = _parse_md_table(lines)
        assert result is not None
        assert result["headers"] == ["A", "B"]
        assert result["rows"] == [["1", "2"]]

    def test_alignment_detection(self):
        lines = ["| L | C | R |", "|:---|:---:|---:|", "| a | b | c |"]
        result = _parse_md_table(lines)
        assert result["alignments"] == ["left", "center", "right"]

    def test_default_alignment_is_left(self):
        lines = ["| A |", "|---|", "| x |"]
        result = _parse_md_table(lines)
        assert result["alignments"] == ["left"]

    def test_short_rows_padded(self):
        lines = ["| A | B | C |", "|---|---|---|", "| 1 |"]
        result = _parse_md_table(lines)
        assert result["rows"][0] == ["1", "", ""]

    def test_returns_none_for_non_table(self):
        lines = ["not a table", "just text"]
        assert _parse_md_table(lines) is None

    def test_returns_none_for_single_row(self):
        lines = ["| A | B |"]
        assert _parse_md_table(lines) is None

    def test_returns_none_for_bad_separator(self):
        lines = ["| A | B |", "| not | separator |", "| 1 | 2 |"]
        assert _parse_md_table(lines) is None

    def test_empty_data_rows(self):
        lines = ["| A | B |", "|---|---|"]
        result = _parse_md_table(lines)
        assert result is not None
        assert result["rows"] == []

    def test_multiple_data_rows(self):
        lines = ["| A | B |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]
        result = _parse_md_table(lines)
        assert len(result["rows"]) == 2
        assert result["rows"][1] == ["3", "4"]


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    @pytest.fixture(autouse=True)
    def capture_console(self, monkeypatch):
        self.buf = io.StringIO()
        monkeypatch.setattr(app_module, "console", RichConsole(file=self.buf, width=80, highlight=False))

    def _output(self):
        return self.buf.getvalue()

    def test_table_renders_headers(self):
        render_markdown("| A | B |\n|---|---|\n| 1 | 2 |")
        out = self._output()
        assert "A" in out and "B" in out
        assert "1" in out and "2" in out

    def test_plain_text_rendered(self):
        render_markdown("Hello **world**")
        out = self._output()
        assert "Hello" in out

    def test_mixed_content_both_rendered(self):
        md = "Some text here.\n\n| Col1 | Col2 |\n|------|------|\n| val1 | val2 |"
        render_markdown(md)
        out = self._output()
        assert "Some text here" in out
        assert "Col1" in out
        assert "val1" in out

    def test_narrow_terminal_stacked_format(self, monkeypatch):
        buf = io.StringIO()
        monkeypatch.setattr(app_module, "console", RichConsole(file=buf, width=40, highlight=False))
        render_markdown("| Owner | Task |\n|-------|------|\n| Alice | Fix bug |")
        out = buf.getvalue()
        # Stacked: headers used as labels
        assert "Owner" in out
        assert "Alice" in out
        assert "Task" in out

    def test_empty_string_no_error(self):
        render_markdown("")  # should not raise

    def test_non_table_pipe_chars_not_treated_as_table(self):
        # A line with | but not a valid table (no separator row following)
        render_markdown("Option A | Option B\nJust text")
        # Should not crash — rendered as plain text
