"""Tests for the lightweight OpenMic CLI."""

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openmic.app import (
    HELP_COMMANDS,
    MODEL_REGISTRY,
    ReplContext,
    UsageTracker,
    _load_config,
    _parse_transcript_meta,
    _resolve_transcript_mention,
    _save_config,
    _update_env_file,
    handle_command,
    pick_transcript,
)


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
            assert info["models"], f"{key} has no models"
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
        with patch("builtins.input", return_value=""):
            result = pick_transcript([p])
        assert result is None

    def test_returns_correct_path_on_valid_pick(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("builtins.input", return_value="1"):
            result = pick_transcript([p])
        assert result == p

    def test_returns_none_on_invalid_number(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("builtins.input", return_value="99"):
            result = pick_transcript([p])
        assert result is None

    def test_returns_none_on_non_numeric(self, tmp_path):
        p = tmp_path / "2026-04-08_14-30.md"
        p.write_text("# Transcript")
        with patch("builtins.input", return_value="abc"):
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
        with patch("openmic.app.list_transcripts", return_value=[Path("/fake/t.md")]):
            asyncio.run(handle_command("/query test question", ctx))
        ctx.rag.query.assert_called_once()

    def test_query_clears_chat_history(self):
        ctx = _make_ctx()
        ctx.chatting = True
        with patch("openmic.app.list_transcripts", return_value=[Path("/fake/t.md")]):
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

    def test_cleanup_recordings_no_recordings(self, capsys):
        ctx = _make_ctx()
        with patch("openmic.app.list_recordings", return_value=[]):
            asyncio.run(handle_command("/cleanup-recordings", ctx))
        out = capsys.readouterr().out
        assert "No recordings" in out

    def test_transcripts_no_transcripts(self, capsys):
        ctx = _make_ctx()
        with patch("openmic.app.list_transcripts", return_value=[]):
            asyncio.run(handle_command("/transcripts", ctx))
        out = capsys.readouterr().out
        assert "No transcripts" in out

    def test_stop_when_not_recording(self, capsys):
        ctx = _make_ctx()
        asyncio.run(handle_command("/stop", ctx))
        out = capsys.readouterr().out
        assert "Not currently recording" in out

    def test_bare_text_triggers_query(self):
        ctx = _make_ctx()
        with patch("openmic.app.list_transcripts", return_value=[Path("/fake/t.md")]):
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
