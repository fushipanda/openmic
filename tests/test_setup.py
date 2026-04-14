"""Tests for the setup wizard."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openmic.setup import (
    _get_required_keys,
    _prompt_provider,
    _save_setup,
    KEY_URLS,
    PROVIDER_DEPS,
    run_setup,
)


class TestGetRequiredKeys:
    """Test _get_required_keys returns correct keys per provider."""

    def test_anthropic_keys(self):
        keys = _get_required_keys("anthropic")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" not in env_keys  # transcription is local
        assert "OPENAI_API_KEY" in env_keys
        assert "ANTHROPIC_API_KEY" in env_keys
        assert len(keys) == 2

    def test_openai_keys(self):
        keys = _get_required_keys("openai")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" not in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "ANTHROPIC_API_KEY" not in env_keys
        assert len(keys) == 1

    def test_gemini_keys(self):
        keys = _get_required_keys("gemini")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" not in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "GEMINI_API_KEY" in env_keys
        assert len(keys) == 2

    def test_openrouter_keys(self):
        keys = _get_required_keys("openrouter")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" not in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "OPENROUTER_API_KEY" in env_keys
        assert len(keys) == 2

    def test_all_keys_have_labels(self):
        for provider in ["anthropic", "openai", "gemini", "openrouter"]:
            keys = _get_required_keys(provider)
            for env_key, label in keys:
                assert isinstance(env_key, str)
                assert isinstance(label, str)
                assert len(label) > 0


class TestProviderDeps:
    """Test PROVIDER_DEPS mapping is correct."""

    def test_anthropic_deps(self):
        deps = PROVIDER_DEPS["anthropic"]
        assert any("langchain-anthropic" in d for d in deps)
        assert any("langchain-openai" in d for d in deps)

    def test_openai_deps(self):
        deps = PROVIDER_DEPS["openai"]
        assert any("langchain-openai" in d for d in deps)
        assert len(deps) == 1

    def test_gemini_deps(self):
        deps = PROVIDER_DEPS["gemini"]
        assert any("langchain-google-genai" in d for d in deps)
        assert any("langchain-openai" in d for d in deps)

    def test_openrouter_deps(self):
        deps = PROVIDER_DEPS["openrouter"]
        assert any("langchain-openai" in d for d in deps)
        assert len(deps) == 1

    def test_all_providers_have_deps(self):
        for provider in ["anthropic", "openai", "gemini", "openrouter"]:
            assert provider in PROVIDER_DEPS
            assert len(PROVIDER_DEPS[provider]) > 0


class TestMainRouting:
    """Test main() routing logic."""

    @patch("openmic.app._load_config")
    def test_setup_subcommand_triggers_wizard(self, mock_config):
        """sys.argv setup should trigger run_setup."""
        with patch.object(sys, "argv", ["openmic", "setup"]):
            with patch("openmic.setup.run_setup") as mock_run:
                from openmic.app import main
                main()
                mock_run.assert_called_once()

    @patch("openmic.app._load_config")
    def test_first_run_triggers_wizard(self, mock_config):
        """No setup_complete in config should trigger wizard."""
        mock_config.return_value = {}
        with patch.object(sys, "argv", ["openmic"]):
            with patch("openmic.setup.run_setup") as mock_run:
                from openmic.app import main
                main()
                mock_run.assert_called()

    @patch("openmic.app._load_config")
    def test_setup_complete_skips_wizard(self, mock_config):
        """setup_complete: true should skip wizard and launch the app."""
        mock_config.return_value = {"setup_complete": True}
        with patch.object(sys, "argv", ["openmic"]):
            with patch("openmic.app.asyncio.run", side_effect=lambda coro: coro.close()) as mock_run, \
                 patch("openmic.app.print_banner"), \
                 patch("openmic.app._check_for_updates_sync"), \
                 patch("openmic.app.TranscriptRAG"):
                from openmic.app import main
                main()
                mock_run.assert_called_once()

    @patch("openmic.app._load_config")
    def test_cancelled_wizard_exits(self, mock_config):
        """If wizard doesn't set setup_complete, main should return without launching."""
        mock_config.return_value = {}
        with patch.object(sys, "argv", ["openmic"]):
            with patch("openmic.setup.run_setup"), \
                 patch("openmic.app.asyncio.run") as mock_run:
                from openmic.app import main
                main()
                # asyncio.run should NOT be called since setup wasn't completed
                mock_run.assert_not_called()


class TestKeysSkippedWhenPresent:
    """Test that keys already in environment are skipped."""

    def test_existing_key_skipped(self):
        """Keys already in os.environ should be auto-collected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            keys = _get_required_keys("openai")
            # The function returns what's required — the wizard skips
            # keys present in env during _prompt_api_keys()
            assert any(k == "OPENAI_API_KEY" for k, _ in keys)


class TestPromptProvider:
    """Test _prompt_provider CLI prompts."""

    def test_default_returns_anthropic(self):
        """Empty input selects default (anthropic)."""
        with patch("builtins.input", return_value=""):
            assert _prompt_provider() == "anthropic"

    def test_valid_choice(self):
        """Numeric input selects correct provider."""
        with patch("builtins.input", return_value="3"):
            assert _prompt_provider() == "gemini"

    def test_invalid_then_valid(self):
        """Bad input re-prompts until valid."""
        with patch("builtins.input", side_effect=["abc", "0", "2"]):
            assert _prompt_provider() == "openai"


class TestKeyUrls:
    """Test KEY_URLS coverage."""

    def test_all_keys_have_urls(self):
        """Every key from _get_required_keys for all providers has a URL."""
        for provider in ["anthropic", "openai", "gemini", "openrouter"]:
            for env_key, _ in _get_required_keys(provider):
                assert env_key in KEY_URLS, f"{env_key} missing from KEY_URLS"


class TestSaveSetup:
    """Test _save_setup writes config correctly."""

    @patch("openmic.setup._update_env_file")
    @patch("openmic.setup._save_config")
    @patch("openmic.setup._load_config", return_value={})
    def test_saves_provider_and_keys(self, mock_load, mock_save, mock_env):
        keys = {"OPENAI_API_KEY": "oai-key", "ANTHROPIC_API_KEY": "ant-key"}
        _save_setup("anthropic", keys)

        saved_config = mock_save.call_args[0][0]
        assert saved_config["setup_complete"] is True
        assert saved_config["llm_provider"] == "anthropic"
        assert "llm_model" in saved_config

        # Should write each key + LLM_PROVIDER to .env
        env_calls = [c[0] for c in mock_env.call_args_list]
        env_keys_written = [k for k, _ in env_calls]
        assert "OPENAI_API_KEY" in env_keys_written
        assert "ANTHROPIC_API_KEY" in env_keys_written
        assert "LLM_PROVIDER" in env_keys_written


class TestRunSetupInterrupt:
    """Test run_setup handles Ctrl+C gracefully."""

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_exits_cleanly(self, mock_input, capsys):
        run_setup()
        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()
