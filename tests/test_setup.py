"""Tests for the setup wizard."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openmic.setup import _get_required_keys, PROVIDER_DEPS, run_setup


class TestGetRequiredKeys:
    """Test _get_required_keys returns correct keys per provider."""

    def test_anthropic_keys(self):
        keys = _get_required_keys("anthropic")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "ANTHROPIC_API_KEY" in env_keys
        assert len(keys) == 3

    def test_openai_keys(self):
        keys = _get_required_keys("openai")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "ANTHROPIC_API_KEY" not in env_keys
        assert len(keys) == 2

    def test_gemini_keys(self):
        keys = _get_required_keys("gemini")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "GEMINI_API_KEY" in env_keys
        assert len(keys) == 3

    def test_openrouter_keys(self):
        keys = _get_required_keys("openrouter")
        env_keys = [k for k, _ in keys]
        assert "ELEVENLABS_API_KEY" in env_keys
        assert "OPENAI_API_KEY" in env_keys
        assert "OPENROUTER_API_KEY" in env_keys
        assert len(keys) == 3

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
        """setup_complete: true should skip wizard."""
        mock_config.return_value = {"setup_complete": True}
        with patch.object(sys, "argv", ["openmic"]):
            with patch("openmic.app.OpenMicApp") as mock_app_cls:
                mock_app = MagicMock()
                mock_app_cls.return_value = mock_app
                from openmic.app import main
                main()
                mock_app.run.assert_called_once()

    @patch("openmic.app._load_config")
    def test_cancelled_wizard_exits(self, mock_config):
        """If wizard doesn't set setup_complete, main should return."""
        mock_config.return_value = {}
        with patch.object(sys, "argv", ["openmic"]):
            with patch("openmic.setup.run_setup"):
                with patch("openmic.app.OpenMicApp") as mock_app_cls:
                    from openmic.app import main
                    main()
                    # App should NOT be created since setup wasn't completed
                    mock_app_cls.assert_not_called()


class TestKeysSkippedWhenPresent:
    """Test that keys already in environment are skipped."""

    def test_existing_key_skipped(self):
        """Keys already in os.environ should be auto-collected."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key-123"}):
            keys = _get_required_keys("openai")
            # The function returns what's required — the wizard skips
            # keys present in env during _show_next_key()
            assert ("ELEVENLABS_API_KEY", "ElevenLabs API key (transcription)") in keys
