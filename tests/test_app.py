"""Tests for the OpenMic TUI application."""

import json
import pytest
from unittest.mock import patch
from pathlib import Path

from openmic.app import (
    OpenMicApp, CommandInput, HELP_COMMANDS,
    _load_config, _save_config, CONFIG_FILE, THEMES,
)


class TestCommandInputPadding:
    """BUG-4: Command bar should have padding so text doesn't touch the border."""

    def test_command_input_css_has_padding(self):
        """Verify the app CSS includes padding for CommandInput."""
        css = OpenMicApp.CSS
        # The CSS should contain a padding rule for CommandInput
        assert "CommandInput" in css
        assert "padding" in css

    def test_command_input_padding_value(self):
        """Verify CommandInput has horizontal padding in the app CSS."""
        css = OpenMicApp.CSS
        # Extract the CommandInput block and check for padding
        # Find the CommandInput section
        start = css.index("CommandInput {")
        end = css.index("}", start) + 1
        command_input_css = css[start:end]
        assert "padding: 0 1" in command_input_css


class TestExitCommand:
    """FR-12: /exit command should be available."""

    def test_exit_in_help_commands(self):
        """Verify /exit is listed in the help commands."""
        commands = [cmd for cmd, _ in HELP_COMMANDS]
        assert "/exit" in commands

    def test_exit_help_description(self):
        """Verify /exit has a description."""
        for cmd, desc in HELP_COMMANDS:
            if cmd == "/exit":
                assert desc
                break
        else:
            pytest.fail("/exit not found in HELP_COMMANDS")


class TestHelpShortcut:
    """FR-15: Help should use Ctrl+? instead of bare ?."""

    def test_ctrl_question_mark_binding_exists(self):
        """Verify Ctrl+? binding is registered."""
        binding_keys = [b.key for b in OpenMicApp.BINDINGS]
        assert "ctrl+question_mark" in binding_keys

    def test_bare_question_mark_binding_removed(self):
        """Verify bare ? binding is no longer registered."""
        binding_keys = [b.key for b in OpenMicApp.BINDINGS]
        assert "question_mark" not in binding_keys

    def test_help_commands_shows_ctrl_question_mark(self):
        """Verify help commands list shows Ctrl+? not bare ?."""
        commands = [cmd for cmd, _ in HELP_COMMANDS]
        assert "Ctrl+?" in commands
        assert "?" not in commands


class TestThemePersistence:
    """FR-16: Theme auto-save and persistence."""

    def test_save_and_load_config(self, tmp_path):
        """Verify config can be saved and loaded."""
        config_file = tmp_path / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file), \
             patch("openmic.app.CONFIG_DIR", tmp_path):
            _save_config({"theme": "nord"})
            config = _load_config()
            assert config["theme"] == "nord"

    def test_load_config_missing_file(self, tmp_path):
        """Verify loading missing config returns empty dict."""
        config_file = tmp_path / "nonexistent" / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file):
            config = _load_config()
            assert config == {}

    def test_load_config_invalid_json(self, tmp_path):
        """Verify loading invalid JSON returns empty dict."""
        config_file = tmp_path / "settings.json"
        config_file.write_text("not valid json{{{")
        with patch("openmic.app.CONFIG_FILE", config_file):
            config = _load_config()
            assert config == {}

    def test_action_cycle_theme_persists(self, tmp_path):
        """Verify cycling theme saves to config."""
        config_file = tmp_path / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file), \
             patch("openmic.app.CONFIG_DIR", tmp_path):
            _save_config({"theme": "openmic"})
            # Simulate what action_cycle_theme does
            config = _load_config()
            names = [t.name for t in THEMES]
            idx = names.index(config["theme"])
            new_theme = names[(idx + 1) % len(names)]
            config["theme"] = new_theme
            _save_config(config)

            reloaded = _load_config()
            assert reloaded["theme"] == "nord"
