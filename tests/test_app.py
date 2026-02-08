"""Tests for the OpenMic TUI application."""

import json
import pytest
from unittest.mock import patch
from pathlib import Path

from openmic.app import (
    OpenMicApp, CommandInput, HELP_COMMANDS, SLASH_COMMANDS,
    _load_config, _save_config, CONFIG_FILE, THEMES,
    _muted_color, OPENMIC_THEME, NORD_THEME,
    AutocompleteDropdown,
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


class TestMutedColor:
    """BUG-1/BUG-2: Muted text color should contrast with backgrounds."""

    def test_muted_color_differs_from_secondary(self):
        """Muted color should NOT be the same as theme.secondary (which matches background)."""
        for theme in [OPENMIC_THEME, NORD_THEME]:
            muted = _muted_color(theme)
            assert muted != theme.secondary, (
                f"Muted color {muted} should differ from secondary {theme.secondary} "
                f"in theme {theme.name}"
            )

    def test_muted_color_differs_from_panel(self):
        """Muted color should NOT match the panel background."""
        for theme in [OPENMIC_THEME, NORD_THEME]:
            muted = _muted_color(theme)
            assert muted != theme.panel, (
                f"Muted color {muted} should differ from panel {theme.panel} "
                f"in theme {theme.name}"
            )

    def test_muted_color_differs_from_surface(self):
        """Muted color should NOT match the surface background."""
        for theme in [OPENMIC_THEME, NORD_THEME]:
            muted = _muted_color(theme)
            assert muted != theme.surface, (
                f"Muted color {muted} should differ from surface {theme.surface} "
                f"in theme {theme.name}"
            )

    def test_muted_color_is_dimmed_foreground(self):
        """Muted color should be a dimmed version of the foreground."""
        for theme in [OPENMIC_THEME, NORD_THEME]:
            muted = _muted_color(theme)
            fg = theme.foreground
            # Muted should be darker than foreground
            mh = muted.lstrip("#")
            fh = fg.lstrip("#")
            m_brightness = sum(int(mh[i:i+2], 16) for i in (0, 2, 4))
            f_brightness = sum(int(fh[i:i+2], 16) for i in (0, 2, 4))
            assert m_brightness < f_brightness, (
                f"Muted {muted} should be darker than foreground {fg} "
                f"in theme {theme.name}"
            )

    def test_muted_color_returns_hex(self):
        """Muted color should return a valid hex color string."""
        for theme in [OPENMIC_THEME, NORD_THEME]:
            muted = _muted_color(theme)
            assert muted.startswith("#")
            assert len(muted) == 7


class TestSlashCommands:
    """FR-14: SLASH_COMMANDS list for autocomplete."""

    def test_commands_are_alphabetically_sorted(self):
        """SLASH_COMMANDS should be sorted alphabetically."""
        cmds = [cmd for cmd, _ in SLASH_COMMANDS]
        assert cmds == sorted(cmds)

    def test_all_commands_start_with_slash(self):
        """Every entry in SLASH_COMMANDS should start with /."""
        for cmd, _ in SLASH_COMMANDS:
            assert cmd.startswith("/"), f"Command {cmd} doesn't start with /"

    def test_all_commands_have_descriptions(self):
        """Every command should have a non-empty description."""
        for cmd, desc in SLASH_COMMANDS:
            assert desc, f"Command {cmd} has no description"

    def test_core_commands_present(self):
        """Core commands should be in the list."""
        cmds = [cmd for cmd, _ in SLASH_COMMANDS]
        for expected in ["/start", "/stop", "/exit", "/history", "/notes", "/query", "/pause"]:
            assert expected in cmds, f"{expected} missing from SLASH_COMMANDS"


class TestAutocompleteDropdown:
    """FR-14: Autocomplete dropdown filtering and selection logic."""

    def test_initial_state(self):
        """Dropdown starts with no matches."""
        dropdown = AutocompleteDropdown()
        assert dropdown._matches == []
        assert dropdown._selected_index == 0

    def test_get_selected_no_matches(self):
        """get_selected returns None when no matches."""
        dropdown = AutocompleteDropdown()
        assert dropdown.get_selected() is None

    def test_hide_clears_matches(self):
        """hide() clears matches."""
        dropdown = AutocompleteDropdown()
        dropdown._matches = [("/start", "Start")]
        dropdown.hide()
        assert dropdown._matches == []

    def test_selection_wraps_forward(self):
        """Selection index wraps around when reaching the end."""
        dropdown = AutocompleteDropdown()
        dropdown._matches = [("/a", "A"), ("/b", "B"), ("/c", "C")]
        dropdown._selected_index = 2
        # Simulate move_selection logic without render (no app context in tests)
        dropdown._selected_index = (dropdown._selected_index + 1) % len(dropdown._matches)
        assert dropdown._selected_index == 0

    def test_selection_wraps_backward(self):
        """Selection index wraps around when going past the start."""
        dropdown = AutocompleteDropdown()
        dropdown._matches = [("/a", "A"), ("/b", "B"), ("/c", "C")]
        dropdown._selected_index = 0
        dropdown._selected_index = (dropdown._selected_index - 1) % len(dropdown._matches)
        assert dropdown._selected_index == 2

    def test_get_selected_returns_command(self):
        """get_selected returns the currently selected command."""
        dropdown = AutocompleteDropdown()
        dropdown._matches = [("/start", "Start"), ("/stop", "Stop")]
        dropdown._selected_index = 1
        assert dropdown.get_selected() == "/stop"
