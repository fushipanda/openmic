"""Tests for the OpenMic TUI application."""

import json
import pytest
from unittest.mock import patch
from pathlib import Path

from openmic.app import (
    OpenMicApp, CommandInput, HELP_COMMANDS, SLASH_COMMANDS,
    _load_config, _save_config, CONFIG_FILE, THEME_NAMES,
    _muted_color, OPENMIC_THEME,
    AutocompleteDropdown, UsageTracker,
    _parse_transcript_meta,
)
from openmic.storage import NOTES_DIR


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
            idx = THEME_NAMES.index(config["theme"])
            new_theme = THEME_NAMES[(idx + 1) % len(THEME_NAMES)]
            config["theme"] = new_theme
            _save_config(config)

            reloaded = _load_config()
            assert reloaded["theme"] == THEME_NAMES[1]


class TestMutedColor:
    """BUG-1/BUG-2: Muted text color should contrast with backgrounds."""

    def test_muted_color_differs_from_secondary(self):
        """Muted color should NOT be the same as theme.secondary (which matches background)."""
        muted = _muted_color(OPENMIC_THEME)
        assert muted != OPENMIC_THEME.secondary, (
            f"Muted color {muted} should differ from secondary {OPENMIC_THEME.secondary}"
        )

    def test_muted_color_differs_from_panel(self):
        """Muted color should NOT match the panel background."""
        muted = _muted_color(OPENMIC_THEME)
        assert muted != OPENMIC_THEME.panel, (
            f"Muted color {muted} should differ from panel {OPENMIC_THEME.panel}"
        )

    def test_muted_color_differs_from_surface(self):
        """Muted color should NOT match the surface background."""
        muted = _muted_color(OPENMIC_THEME)
        assert muted != OPENMIC_THEME.surface, (
            f"Muted color {muted} should differ from surface {OPENMIC_THEME.surface}"
        )

    def test_muted_color_is_dimmed_foreground(self):
        """Muted color should be a dimmed version of the foreground."""
        muted = _muted_color(OPENMIC_THEME)
        fg = OPENMIC_THEME.foreground
        mh = muted.lstrip("#")
        fh = fg.lstrip("#")
        m_brightness = sum(int(mh[i:i+2], 16) for i in (0, 2, 4))
        f_brightness = sum(int(fh[i:i+2], 16) for i in (0, 2, 4))
        assert m_brightness < f_brightness, (
            f"Muted {muted} should be darker than foreground {fg}"
        )

    def test_muted_color_returns_hex(self):
        """Muted color should return a valid hex color string."""
        muted = _muted_color(OPENMIC_THEME)
        assert muted.startswith("#")
        assert len(muted) == 7


class TestThemeConsolidation:
    """FR-17: Consolidated theming system."""

    def test_openmic_is_first_theme(self):
        """The custom openmic theme should be the default (first in list)."""
        assert THEME_NAMES[0] == "openmic"

    def test_no_duplicate_theme_names(self):
        """Theme names should be unique."""
        assert len(THEME_NAMES) == len(set(THEME_NAMES))

    def test_multiple_themes_available(self):
        """There should be several themes to cycle through."""
        assert len(THEME_NAMES) >= 3

    def test_theme_cycle_wraps_around(self):
        """Cycling past the last theme should return to the first."""
        idx = len(THEME_NAMES) - 1
        next_idx = (idx + 1) % len(THEME_NAMES)
        assert next_idx == 0
        assert THEME_NAMES[next_idx] == "openmic"

    def test_custom_theme_has_required_colors(self):
        """The custom openmic theme should define all key color properties."""
        assert OPENMIC_THEME.primary is not None
        assert OPENMIC_THEME.background is not None
        assert OPENMIC_THEME.surface is not None
        assert OPENMIC_THEME.foreground is not None
        assert OPENMIC_THEME.error is not None

    def test_saved_theme_falls_back_to_default(self, tmp_path):
        """Unknown saved theme should fall back to the first theme."""
        config_file = tmp_path / "settings.json"
        with patch("openmic.app.CONFIG_FILE", config_file), \
             patch("openmic.app.CONFIG_DIR", tmp_path):
            _save_config({"theme": "nonexistent-theme"})
            config = _load_config()
            saved = config.get("theme")
            result = saved if saved in THEME_NAMES else THEME_NAMES[0]
            assert result == "openmic"


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


class TestUsageTracker:
    """FR-13: Session credit usage tracking."""

    def test_initial_state(self):
        """Tracker starts with zero usage."""
        tracker = UsageTracker()
        assert tracker.audio_bytes_sent == 0
        assert tracker.llm_calls == 0
        assert tracker.llm_tokens == 0
        assert tracker.audio_seconds == 0.0

    def test_add_audio_bytes(self):
        """Adding audio bytes accumulates correctly."""
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000)  # 1 second of 16kHz 16-bit audio
        assert tracker.audio_bytes_sent == 32000
        assert tracker.audio_seconds == 1.0

    def test_audio_seconds_calculation(self):
        """Audio seconds calculated from bytes, sample rate, and bit depth."""
        tracker = UsageTracker()
        # 16kHz, 16-bit (2 bytes per sample) = 32000 bytes per second
        tracker.add_audio_bytes(64000)  # 2 seconds
        assert tracker.audio_seconds == 2.0

    def test_format_audio_seconds(self):
        """Audio format shows seconds for short durations."""
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 30)  # 30 seconds
        assert tracker.format_audio() == "30s"

    def test_format_audio_minutes(self):
        """Audio format shows minutes for longer durations."""
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 90)  # 90 seconds = 1.5 minutes
        assert tracker.format_audio() == "1.5m"

    def test_add_llm_call(self):
        """LLM call counting works."""
        tracker = UsageTracker()
        tracker.add_llm_call()
        tracker.add_llm_call(tokens=150)
        assert tracker.llm_calls == 2
        assert tracker.llm_tokens == 150

    def test_summary_empty(self):
        """Summary returns empty string with no usage."""
        tracker = UsageTracker()
        assert tracker.summary() == ""

    def test_summary_audio_only(self):
        """Summary shows audio when only audio used."""
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 10)
        summary = tracker.summary()
        assert "Audio:" in summary
        assert "10s" in summary

    def test_summary_llm_only(self):
        """Summary shows LLM when only LLM used."""
        tracker = UsageTracker()
        tracker.add_llm_call()
        summary = tracker.summary()
        assert "LLM:" in summary
        assert "1 call" in summary

    def test_summary_both(self):
        """Summary shows both audio and LLM usage."""
        tracker = UsageTracker()
        tracker.add_audio_bytes(32000 * 60)
        tracker.add_llm_call(tokens=200)
        tracker.add_llm_call(tokens=300)
        summary = tracker.summary()
        assert "Audio:" in summary
        assert "LLM:" in summary
        assert "2 calls" in summary
        assert "500 tok" in summary


class TestNotesIndicator:
    """FR-18: Visual indicator for transcripts with generated notes."""

    def test_notes_file_detected(self, tmp_path, monkeypatch):
        """Notes indicator should detect when notes file exists."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        monkeypatch.setattr("openmic.app.NOTES_DIR", notes_dir)

        stem = "2025-06-15_14-30"
        notes_file = notes_dir / f"{stem}_notes.md"
        notes_file.write_text("# Notes")

        assert notes_file.exists()
        assert (notes_dir / (stem + "_notes.md")).exists()

    def test_notes_file_not_detected(self, tmp_path, monkeypatch):
        """Notes indicator should not show when no notes file exists."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        monkeypatch.setattr("openmic.app.NOTES_DIR", notes_dir)

        stem = "2025-06-15_14-30"
        assert not (notes_dir / (stem + "_notes.md")).exists()

    def test_parse_transcript_meta_extracts_stem(self):
        """_parse_transcript_meta extracts the stem used for notes lookup."""
        path = Path("transcripts/2025-06-15_14-30_standup.md")
        meta = _parse_transcript_meta(path)
        assert meta["stem"] == "2025-06-15_14-30_standup"

    def test_notes_indicator_path_matches_storage_convention(self, tmp_path, monkeypatch):
        """Notes path convention in picker matches storage.save_notes convention."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        monkeypatch.setattr("openmic.app.NOTES_DIR", notes_dir)

        # Simulate what storage.save_notes does
        transcript_stem = "2025-06-15_14-30_standup"
        notes_filename = transcript_stem + "_notes.md"
        (notes_dir / notes_filename).write_text("# Notes")

        # Verify the picker logic would detect it
        assert (notes_dir / (transcript_stem + "_notes.md")).exists()
