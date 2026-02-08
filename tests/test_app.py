"""Tests for the OpenMic TUI application."""

import pytest

from openmic.app import OpenMicApp, CommandInput


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
