"""OpenMic TUI application."""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, Static
from textual.binding import Binding


class StatusBar(Static):
    """Status bar showing recording state."""

    def __init__(self) -> None:
        super().__init__("Status: IDLE")
        self.recording = False

    def set_recording(self, recording: bool) -> None:
        self.recording = recording
        status = "RECORDING" if recording else "IDLE"
        self.update(f"Status: {status}")


class TranscriptPane(Static):
    """Main pane displaying transcript text."""

    def __init__(self) -> None:
        super().__init__("")

    def append_text(self, text: str) -> None:
        current = str(self.renderable)
        self.update(current + text)

    def set_text(self, text: str) -> None:
        self.update(text)

    def clear(self) -> None:
        self.update("")


class CommandInput(Input):
    """Command input at the bottom of the screen."""

    def __init__(self) -> None:
        super().__init__(placeholder="Type /start, /stop, /query, or /notes...")


class OpenMicApp(App):
    """OpenMic TUI application."""

    CSS = """
    StatusBar {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    StatusBar.recording {
        background: $error;
    }

    TranscriptPane {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        overflow-y: auto;
    }

    CommandInput {
        dock: bottom;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+r", "toggle_recording", "Record"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.status_bar = StatusBar()
        self.transcript_pane = TranscriptPane()
        self.command_input = CommandInput()

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.status_bar
        yield Container(self.transcript_pane)
        yield self.command_input
        yield Footer()

    def action_toggle_recording(self) -> None:
        """Toggle recording state."""
        new_state = not self.status_bar.recording
        self.status_bar.set_recording(new_state)
        if new_state:
            self.status_bar.add_class("recording")
        else:
            self.status_bar.remove_class("recording")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input."""
        command = event.value.strip()
        self.command_input.value = ""

        if command == "/start":
            self.action_toggle_recording()
            if self.status_bar.recording:
                self.transcript_pane.set_text("Recording started...\n")
        elif command == "/stop":
            if self.status_bar.recording:
                self.action_toggle_recording()
                self.transcript_pane.append_text("\nRecording stopped.\n")
        elif command.startswith("/query"):
            self.transcript_pane.append_text("\n[Query feature not yet implemented]\n")
        elif command == "/notes":
            self.transcript_pane.append_text("\n[Notes feature not yet implemented]\n")
        elif command:
            self.transcript_pane.append_text(f"\nUnknown command: {command}\n")


def main() -> None:
    """Entry point for the OpenMic application."""
    app = OpenMicApp()
    app.run()


if __name__ == "__main__":
    main()
