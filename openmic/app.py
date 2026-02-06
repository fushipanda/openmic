"""OpenMic TUI application."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import json
from pathlib import Path

from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Static
from textual.binding import Binding
from textual.theme import Theme

from dotenv import load_dotenv

CONFIG_DIR = Path.home() / ".config" / "openmic"
CONFIG_FILE = CONFIG_DIR / "settings.json"


def _load_config() -> dict:
    """Load persistent config."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(config: dict) -> None:
    """Save persistent config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

from openmic.audio import AudioRecorder
from openmic.transcribe import BatchTranscriber, RealtimeTranscriber
from openmic.storage import save_transcript, list_transcripts, rename_transcript
from openmic.rag import TranscriptRAG
from openmic.notes import generate_notes_for_latest

load_dotenv()

BANNER = """\
 ██████  ██████  ███████ ███    ██ ███    ███ ██  ██████
██    ██ ██   ██ ██      ████   ██ ████  ████ ██ ██
██    ██ ██████  █████   ██ ██  ██ ██ ████ ██ ██ ██
██    ██ ██      ██      ██  ██ ██ ██  ██  ██ ██ ██
 ██████  ██      ███████ ██   ████ ██      ██ ██  ██████"""

OPENMIC_THEME = Theme(
    name="openmic",
    primary="#00d4aa",
    secondary="#1a1b2e",
    accent="#00d4aa",
    background="#0d0e1a",
    surface="#141527",
    panel="#1a1b2e",
    error="#ff4757",
    success="#2ed573",
    foreground="#e8e8e8",
    dark=True,
)

NORD_THEME = Theme(
    name="nord",
    primary="#88c0d0",
    secondary="#434c5e",
    accent="#88c0d0",
    background="#2e3440",
    surface="#3b4252",
    panel="#434c5e",
    error="#bf616a",
    success="#a3be8c",
    foreground="#eceff4",
    dark=True,
)

THEMES = [OPENMIC_THEME, NORD_THEME]


class StatusBar(Static):
    """Status bar showing recording state."""

    def __init__(self) -> None:
        super().__init__("○ IDLE")
        self.recording = False

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        theme = self.app.current_theme
        if self.recording:
            text = Text("◉ RECORDING", style=f"bold {theme.error}")
        else:
            muted = theme.secondary or "#555577"
            text = Text("○ IDLE", style=muted)
        self.update(text)

    def set_recording(self, recording: bool) -> None:
        self.recording = recording
        self._update_display()


class TranscriptPane(Static):
    """Main pane displaying transcript text."""

    def __init__(self) -> None:
        super().__init__("")
        self._text = ""
        self._show_banner = True
        self._auto_scroll = True

    def on_mount(self) -> None:
        if self._show_banner:
            self._render_banner()

    def on_scroll_up(self) -> None:
        """User scrolled up — disable auto-scroll."""
        self._auto_scroll = False

    def on_scroll_down(self) -> None:
        """User scrolled down — re-enable auto-scroll if at bottom."""
        if self.scroll_offset.y >= self.max_scroll_y - 1:
            self._auto_scroll = True

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom if auto-scroll is enabled."""
        if self._auto_scroll:
            self.call_after_refresh(self.scroll_end, animate=False)

    @staticmethod
    def _dim_color(hex_color: str, factor: float = 0.35) -> str:
        """Dim a hex color by a factor (0-1) for shadow effect."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _render_banner(self) -> None:
        theme = self.app.current_theme
        primary = theme.primary or "#00d4aa"
        shadow = self._dim_color(primary)
        muted = theme.secondary or "#555577"

        lines = BANNER.split("\n")
        num_lines = len(lines)
        max_width = max(len(l) for l in lines)
        # Composite grid: main at (row, col+1), shadow at (row+1, col)
        rows = num_lines + 1
        cols = max_width + 1
        banner = Text()
        for r in range(rows):
            for c in range(cols):
                main_ch = " "
                if r < num_lines and (c - 1) >= 0 and (c - 1) < len(lines[r]):
                    main_ch = lines[r][c - 1]
                shadow_ch = " "
                if (r - 1) >= 0 and (r - 1) < num_lines and c < len(lines[r - 1]):
                    shadow_ch = lines[r - 1][c]
                if main_ch != " ":
                    banner.append(main_ch, style=f"bold {primary}")
                elif shadow_ch != " ":
                    banner.append(shadow_ch, style=shadow)
                else:
                    banner.append(" ")
            banner.append("\n")
        banner.append("\n")
        banner.append("voice → text → insight", style=f"italic {muted}")
        self.update(banner)

    def append_text(self, text: str) -> None:
        self._show_banner = False
        self._text += text
        self.update(self._text)
        self._scroll_to_bottom()

    def set_text(self, text: str) -> None:
        self._show_banner = False
        self._text = text
        self.update(self._text)
        self._scroll_to_bottom()

    def clear(self) -> None:
        self._text = ""
        self._show_banner = True
        self._auto_scroll = True
        self._render_banner()


class CommandInput(Input):
    """Command input at the bottom of the screen."""

    def __init__(self) -> None:
        super().__init__(placeholder="/start · /stop · /history · /query · /notes")


HELP_COMMANDS = [
    ("/start [name]", "Start recording (optionally with session name)"),
    ("/stop [name]", "Stop recording and run batch transcription"),
    ("/pause", "Pause recording (resume with /start)"),
    ("/history", "List saved transcripts"),
    ("/transcript <n>", "View transcript by number or name"),
    ("/query <question>", "Ask a question about your transcripts"),
    ("/notes", "Generate structured meeting notes"),
    ("/name <name>", "Rename the latest transcript"),
    ("/verbose", "Toggle debug output"),
    ("", ""),
    ("Ctrl+R", "Toggle recording on/off"),
    ("Ctrl+T", "Cycle theme"),
    ("Ctrl+C", "Quit"),
    ("?", "Show this help"),
]


class HelpScreen(ModalScreen):
    """Modal help popup showing available commands."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Vertical {
        width: 64;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    HelpScreen > Vertical > Static {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="help-content")

    def on_mount(self) -> None:
        theme = self.app.current_theme
        primary = theme.primary or "#00d4aa"
        muted = theme.secondary or "#555577"
        fg = theme.foreground or "#e8e8e8"

        text = Text()
        text.append("Help\n\n", style=f"bold {primary}")

        text.append("Commands\n", style=f"bold {fg}")
        for cmd, desc in HELP_COMMANDS:
            if not cmd:
                text.append("\n")
                text.append("Shortcuts\n", style=f"bold {fg}")
                continue
            text.append(f"  {cmd:<22}", style=f"bold {primary}")
            text.append(f" {desc}\n", style=muted)

        text.append(f"\nPress Esc to close", style=f"italic {muted}")
        self.query_one("#help-content").update(text)


class OpenMicApp(App):
    """OpenMic TUI application."""

    CSS = """
    Screen {
        background: $background;
    }

    StatusBar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
        text-align: center;
        border-bottom: solid $panel;
    }

    StatusBar.recording {
        background: #2a0a0f;
    }

    TranscriptPane {
        height: 1fr;
        padding: 1 2;
        background: $panel;
        border: round $primary 50%;
        overflow-y: auto;
        margin: 1 1 0 1;
    }

    CommandInput {
        dock: bottom;
        margin: 1;
        border: tall $primary 50%;
    }

    CommandInput:focus {
        border: tall $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+r", "toggle_recording", "Record"),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        for t in THEMES:
            self.register_theme(t)
        config = _load_config()
        saved_theme = config.get("theme")
        theme_names = [t.name for t in THEMES]
        self.theme = saved_theme if saved_theme in theme_names else THEMES[0].name
        self.status_bar = StatusBar()
        self.transcript_pane = TranscriptPane()
        self.command_input = CommandInput()
        self.audio_recorder = AudioRecorder(
            output_dir=Path("."),
            on_audio_chunk=self._on_audio_chunk,
        )
        self.transcriber = RealtimeTranscriber(
            on_partial=self._on_partial_transcript,
            on_committed=self._on_committed_transcript,
            on_error=self._on_transcriber_error,
            on_debug=self._on_transcriber_debug,
        )
        self._verbose = False
        self._live_text = ""
        self.batch_transcriber = BatchTranscriber()
        self._current_wav_path: Path | None = None
        self._latest_transcript_path: Path | None = None
        self.rag = TranscriptRAG()
        self._session_name: str | None = None

    def compose(self) -> ComposeResult:
        yield self.status_bar
        yield Container(self.transcript_pane)
        yield self.command_input
        yield Footer()

    def action_cycle_theme(self) -> None:
        """Cycle through available themes and persist the choice."""
        names = [t.name for t in THEMES]
        idx = names.index(self.theme) if self.theme in names else -1
        self.theme = names[(idx + 1) % len(names)]
        # Persist
        config = _load_config()
        config["theme"] = self.theme
        _save_config(config)
        # Re-render theme-aware widgets
        if self.transcript_pane._show_banner:
            self.transcript_pane._render_banner()
        self.status_bar._update_display()

    def action_show_help(self) -> None:
        """Show the help popup."""
        self.push_screen(HelpScreen())

    async def action_toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.status_bar.recording:
            await self._stop_recording()
        else:
            await self._start_recording()

    async def _start_recording(self) -> None:
        """Start audio recording and transcription."""
        self._live_text = ""
        self._current_wav_path = self.audio_recorder.start()
        await self.transcriber.connect()
        self.status_bar.set_recording(True)
        self.status_bar.add_class("recording")
        session_info = f" [{self._session_name}]" if self._session_name else ""
        self.transcript_pane.set_text(f"Recording started...{session_info} ({self._current_wav_path.name})\n\n")

    async def _stop_recording(self) -> None:
        """Stop audio recording and transcription."""
        wav_path = self.audio_recorder.stop()
        await self.transcriber.disconnect()
        self.status_bar.set_recording(False)
        self.status_bar.remove_class("recording")
        if wav_path:
            self.transcript_pane.append_text(f"\n\nRecording stopped. Processing with diarization...\n")
            self._current_wav_path = wav_path
            await self._run_batch_transcription(wav_path)

    def _on_audio_chunk(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from recorder."""
        self.transcriber.send_audio_chunk(audio_bytes)

    def _on_partial_transcript(self, text: str) -> None:
        """Handle partial transcript update."""
        self._update_partial(text)

    def _on_committed_transcript(self, text: str) -> None:
        """Handle committed transcript."""
        self._update_committed(text)

    def _on_transcriber_error(self, message: str) -> None:
        """Handle transcriber status/error messages."""
        self.transcript_pane.append_text(f"\n[{message}]\n")

    def _on_transcriber_debug(self, message: str) -> None:
        """Handle verbose debug messages from transcriber."""
        if self._verbose:
            self.transcript_pane.append_text(f"\n  dbg: {message}\n")

    def _update_partial(self, text: str) -> None:
        """Update transcript pane with partial text."""
        display = self._live_text + text
        self.transcript_pane.set_text(display)

    def _update_committed(self, text: str) -> None:
        """Update transcript pane with committed text.

        Each committed segment (triggered by VAD silence detection) starts
        on a new line to create natural paragraph breaks.
        """
        separator = "\n" if self._live_text else ""
        self._live_text += separator + text
        self.transcript_pane.set_text(self._live_text)

    async def _run_batch_transcription(self, wav_path: Path) -> None:
        """Run batch transcription with diarization."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.batch_transcriber.transcribe_file(str(wav_path)),
            )
            segments = BatchTranscriber.parse_diarized_result(result)
            self._latest_transcript_path = save_transcript(segments, self._session_name)
            self._display_diarized_transcript(segments)
            self._cleanup_wav(wav_path)
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError during transcription: {e}\n")

    def _display_diarized_transcript(self, segments: list[dict]) -> None:
        """Display diarized transcript with styled speaker labels."""
        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = theme.secondary or "#555577"
        text = Text()
        for segment in segments:
            speaker = segment.get("speaker", "Speaker")
            content = segment.get("text", "")
            text.append(f"[{speaker}]", style=f"bold {primary}")
            text.append(f": {content}\n")

        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(text)

        if self._latest_transcript_path:
            saved_text = Text()
            saved_text.append(text)
            saved_text.append(f"\n\nSaved to: {self._latest_transcript_path}\n", style=f"italic {muted}")
            self.transcript_pane.update(saved_text)

    def _cleanup_wav(self, wav_path: Path) -> None:
        """Delete temporary WAV file after successful transcription."""
        try:
            wav_path.unlink()
        except OSError:
            pass

    async def _run_query(self, question: str) -> None:
        """Run a RAG query against the transcripts."""
        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = theme.secondary or "#555577"
        fg = theme.foreground or "#e8e8e8"
        processing = Text("Querying: ", style=f"italic {muted}")
        processing.append(question, style=fg)
        processing.append("\n\nSearching transcripts...", style=f"italic {muted}")
        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(processing)
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.rag.query(question),
            )
            result = Text()
            result.append("Q: ", style=f"bold {primary}")
            result.append(f"{question}\n\n")
            result.append("A: ", style=f"bold {primary}")
            result.append(f"{answer}\n")
            self.transcript_pane.update(result)
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError during query: {e}\n")

    async def _generate_notes(self) -> None:
        """Generate meeting notes from the latest transcript."""
        muted = self.current_theme.secondary or "#555577"
        processing = Text("Generating meeting notes...", style=f"italic {muted}")
        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(processing)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                generate_notes_for_latest,
            )
            if result is None:
                self.transcript_pane.set_text("No transcripts available to generate notes from.\n")
            else:
                notes_content, notes_path = result
                self.transcript_pane.set_text(f"{notes_content}\n\nSaved to: {notes_path}\n")
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError generating notes: {e}\n")

    def _display_transcripts(self) -> None:
        """Display a list of saved transcripts."""
        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = theme.secondary or "#555577"
        fg = theme.foreground or "#e8e8e8"

        transcripts = list_transcripts()
        if not transcripts:
            self.transcript_pane.set_text("No transcripts found.\n")
            return

        text = Text()
        text.append("Transcripts\n\n", style=f"bold {primary}")
        for i, path in enumerate(transcripts, 1):
            stem = path.stem
            # Parse: timestamp is first 16 chars, rest after _ is the name
            timestamp = stem[:16].replace("_", " ")
            name = stem[17:].replace("_", " ") if len(stem) > 16 else ""
            text.append(f"  {i}. ", style=f"bold {primary}")
            if name:
                text.append(f"{name}", style=f"bold {fg}")
                text.append(f"  {timestamp}\n", style=muted)
            else:
                text.append(f"{timestamp}\n", style=fg)

        text.append(f"\nUse /transcript <number> to view\n", style=f"italic {muted}")

        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(text)

    def _view_transcript(self, identifier: str) -> None:
        """View a specific transcript by number or name."""
        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = theme.secondary or "#555577"

        transcripts = list_transcripts()
        if not transcripts:
            self.transcript_pane.set_text("No transcripts found.\n")
            return

        # Try as a number first
        target = None
        try:
            idx = int(identifier) - 1
            if 0 <= idx < len(transcripts):
                target = transcripts[idx]
        except ValueError:
            pass

        # Try as a name match
        if target is None:
            search = identifier.lower().replace(" ", "_")
            for path in transcripts:
                if search in path.stem.lower():
                    target = path
                    break

        if target is None:
            self.transcript_pane.set_text(f"Transcript not found: {identifier}\n")
            return

        content = target.read_text()
        stem = target.stem
        name = stem[17:].replace("_", " ") if len(stem) > 16 else stem

        text = Text()
        if name:
            text.append(f"{name}\n\n", style=f"bold {primary}")
        text.append(content)
        text.append(f"\n{target}\n", style=f"italic {muted}")

        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(text)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input."""
        command = event.value.strip()
        self.command_input.value = ""

        if command == "/start":
            if not self.status_bar.recording:
                self._session_name = None
                await self._start_recording()
        elif command.startswith("/start "):
            if not self.status_bar.recording:
                self._session_name = command[7:].strip().replace(" ", "_")
                await self._start_recording()
        elif command == "/stop" or command.startswith("/stop "):
            if self.status_bar.recording:
                name = command[5:].strip() if command.startswith("/stop ") else None
                if name:
                    self._session_name = name.replace(" ", "_")
                await self._stop_recording()
        elif command.startswith("/query"):
            query_text = command[6:].strip()
            if query_text:
                await self._run_query(query_text)
            else:
                self.transcript_pane.append_text("\nUsage: /query <your question>\n")
        elif command == "/notes":
            await self._generate_notes()
        elif command.startswith("/name "):
            new_name = command[6:].strip()
            if not new_name:
                self.transcript_pane.append_text("\nUsage: /name <transcript name>\n")
            elif self._latest_transcript_path and self._latest_transcript_path.exists():
                new_path = rename_transcript(self._latest_transcript_path, new_name)
                self._latest_transcript_path = new_path
                self.transcript_pane.append_text(f"\nRenamed to: {new_path.name}\n")
            else:
                self.transcript_pane.append_text("\nNo recent transcript to rename.\n")
        elif command in ("/transcripts", "/history", "/transcript"):
            self._display_transcripts()
        elif command.startswith("/transcript ") or command.startswith("/history "):
            prefix_len = len(command.split()[0]) + 1
            name = command[prefix_len:].strip()
            if name:
                self._view_transcript(name)
            else:
                self.transcript_pane.append_text("\nUsage: /transcript <name or number>\n")
        elif command in ("/help", "?"):
            self.action_show_help()
        elif command == "/verbose":
            self._verbose = not self._verbose
            self.transcriber.verbose = self._verbose
            state = "ON" if self._verbose else "OFF"
            self.transcript_pane.append_text(f"\nVerbose mode: {state}\n")
        elif command:
            self.transcript_pane.append_text(f"\nUnknown command: {command}\n")


def main() -> None:
    """Entry point for the OpenMic application."""
    app = OpenMicApp()
    app.run()


if __name__ == "__main__":
    main()
