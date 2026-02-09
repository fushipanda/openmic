"""OpenMic TUI application."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import json
from datetime import datetime, date
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option
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
from openmic.storage import save_transcript, list_transcripts, rename_transcript, NOTES_DIR
from openmic.rag import TranscriptRAG
from openmic.notes import generate_meeting_notes, generate_notes_for_latest, get_existing_notes

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

# Curated list of themes to cycle through.
# The first entry is our custom theme; the rest are Textual built-ins.
THEME_NAMES = [
    "openmic",
    "nord",
    "dracula",
    "tokyo-night",
    "monokai",
    "catppuccin-mocha",
    "gruvbox",
    "rose-pine",
]


def _muted_color(theme) -> str:
    """Get a muted text color that contrasts with surface/panel backgrounds.

    Uses a dimmed version of the foreground color rather than theme.secondary,
    which may match the background color.
    """
    fg = theme.foreground or "#e8e8e8"
    h = fg.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # Dim to ~50% brightness for visible-but-muted text
    r, g, b = int(r * 0.5), int(g * 0.5), int(b * 0.5)
    return f"#{r:02x}{g:02x}{b:02x}"


class UsageTracker:
    """Tracks ElevenLabs and LLM usage for the current session."""

    # ElevenLabs realtime sends 16kHz 16-bit mono audio in chunks
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2

    def __init__(self) -> None:
        self.audio_bytes_sent: int = 0
        self.llm_calls: int = 0
        self.llm_tokens: int = 0

    def add_audio_bytes(self, n: int) -> None:
        self.audio_bytes_sent += n

    def add_llm_call(self, tokens: int = 0) -> None:
        self.llm_calls += 1
        self.llm_tokens += tokens

    @property
    def audio_seconds(self) -> float:
        samples = self.audio_bytes_sent / self.BYTES_PER_SAMPLE
        return samples / self.SAMPLE_RATE

    def format_audio(self) -> str:
        secs = self.audio_seconds
        if secs < 60:
            return f"{secs:.0f}s"
        mins = secs / 60
        return f"{mins:.1f}m"

    def summary(self) -> str:
        parts = []
        if self.audio_bytes_sent > 0:
            parts.append(f"Audio: {self.format_audio()}")
        if self.llm_calls > 0:
            token_str = f" ({self.llm_tokens} tok)" if self.llm_tokens else ""
            parts.append(f"LLM: {self.llm_calls} call{'s' if self.llm_calls != 1 else ''}{token_str}")
        return " · ".join(parts) if parts else ""


class StatusBar(Static):
    """Status bar showing recording state and session usage."""

    def __init__(self, usage_tracker: UsageTracker | None = None) -> None:
        super().__init__("○ IDLE")
        self.recording = False
        self.paused = False
        self._usage_tracker = usage_tracker

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        theme = self.app.current_theme
        muted = _muted_color(theme)

        # Left side: recording state
        if self.paused:
            warning = theme.accent or "#ffaa00"
            status = Text("⏸ PAUSED", style=f"bold {warning}")
        elif self.recording:
            status = Text("◉ RECORDING", style=f"bold {theme.error}")
        else:
            status = Text("○ IDLE", style=muted)

        # Right side: usage info
        usage_str = ""
        if self._usage_tracker:
            usage_str = self._usage_tracker.summary()

        if usage_str:
            # Build a line with status centered/left and usage right-aligned
            try:
                width = self.size.width - 2  # account for padding
            except Exception:
                width = 80
            status_len = len(status.plain)
            usage_len = len(usage_str)
            padding = max(1, width - status_len - usage_len)
            text = Text()
            text.append(status)
            text.append(" " * padding)
            text.append(usage_str, style=muted)
            self.update(text)
        else:
            self.update(status)

    def set_recording(self, recording: bool) -> None:
        self.recording = recording
        self.paused = False
        self._update_display()

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self._update_display()

    def refresh_usage(self) -> None:
        """Refresh the usage display."""
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

    def watch_scroll_y(self, old: float, new: float) -> None:
        """Track scroll position to manage auto-scroll."""
        if new < old:
            # Scrolled up — disable auto-scroll
            self._auto_scroll = False
        elif self.max_scroll_y > 0 and new >= self.max_scroll_y - 1:
            # Scrolled to bottom — re-enable auto-scroll
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
        muted = _muted_color(theme)

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

    def set_markdown(self, markdown_text: str) -> None:
        """Render markdown content with rich formatting."""
        self._show_banner = False
        self._text = markdown_text
        self.update(RichMarkdown(markdown_text))
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


SLASH_COMMANDS = [
    ("/exit", "Quit OpenMic"),
    ("/help", "Show help"),
    ("/history", "List saved transcripts"),
    ("/name", "Rename the latest transcript"),
    ("/notes", "Generate structured meeting notes"),
    ("/pause", "Pause recording (resume with /start)"),
    ("/query", "Ask a question about your transcripts"),
    ("/start", "Start recording"),
    ("/stop", "Stop recording and run batch transcription"),
    ("/transcript", "View transcript by number or name"),
    ("/verbose", "Toggle debug output"),
]

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
    ("/exit", "Quit OpenMic"),
    ("", ""),
    ("Ctrl+R", "Toggle recording on/off"),
    ("Ctrl+T", "Cycle theme"),
    ("Esc", "Return to home screen"),
    ("Ctrl+C", "Quit"),
    ("Ctrl+?", "Show this help"),
]


class AutocompleteDropdown(Static):
    """Dropdown popup showing matching commands above the command input."""

    DEFAULT_CSS = """
    AutocompleteDropdown {
        layer: overlay;
        dock: bottom;
        height: auto;
        max-height: 14;
        margin: 0 0 3 1;
        background: $surface;
        color: $foreground;
        border: round $primary;
        padding: 0 1;
        display: none;
        scrollbar-size: 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__(" ")
        self._matches: list[tuple[str, str]] = []
        self._selected_index: int = 0

    def update_matches(self, query: str) -> None:
        """Filter commands matching the query and update the display."""
        if not query.startswith("/") or len(query) < 1:
            self._matches = []
            self.display = False
            return

        search = query.lower()
        self._matches = [
            (cmd, desc) for cmd, desc in SLASH_COMMANDS
            if cmd.startswith(search)
        ]
        self._selected_index = 0

        if not self._matches:
            self.display = False
            return

        # Update content before making visible to avoid None render
        self._render_content()
        self.display = True

    def _render_content(self) -> None:
        """Render the dropdown with the current matches and selection."""
        if not self._matches:
            return

        # Use Rich Text for better styling
        from rich.text import Text as RichText
        theme = self.app.current_theme

        lines = []
        for i, (cmd, desc) in enumerate(self._matches):
            line = RichText()
            if i == self._selected_index:
                # Highlighted selection
                line.append("▸ ", style=f"bold {theme.primary}")
                line.append(f"{cmd:<16}", style=f"bold {theme.accent}")
                line.append(f" {desc}", style=theme.foreground)
            else:
                line.append("  ")
                line.append(f"{cmd:<16}", style=theme.foreground)
                line.append(f" {desc}", style=_muted_color(theme))
            lines.append(line)

        # Combine all lines
        combined = RichText()
        for i, line in enumerate(lines):
            if i > 0:
                combined.append("\n")
            combined.append(line)

        self.update(combined)

    def move_selection(self, delta: int) -> None:
        """Move the selection up or down."""
        if not self._matches:
            return
        self._selected_index = (self._selected_index + delta) % len(self._matches)
        self._render_content()

    def get_selected(self) -> str | None:
        """Get the currently selected command."""
        if self._matches and 0 <= self._selected_index < len(self._matches):
            return self._matches[self._selected_index][0]
        return None

    def hide(self) -> None:
        """Hide the dropdown."""
        self._matches = []
        self.update(" ")
        self.display = False


class HelpScreen(ModalScreen):
    """Modal help popup showing available commands."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
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
        muted = _muted_color(theme)
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


def _parse_transcript_meta(path: Path) -> dict:
    """Extract metadata from a transcript filename."""
    stem = path.stem
    # Timestamp is first 16 chars: YYYY-MM-DD_HH-MM
    ts_str = stem[:16]
    name = stem[17:].replace("_", " ") if len(stem) > 16 else ""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M")
    except ValueError:
        dt = None
    return {"path": path, "name": name, "datetime": dt, "stem": stem}


def _date_header(dt: datetime) -> str:
    """Return a human-readable date header."""
    today = date.today()
    d = dt.date()
    if d == today:
        return "Today"
    from datetime import timedelta
    if d == today - timedelta(days=1):
        return "Yesterday"
    # e.g. "Jan 6th 2026"
    day = dt.day
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return dt.strftime(f"%b {day}{suffix} %Y")


class TranscriptPickerScreen(ModalScreen[Path | None]):
    """Modal popup for selecting a transcript."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def action_cancel(self) -> None:
        self.dismiss(None)

    DEFAULT_CSS = """
    TranscriptPickerScreen {
        align: center middle;
    }

    TranscriptPickerScreen > Vertical {
        width: 64;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    TranscriptPickerScreen > Vertical > Static {
        width: 100%;
        height: auto;
    }

    TranscriptPickerScreen > Vertical > OptionList {
        height: auto;
        max-height: 20;
        background: $surface;
        border: none;
    }
    """

    def __init__(self, transcripts: list[Path]) -> None:
        super().__init__()
        self._transcripts = transcripts
        self._path_map: dict[str, Path] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="picker-title")
            yield OptionList(id="picker-list")

    def on_mount(self) -> None:
        theme = self.app.current_theme
        primary = theme.primary or "#00d4aa"

        title = Text("Transcripts", style=f"bold {primary}")
        self.query_one("#picker-title").update(title)

        option_list = self.query_one("#picker-list", OptionList)
        metas = [_parse_transcript_meta(p) for p in self._transcripts]

        last_header = None
        for meta in metas:
            dt = meta["datetime"]
            if dt:
                header = _date_header(dt)
                if header != last_header:
                    if last_header is not None:
                        option_list.add_option(None)
                    prompt = Text(f"  {header}", style="bold")
                    option_list.add_option(Option(prompt, disabled=True))
                    last_header = header

                name = meta["name"] or "Untitled"
                time_str = dt.strftime("%-I:%M %p")
                has_notes = (NOTES_DIR / (meta["stem"] + "_notes.md")).exists()
                notes_indicator = " *" if has_notes else ""
                # Build a formatted line: name left, time right
                label = Text()
                label.append(f"  {name}", style=f"bold {theme.foreground or '#e8e8e8'}")
                if has_notes:
                    label.append(" *", style=f"bold {theme.success or '#2ed573'}")
                display_len = len(name) + len(notes_indicator) + len(time_str)
                pad = max(1, 50 - display_len)
                label.append(" " * pad)
                label.append(time_str, style=_muted_color(theme))

                option_id = str(meta["path"])
                self._path_map[option_id] = meta["path"]
                option_list.add_option(Option(label, id=option_id))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and event.option.id in self._path_map:
            self.dismiss(self._path_map[event.option.id])


class OpenMicApp(App):
    """OpenMic TUI application."""

    CSS = """
    Screen {
        background: $background;
        layers: default overlay;
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
        padding: 0 1;
    }

    CommandInput:focus {
        border: tall $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+r", "toggle_recording", "Record"),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("ctrl+question_mark", "show_help", "Help", show=False),
        Binding("escape", "go_back", "Back", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(OPENMIC_THEME)
        config = _load_config()
        saved_theme = config.get("theme")
        self.theme = saved_theme if saved_theme in THEME_NAMES else THEME_NAMES[0]
        self.usage_tracker = UsageTracker()
        self.status_bar = StatusBar(usage_tracker=self.usage_tracker)
        self.transcript_pane = TranscriptPane()
        self.command_input = CommandInput()
        self.autocomplete = AutocompleteDropdown()
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
        self._awaiting_session_name = False
        self._viewing = False  # True when viewing a transcript/notes (Esc returns to home)

    def compose(self) -> ComposeResult:
        yield self.status_bar
        yield Container(self.transcript_pane)
        yield self.autocomplete
        yield self.command_input
        yield Footer()

    def action_cycle_theme(self) -> None:
        """Cycle through available themes and persist the choice."""
        idx = THEME_NAMES.index(self.theme) if self.theme in THEME_NAMES else -1
        self.theme = THEME_NAMES[(idx + 1) % len(THEME_NAMES)]
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

    def action_go_back(self) -> None:
        """Return to home screen when viewing a transcript or notes."""
        if self._viewing:
            self._viewing = False
            self.transcript_pane.clear()

    async def action_toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.status_bar.paused:
            await self._resume_recording()
        elif self.status_bar.recording:
            await self._stop_recording()
        else:
            await self._start_recording()

    async def _start_recording(self) -> None:
        """Start audio recording and transcription."""
        # If paused, resume instead of starting fresh
        if self.audio_recorder.is_paused:
            await self._resume_recording()
            return
        self._viewing = False
        self._live_text = ""
        self._current_wav_path = self.audio_recorder.start()
        await self.transcriber.connect()
        self.status_bar.set_recording(True)
        self.status_bar.add_class("recording")
        session_info = f" [{self._session_name}]" if self._session_name else ""
        self.transcript_pane.set_text(f"Recording started...{session_info} ({self._current_wav_path.name})\n\n")

    async def _pause_recording(self) -> None:
        """Pause recording without stopping the session."""
        if not self.status_bar.recording or self.status_bar.paused:
            return
        self.audio_recorder.pause()
        await self.transcriber.disconnect()
        self.status_bar.set_paused(True)
        self.status_bar.remove_class("recording")
        self.transcript_pane.append_text("\n\n[Paused]\n")

    async def _resume_recording(self) -> None:
        """Resume recording after pause."""
        if not self.audio_recorder.is_paused:
            return
        self.audio_recorder.resume()
        await self.transcriber.connect()
        self.status_bar.set_paused(False)
        self.status_bar.set_recording(True)
        self.status_bar.add_class("recording")
        self.transcript_pane.append_text("\n[Resumed]\n\n")

    async def _stop_recording(self) -> None:
        """Stop audio recording and transcription."""
        was_paused = self.audio_recorder.is_paused
        wav_path = self.audio_recorder.stop()
        if not was_paused:
            await self.transcriber.disconnect()
        self.status_bar.set_recording(False)
        self.status_bar.remove_class("recording")
        if wav_path:
            self.transcript_pane.append_text(f"\n\nRecording stopped. Processing with diarization...\n")
            self._current_wav_path = wav_path
            await self._run_batch_transcription(wav_path)

    def _on_audio_chunk(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from recorder."""
        self.usage_tracker.add_audio_bytes(len(audio_bytes))
        self.transcriber.send_audio_chunk(audio_bytes)
        self.status_bar.refresh_usage()

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
            # Prompt for session name if none was provided
            if not self._session_name:
                self._awaiting_session_name = True
                self.command_input.placeholder = "Name this session (Enter to skip)"
                self.command_input.focus()
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError during transcription: {e}\n")

    def _display_diarized_transcript(self, segments: list[dict]) -> None:
        """Display diarized transcript with styled speaker labels."""
        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = _muted_color(theme)
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
        """Run a RAG query — show picker if multiple transcripts exist."""
        transcripts = list_transcripts()
        if not transcripts:
            self.transcript_pane.set_text("No transcripts available to query.\n")
            return
        if len(transcripts) == 1:
            await self._run_query_on_path(question, transcripts[0])
            return

        def on_selected(path: Path | None) -> None:
            if path is not None:
                self.call_later(self._run_query_on_path, question, path)

        self.push_screen(TranscriptPickerScreen(transcripts), on_selected)

    async def _run_query_on_path(self, question: str, transcript_path: Path) -> None:
        """Run a RAG query against a specific transcript."""
        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = _muted_color(theme)
        fg = theme.foreground or "#e8e8e8"
        processing = Text("Querying: ", style=f"italic {muted}")
        processing.append(question, style=fg)
        processing.append(f"\n\nSearching {transcript_path.stem}...", style=f"italic {muted}")
        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(processing)
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.rag.query_file(question, transcript_path),
            )
            self.usage_tracker.add_llm_call()
            self.status_bar.refresh_usage()
            result = Text()
            result.append("Q: ", style=f"bold {primary}")
            result.append(f"{question}\n\n")
            result.append("A: ", style=f"bold {primary}")
            result.append(f"{answer}\n")
            self.transcript_pane.update(result)
            self._viewing = True
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError during query: {e}\n")

    async def _generate_notes(self) -> None:
        """Generate meeting notes — show picker if multiple transcripts exist."""
        transcripts = list_transcripts()
        if not transcripts:
            self.transcript_pane.set_text("No transcripts available to generate notes from.\n")
            return
        if len(transcripts) == 1:
            await self._generate_notes_for_path(transcripts[0])
            return

        def on_selected(path: Path | None) -> None:
            if path is not None:
                self.call_later(self._generate_notes_for_path, path)

        self.push_screen(TranscriptPickerScreen(transcripts), on_selected)

    async def _generate_notes_for_path(self, transcript_path: Path) -> None:
        """Generate meeting notes for a specific transcript."""
        was_cached = get_existing_notes(transcript_path) is not None
        muted = _muted_color(self.current_theme)
        if was_cached:
            processing = Text("Loading saved notes...", style=f"italic {muted}")
        else:
            processing = Text("Generating meeting notes...", style=f"italic {muted}")
        self.transcript_pane._show_banner = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(processing)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_meeting_notes(transcript_path),
            )
            notes_content, notes_path = result
            if not was_cached:
                self.usage_tracker.add_llm_call()
                self.status_bar.refresh_usage()
            self.transcript_pane.set_markdown(f"{notes_content}\n\n---\n\n*Saved to: {notes_path}*\n")
            self._viewing = True
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError generating notes: {e}\n")

    def _display_transcripts(self) -> None:
        """Show transcript picker popup."""
        transcripts = list_transcripts()
        if not transcripts:
            self.transcript_pane.set_text("No transcripts found.\n")
            return

        def on_selected(path: Path | None) -> None:
            if path is not None:
                self._view_transcript_path(path)

        self.push_screen(TranscriptPickerScreen(transcripts), on_selected)

    def _view_transcript_path(self, target: Path) -> None:
        """View a transcript by its file path."""
        content = target.read_text()
        footer = f"\n---\n\n*{target}*\n"
        self.transcript_pane.set_markdown(content + footer)
        self._viewing = True

    def _view_transcript(self, identifier: str) -> None:
        """View a specific transcript by number or name."""
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

        self._view_transcript_path(target)

    def _reset_command_input(self) -> None:
        """Reset command input to default state."""
        self.command_input.placeholder = "/start · /stop · /history · /query · /notes"
        self._awaiting_session_name = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update autocomplete dropdown as the user types."""
        self.autocomplete.update_matches(event.value.strip())

    def on_key(self, event) -> None:
        """Handle keyboard events for autocomplete navigation."""
        if not self.autocomplete._matches:
            return
        if event.key == "up":
            self.autocomplete.move_selection(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self.autocomplete.move_selection(1)
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            # Tab key autocomplete: fill in the selected command
            selected = self.autocomplete.get_selected()
            if selected:
                self.command_input.value = selected + " "
                self.command_input.cursor_position = len(self.command_input.value)
                self.autocomplete.hide()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            self.autocomplete.hide()
            event.prevent_default()
            event.stop()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input."""
        # If autocomplete is showing and typed text is a partial match,
        # fill in the selected command instead of executing
        typed = event.value.strip()
        selected = self.autocomplete.get_selected()
        if selected and typed and typed != selected and not any(
            typed == cmd for cmd, _ in SLASH_COMMANDS
        ):
            self.command_input.value = selected + " "
            self.command_input.cursor_position = len(self.command_input.value)
            self.autocomplete.hide()
            return

        self.autocomplete.hide()
        command = typed
        self.command_input.value = ""

        # Handle session naming prompt
        if self._awaiting_session_name:
            self._reset_command_input()
            if command and self._latest_transcript_path and self._latest_transcript_path.exists():
                new_name = command.replace(" ", "_")
                new_path = rename_transcript(self._latest_transcript_path, new_name)
                self._latest_transcript_path = new_path
                self.transcript_pane.append_text(f"\nRenamed to: {new_path.name}\n")
            return

        if command == "/start":
            if self.status_bar.paused:
                await self._resume_recording()
            elif not self.status_bar.recording:
                self._session_name = None
                await self._start_recording()
        elif command.startswith("/start "):
            if self.status_bar.paused:
                await self._resume_recording()
            elif not self.status_bar.recording:
                self._session_name = command[7:].strip().replace(" ", "_")
                await self._start_recording()
        elif command == "/pause":
            if self.status_bar.recording and not self.status_bar.paused:
                await self._pause_recording()
        elif command == "/stop" or command.startswith("/stop "):
            if self.status_bar.recording or self.status_bar.paused:
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
        elif command == "/exit":
            self.exit()
        elif command == "/help":
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
