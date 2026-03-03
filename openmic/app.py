"""OpenMic TUI application."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import json
from datetime import datetime, date
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, OptionList, Static
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
from openmic.storage import save_transcript, list_transcripts, rename_transcript, format_transcript_title, TRANSCRIPTS_DIR, NOTES_DIR, RECORDINGS_DIR, list_recordings, delete_all_recordings
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
        """Return a summary of session usage/credits."""
        parts = []
        if self.audio_bytes_sent > 0:
            parts.append(f"Audio: {self.format_audio()}")
        if self.llm_calls > 0:
            token_str = f" ({self.llm_tokens} tok)" if self.llm_tokens else ""
            parts.append(f"LLM: {self.llm_calls} call{'s' if self.llm_calls != 1 else ''}{token_str}")

        if parts:
            return "Session: " + " · ".join(parts)
        return ""


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


class TranscriptPane(VerticalScroll):
    """Main pane displaying transcript text, with native scrolling."""

    can_focus = True

    @property
    def allow_vertical_scroll(self) -> bool:
        """Allow vertical scrolling regardless of scrollbar visibility or mount state."""
        return True

    def __init__(self) -> None:
        super().__init__()
        self._content = Static("", id="transcript-content")
        self._text = ""
        self._show_banner = False
        self._show_welcome = True
        self._auto_scroll = True

    def compose(self) -> ComposeResult:
        yield self._content

    def on_mount(self) -> None:
        if self._show_welcome:
            self._render_welcome()

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
        self._content.update(banner)

    def _render_welcome(self) -> None:
        """Render the query-first welcome screen."""
        theme = self.app.current_theme
        primary = theme.primary or "#00d4aa"
        muted = _muted_color(theme)

        transcripts = list_transcripts()
        count = len(transcripts)

        welcome = Text()
        welcome.append("OpenMic", style=f"bold {primary}")
        welcome.append("\n\n")
        if count > 0:
            welcome.append(f"{count} transcript{'s' if count != 1 else ''} available", style=muted)
            welcome.append(" — ask anything\n\n", style=muted)
            welcome.append("Type a question and press Enter to search across all transcripts.\n", style=f"italic {muted}")
            welcome.append("Use / for commands: ", style=f"italic {muted}")
            welcome.append("/start  /history  /notes", style=f"italic {primary}")
        else:
            welcome.append("No transcripts yet", style=muted)
            welcome.append(" — use ", style=muted)
            welcome.append("/start", style=f"bold {primary}")
            welcome.append(" to record your first meeting\n", style=muted)

        self._content.update(welcome)

    def append_text(self, text: str) -> None:
        self._show_banner = False
        self._show_welcome = False
        self._text += text
        self._content.update(self._text)
        self._scroll_to_bottom()

    def set_text(self, text: str) -> None:
        self._show_banner = False
        self._show_welcome = False
        self._text = text
        self._content.update(self._text)
        self._scroll_to_bottom()

    def set_markdown(self, markdown_text: str) -> None:
        """Render markdown content with rich formatting."""
        self._show_banner = False
        self._show_welcome = False
        self._text = markdown_text
        self._content.update(RichMarkdown(markdown_text))
        self._scroll_to_bottom()

    def update(self, renderable="") -> None:
        """Delegate content updates to the inner Static widget."""
        self._content.update(renderable)

    def clear(self) -> None:
        self._text = ""
        self._show_banner = False
        self._show_welcome = True
        self._auto_scroll = True
        self._render_welcome()


class CommandInput(Input):
    """Command input at the bottom of the screen."""

    def __init__(self) -> None:
        super().__init__(placeholder="Ask a question, or type / for commands...")


SLASH_COMMANDS = [
    ("/cleanup-recordings", "Delete all saved recordings"),
    ("/exit", "Quit OpenMic"),
    ("/help", "Show help"),
    ("/history", "List saved transcripts"),
    ("/name", "Rename the latest transcript"),
    ("/notes", "Generate structured meeting notes"),
    ("/pause", "Pause recording (resume with /start)"),
    ("/query", "Ask a question across all transcripts"),
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
    ("/query <question>", "Ask a question across all transcripts"),
    ("/notes", "Generate notes (with template selection)"),
    ("/name <name>", "Rename the latest transcript"),
    ("/cleanup-recordings", "Delete all saved recordings"),
    ("/verbose", "Toggle debug output"),
    ("/exit", "Quit OpenMic"),
    ("", ""),
    ("Ctrl+R", "Toggle recording on/off"),
    ("Ctrl+T", "Cycle theme"),
    ("Esc", "Return to home screen"),
    ("Ctrl+C", "Quit"),
    ("Ctrl+?", "Show this help"),
    ("", ""),
    ("Tip", "Type any question directly to search all transcripts"),
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

        from textual.content import Content
        from textual.visual import Style as TStyle
        theme = self.app.current_theme

        primary_style = TStyle(foreground=theme.primary, bold=True)
        accent_style = TStyle(foreground=theme.accent, bold=True)
        fg_style = TStyle(foreground=theme.foreground)
        muted_style = TStyle(foreground=_muted_color(theme))

        lines = []
        for i, (cmd, desc) in enumerate(self._matches):
            if i == self._selected_index:
                line = Content.assemble(
                    Content.styled("▸ ", primary_style),
                    Content.styled(f"{cmd:<16}", accent_style),
                    Content.styled(f" {desc}", fg_style),
                )
            else:
                line = Content.assemble(
                    Content("  "),
                    Content.styled(f"{cmd:<16}", fg_style),
                    Content.styled(f" {desc}", muted_style),
                )
            lines.append(line)

        combined = Content("\n").join(lines)
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
        self.update("")
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

        section_headers = iter(["Commands", "Shortcuts", ""])
        text.append(f"{next(section_headers)}\n", style=f"bold {fg}")
        for cmd, desc in HELP_COMMANDS:
            if not cmd:
                text.append("\n")
                header = next(section_headers, "")
                if header:
                    text.append(f"{header}\n", style=f"bold {fg}")
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


class TemplatePickerScreen(ModalScreen[str | None]):
    """Modal popup for selecting a note template."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def action_cancel(self) -> None:
        self.dismiss(None)

    DEFAULT_CSS = """
    TemplatePickerScreen {
        align: center middle;
    }

    TemplatePickerScreen > Vertical {
        width: 72;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    TemplatePickerScreen > Vertical > Static {
        width: 100%;
        height: auto;
    }

    TemplatePickerScreen > Vertical > OptionList {
        height: auto;
        max-height: 24;
        background: $surface;
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="template-picker-title")
            yield OptionList(id="template-picker-list")

    def on_mount(self) -> None:
        from openmic.templates import TemplateManager

        theme = self.app.current_theme
        primary = theme.primary or "#00d4aa"

        title = Text("Select Note Template", style=f"bold {primary}")
        self.query_one("#template-picker-title").update(title)

        option_list = self.query_one("#template-picker-list", OptionList)
        template_manager = TemplateManager()

        # Get built-in and user templates
        builtin_templates = template_manager.get_builtin_templates()
        user_templates = template_manager.get_user_templates()

        # Add built-in templates
        if builtin_templates:
            header = Text("  Built-in Templates", style="bold")
            option_list.add_option(Option(header, disabled=True))

            for template in sorted(builtin_templates, key=lambda t: t.id):
                label = Text()
                label.append(f"  {template.name}", style=f"bold {theme.foreground or '#e8e8e8'}")
                label.append("\n")
                label.append(f"    {template.description}", style=_muted_color(theme))
                option_list.add_option(Option(label, id=template.id))

        # Add separator and user templates if any exist
        if user_templates:
            option_list.add_option(None)  # Separator
            header = Text("  Custom Templates", style="bold")
            option_list.add_option(Option(header, disabled=True))

            for template in sorted(user_templates, key=lambda t: t.id):
                label = Text()
                label.append(f"  {template.name}", style=f"bold {theme.foreground or '#e8e8e8'}")
                label.append("\n")
                label.append(f"    {template.description}", style=_muted_color(theme))
                option_list.add_option(Option(label, id=template.id))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(event.option.id)


class ConfirmOverwriteScreen(ModalScreen[bool]):
    """Modal prompt to confirm overwriting existing notes with a different template."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
    ]

    def action_cancel(self) -> None:
        self.dismiss(False)

    DEFAULT_CSS = """
    ConfirmOverwriteScreen {
        align: center middle;
    }

    ConfirmOverwriteScreen > Vertical {
        width: 60;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }

    ConfirmOverwriteScreen > Vertical > Static {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    ConfirmOverwriteScreen > Vertical > Horizontal {
        width: 100%;
        height: auto;
        align: center middle;
    }

    ConfirmOverwriteScreen > Vertical > Horizontal > Button {
        margin: 0 1;
    }
    """

    def __init__(self, existing_template: str, new_template: str) -> None:
        super().__init__()
        self._existing_template = existing_template
        self._new_template = new_template

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="overwrite-message")
            with Horizontal():
                yield Button("Replace", variant="primary", id="btn-replace")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        theme = self.app.current_theme
        primary = theme.primary or "#00d4aa"

        msg = Text()
        msg.append("Notes already exist", style=f"bold {primary}")
        msg.append(f"\n\nGenerated with: ", style="")
        msg.append(self._existing_template, style="bold")
        msg.append(f"\nReplace with: ", style="")
        msg.append(self._new_template, style="bold")
        self.query_one("#overwrite-message").update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-replace")


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
                notes_indicator = " *" if not has_notes else ""
                # Build a formatted line: name left, time right
                label = Text()
                label.append(f"  {name}", style=f"bold {theme.foreground or '#e8e8e8'}")
                if not has_notes:
                    label.append(" *", style=f"bold {theme.accent or '#00d4aa'}")
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
        margin: 1 1 0 1;
        layout: vertical;
        overflow-y: scroll;
        overflow-x: hidden;
        scrollbar-size-vertical: 0;
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
        Binding("pageup", "scroll_up_page", show=False),
        Binding("pagedown", "scroll_down_page", show=False),
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
            output_dir=RECORDINGS_DIR,
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
        self._tab_cycling = False  # True when Tab is cycling through autocomplete matches
        # Ensure user templates directory exists
        self._user_templates_dir = CONFIG_DIR / "templates"
        self._user_templates_dir.mkdir(parents=True, exist_ok=True)

    def compose(self) -> ComposeResult:
        yield self.status_bar
        yield self.transcript_pane
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
        if self.transcript_pane._show_welcome:
            self.transcript_pane._render_welcome()
        elif self.transcript_pane._show_banner:
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

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Handle mouse scroll at app level — ensures scrolling works regardless of focus."""
        if not isinstance(self.screen, ModalScreen):
            self.transcript_pane.scroll_down(animate=False)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Handle mouse scroll at app level — ensures scrolling works regardless of focus."""
        if not isinstance(self.screen, ModalScreen):
            self.transcript_pane.scroll_up(animate=False)

    def action_scroll_up_page(self) -> None:
        """Scroll the transcript pane up by one page."""
        self.transcript_pane.scroll_page_up(animate=False)

    def action_scroll_down_page(self) -> None:
        """Scroll the transcript pane down by one page."""
        self.transcript_pane.scroll_page_down(animate=False)

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
        self.transcript_pane._show_welcome = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(text)

        if self._latest_transcript_path:
            saved_text = Text()
            saved_text.append(text)
            saved_text.append(f"\n\nSaved to: {self._latest_transcript_path}\n", style=f"italic {muted}")
            self.transcript_pane.update(saved_text)

    def _cleanup_recordings(self) -> None:
        """Delete all saved recordings and show summary."""
        recordings = list_recordings()
        if not recordings:
            self.transcript_pane.append_text("\nNo recordings to clean up.\n")
            return
        count, total_bytes = delete_all_recordings()
        if total_bytes >= 1_048_576:
            size_str = f"{total_bytes / 1_048_576:.1f} MB"
        else:
            size_str = f"{total_bytes / 1024:.1f} KB"
        self.transcript_pane.append_text(
            f"\nDeleted {count} recording{'s' if count != 1 else ''} ({size_str} freed).\n"
        )

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
        self.transcript_pane._show_welcome = False
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

    async def _run_query_all(self, question: str) -> None:
        """Run a RAG query across all transcripts with source citations."""
        transcripts = list_transcripts()
        if not transcripts:
            self.transcript_pane.set_text("No transcripts available to query.\nUse /start to record your first meeting.\n")
            return

        theme = self.current_theme
        primary = theme.primary or "#00d4aa"
        muted = _muted_color(theme)
        fg = theme.foreground or "#e8e8e8"

        # Show loading state
        loading = Text("Q: ", style=f"bold {primary}")
        loading.append(f"{question}\n\n", style=fg)
        loading.append(f"Searching across {len(transcripts)} transcript{'s' if len(transcripts) != 1 else ''}...", style=f"italic {muted}")
        self.transcript_pane._show_banner = False
        self.transcript_pane._show_welcome = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(loading)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.rag.query(question),
            )
            self.usage_tracker.add_llm_call()
            self.status_bar.refresh_usage()

            answer = result["answer"]
            sources = result["sources"]

            display = Text()
            display.append("Q: ", style=f"bold {primary}")
            display.append(f"{question}\n\n", style=fg)
            display.append("A: ", style=f"bold {primary}")
            display.append(f"{answer}\n", style=fg)

            if sources:
                display.append("\nSources:\n", style=f"bold {muted}")
                for src in sources:
                    display.append(f"  · {src}\n", style=muted)

            self.transcript_pane.update(display)
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
            self._show_template_picker(transcripts[0])
            return

        def on_selected(path: Path | None) -> None:
            if path is not None:
                self._show_template_picker(path)

        self.push_screen(TranscriptPickerScreen(transcripts), on_selected)

    def _show_template_picker(self, transcript_path: Path) -> None:
        """Show template picker — or skip it if notes already exist."""
        existing = get_existing_notes(transcript_path)
        if existing is not None:
            # Notes already exist — show them directly, no template picker needed
            self.call_later(self._generate_notes_for_path, transcript_path)
            return

        def on_template_selected(template_id: str | None) -> None:
            if template_id is not None:
                self.call_later(self._generate_notes_with_template, transcript_path, template_id)

        self.push_screen(TemplatePickerScreen(), on_template_selected)

    async def _generate_notes_with_template(self, transcript_path: Path, template_id: str) -> None:
        """Generate notes with a specific template, handling overwrite logic."""
        from openmic.templates import TemplateManager

        existing = get_existing_notes(transcript_path)
        if existing is not None:
            _, _, existing_template = existing
            # Same template — use cached notes
            if existing_template == template_id:
                await self._generate_notes_for_path(transcript_path, template_id)
                return
            # Different template — ask to overwrite
            template_manager = TemplateManager()
            existing_name = existing_template or "Standard Meeting Notes"
            if existing_template:
                tmpl = template_manager.get_template(existing_template)
                if tmpl:
                    existing_name = tmpl.name
            new_tmpl = template_manager.get_template(template_id)
            new_name = new_tmpl.name if new_tmpl else template_id

            def on_confirm(replace: bool) -> None:
                if replace:
                    self.call_later(self._generate_notes_for_path, transcript_path, template_id, True)
                else:
                    # Show existing notes without regenerating
                    self.call_later(self._generate_notes_for_path, transcript_path, existing_template or "default")

            self.push_screen(ConfirmOverwriteScreen(existing_name, new_name), on_confirm)
            return

        # No existing notes — generate directly
        await self._generate_notes_for_path(transcript_path, template_id)

    async def _generate_notes_for_path(
        self, transcript_path: Path, template_id: str = "default", force_regenerate: bool = False
    ) -> None:
        """Generate meeting notes for a specific transcript.

        Args:
            transcript_path: Path to the transcript file
            template_id: Template ID to use for generation
            force_regenerate: If True, delete existing notes and regenerate
        """
        from openmic.templates import TemplateManager

        # If forcing regeneration, delete existing notes first
        if force_regenerate:
            notes_path = NOTES_DIR / (transcript_path.stem + "_notes.md")
            if notes_path.exists():
                notes_path.unlink()

        existing = get_existing_notes(transcript_path)
        will_use_cache = existing is not None and existing[2] == template_id

        muted = _muted_color(self.current_theme)
        template_manager = TemplateManager()
        tmpl = template_manager.get_template(template_id)
        template_name = tmpl.name if tmpl else template_id

        if will_use_cache:
            processing = Text("Loading saved notes...", style=f"italic {muted}")
        else:
            processing = Text(f"Generating notes ({template_name})...", style=f"italic {muted}")
        self.transcript_pane._show_banner = False
        self.transcript_pane._show_welcome = False
        self.transcript_pane._text = ""
        self.transcript_pane.update(processing)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_meeting_notes(transcript_path, template_id),
            )
            notes_content, notes_path, used_cache = result
            if not used_cache:
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
        self.command_input.placeholder = "Ask a question, or type / for commands..."
        self._awaiting_session_name = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update autocomplete dropdown as the user types."""
        if self._tab_cycling:
            # Don't reset matches while Tab is cycling through commands
            self._tab_cycling = False
            return
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
            # Tab cycles forward through matching commands
            selected = self.autocomplete.get_selected()
            if selected:
                current_val = self.command_input.value.strip()
                if current_val == selected:
                    # Already showing this match — advance to next
                    self.autocomplete.move_selection(1)
                    selected = self.autocomplete.get_selected()
                # Fill the input with the selected command (set flag to avoid resetting matches)
                if selected:
                    self._tab_cycling = True
                    self.command_input.value = selected
                    self.command_input.cursor_position = len(self.command_input.value)
            event.prevent_default()
            event.stop()
        elif event.key == "shift+tab":
            # Shift+Tab cycles backward through matching commands
            selected = self.autocomplete.get_selected()
            if selected:
                current_val = self.command_input.value.strip()
                if current_val == selected:
                    # Already showing this match — go to previous
                    self.autocomplete.move_selection(-1)
                    selected = self.autocomplete.get_selected()
                if selected:
                    self._tab_cycling = True
                    self.command_input.value = selected
                    self.command_input.cursor_position = len(self.command_input.value)
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
                await self._run_query_all(query_text)
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
        elif command == "/cleanup-recordings":
            self._cleanup_recordings()
        elif command == "/verbose":
            self._verbose = not self._verbose
            self.transcriber.verbose = self._verbose
            state = "ON" if self._verbose else "OFF"
            self.transcript_pane.append_text(f"\nVerbose mode: {state}\n")
        elif command:
            # Non-slash input = query across all transcripts
            await self._run_query_all(command)


def main() -> None:
    """Entry point for the OpenMic application."""
    app = OpenMicApp()
    app.run()


if __name__ == "__main__":
    main()
