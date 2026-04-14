"""OpenMic — lightweight CLI for meeting transcription and RAG querying."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import json
import os
import re
import select as _select
import sys
import termios
import tty
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.table import Table, box
from rich.text import Text
from rich.theme import Theme

from dotenv import load_dotenv

CONFIG_DIR = Path.home() / ".config" / "openmic"
CONFIG_FILE = CONFIG_DIR / "settings.json"

_MD_THEME = Theme({
    "markdown.h1":        "bold #00d4aa underline",
    "markdown.h2":        "bold #00d4aa",
    "markdown.h3":        "#00d4aa",
    "markdown.h1.border": "#007a63",
    "markdown.strong":    "bold bright_white",
    "markdown.emph":      "italic",
    "markdown.code":      "bold #569cd6",
})

console = Console(theme=_MD_THEME, force_terminal=True, color_system="truecolor")


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


def _update_env_file(key: str, value: str) -> None:
    """Write or update a key=value pair in the .env file."""
    value = value.replace("\n", "").replace("\r", "")
    env_path = Path(".env")
    if not env_path.exists():
        env_path = CONFIG_DIR / ".env"
    if env_path.exists():
        lines = env_path.read_text().splitlines(keepends=True)
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}\n")
        env_path.write_text("".join(lines))
    else:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(f"{key}={value}\n")


MODEL_REGISTRY: dict[str, dict] = {
    "ollama": {
        "label": "Ollama (Local — private)",
        "env_key": None,   # No API key required
        "models": [],      # Populated dynamically from running Ollama instance
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            ("claude-opus-4-6", "Most capable"),
            ("claude-sonnet-4-6", "Recommended"),
            ("claude-3-5-sonnet-20241022", "Stable"),
            ("claude-haiku-4-5-20251001", "Fastest"),
        ],
    },
    "openai": {
        "label": "OpenAI (GPT)",
        "env_key": "OPENAI_API_KEY",
        "models": [
            ("gpt-5.4", "Most capable"),
            ("gpt-5.4-pro", "Highest quality"),
            ("gpt-4.1-mini", "Fast & efficient"),
            ("o3", "Reasoning"),
            ("o3-mini", "Fast reasoning"),
        ],
    },
    "gemini": {
        "label": "Google (Gemini)",
        "env_key": "GEMINI_API_KEY",
        "models": [
            ("gemini-2.5-pro", "Most capable"),
            ("gemini-2.5-flash", "Recommended"),
            ("gemini-2.5-flash-lite", "Budget"),
        ],
    },
    "openrouter": {
        "label": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
            ("mistralai/mistral-large", "Mistral Large"),
            ("deepseek/deepseek-chat", "DeepSeek Chat"),
            ("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
        ],
    },
}

# Load project .env first, then fall back to the config-dir .env
# (written by _update_env_file when no project .env exists).
load_dotenv()
load_dotenv(CONFIG_DIR / ".env", override=False)

WHISPER_MODELS = [
    ("tiny.en",        "Tiny — fastest"),
    ("base.en",        "Base"),
    ("small.en",       "Small — recommended"),
    ("medium.en",      "Medium"),
    ("large-v3",       "Large — most accurate"),
    ("large-v3-turbo", "Large Turbo — fast + accurate"),
]


def _get_transcribers():
    """Return (RealtimeTranscriber, BatchTranscriber) — always local whisper.cpp."""
    from openmic.local_transcribe import LocalRealtimeTranscriber, LocalBatchTranscriber
    return LocalRealtimeTranscriber, LocalBatchTranscriber

from openmic.audio import AudioRecorder
from openmic.storage import (
    save_transcript,
    list_transcripts,
    get_latest_transcript,
    rename_transcript,
    format_transcript_title,
    TRANSCRIPTS_DIR,
    NOTES_DIR,
    RECORDINGS_DIR,
    list_recordings,
    delete_all_recordings,
)
from openmic.session import (
    create_session,
    append_transcript as session_append_transcript,
    append_notes as session_append_notes,
    append_title_update,
    append_rename,
    display_title,
    list_sessions,
    read_session,
    get_session_meta,
    session_duration_s,
)
from openmic.rag import TranscriptRAG, generate_session_title
from openmic.notes import generate_meeting_notes, get_existing_notes

BANNER = """\
 ██████  ██████  ███████ ███    ██ ███    ███ ██  ██████
██    ██ ██   ██ ██      ████   ██ ████  ████ ██ ██
██    ██ ██████  █████   ██ ██  ██ ██ ████ ██ ██ ██
██    ██ ██      ██      ██  ██ ██ ██  ██  ██ ██ ██
 ██████  ██      ███████ ██   ████ ██      ██ ██  ██████"""

HELP_COMMANDS = [
    ("/start [name]",    "Start a new recording session"),
    ("/record [name]",   "Start a new recording session"),
    ("/stop",            "Stop recording and run batch transcription"),
    ("/resume",          "Browse sessions and set the active session"),
    ("/delete",          "Permanently delete a session"),
    ("/transcript <n>",  "View a session by number or name"),
    ("/query <question>","Ask a question across all transcripts"),
    ("/notes",           "Generate notes (with template selection)"),
    ("/notes <template>","Regenerate notes with a specific template"),
    ("/notes copy",      "Copy latest notes to clipboard"),
    ("/notes export",    "Export latest notes to a markdown file (use 'html' for email-ready output)"),
    ("/regen",           "Regenerate notes using the saved template"),
    ("/model",           "Select LLM provider and model"),
    ("/transcribe",      "Select Whisper model size"),
    ("/rename <title>",  "Set a custom display title for the active session"),
    ("/name <name>",     "Rename the latest transcript"),
    ("/cleanup-recordings", "Delete all saved recordings"),
    ("/clear",           "Exit active session and clear the screen"),
    ("/verbose",         "Toggle debug output"),
    ("/version",         "Show version and check for updates"),
    ("/exit",            "Quit OpenMic"),
]


class UsageTracker:
    """Tracks audio and LLM usage for the current session."""

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
        return f"{secs / 60:.1f}m"

    def summary(self) -> str:
        parts = []
        if self.audio_bytes_sent > 0:
            parts.append(f"Audio: {self.format_audio()}")
        if self.llm_calls > 0:
            tok = f" ({self.llm_tokens} tok)" if self.llm_tokens else ""
            parts.append(f"LLM: {self.llm_calls} call{'s' if self.llm_calls != 1 else ''}{tok}")
        return "Session: " + " · ".join(parts) if parts else ""

    @staticmethod
    def current_model_label() -> str:
        model = os.environ.get("LLM_MODEL", "")
        if model:
            return model
        provider = os.environ.get("LLM_PROVIDER", "")
        info = MODEL_REGISTRY.get(provider)
        if info and info["models"]:
            return info["models"][0][0]
        return ""


@dataclass
class ReplContext:
    """Mutable state threaded through the REPL."""
    rag: TranscriptRAG
    usage: UsageTracker = field(default_factory=UsageTracker)
    latest_transcript_path: Path | None = None
    active_session_path: Path | None = None
    active_session_name: str | None = None
    verbose: bool = False
    chatting: bool = False


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Print ASCII banner and tagline."""
    from openmic.version import get_version
    console.print(f"[bold #00d4aa]{BANNER}[/]")
    console.print()
    console.print(f"[dim italic]voice → text → insight[/]  [dim]v{get_version()}[/]")
    console.print()


def print_help() -> None:
    """Print command reference table."""
    t = Table(show_header=False, box=None, padding=(0, 1))
    for cmd, desc in HELP_COMMANDS:
        if cmd:
            t.add_row(f"[bold #00d4aa]{cmd}[/]", f"[dim]{desc}[/]")
        else:
            t.add_row("", "")
    t.add_row("", "")
    t.add_row("[dim]Tip[/]", "[dim]Type any question directly to search all transcripts[/]")
    t.add_row("[dim]@mention[/]", "[dim]Prefix with @name to scope query to one transcript[/]")
    console.print(t)


def _parse_transcript_meta(path: Path) -> dict:
    """Extract metadata from a transcript filename."""
    stem = path.stem
    ts_str = stem[:16]
    name = stem[17:].replace("_", " ") if len(stem) > 16 else ""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M")
    except ValueError:
        dt = None
    return {"path": path, "name": name, "datetime": dt, "stem": stem}


def _date_header(dt: datetime) -> str:
    today = date.today()
    d = dt.date()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return dt.strftime(f"%b {day}{suffix} %Y")


def _arrow_select(rows: list[dict]) -> Any | None:
    """
    Inline arrow-key selector. Returns the selected item's value, or None on cancel.

    Row kinds:
      {"kind": "header", "text": str}
      {"kind": "note",   "text": str}
      {"kind": "item",   "primary": str, "secondary": str, "value": Any, "current": bool}
    """
    TEAL  = "\033[38;2;0;212;170m"
    DIM   = "\033[2m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"
    CL    = "\r\033[K"

    selectable = [i for i, r in enumerate(rows) if r.get("kind") == "item"]
    if not selectable:
        return None

    # Start on the first item marked current, else 0
    cursor = 0
    for j, i in enumerate(selectable):
        if rows[i].get("current"):
            cursor = j
            break

    def _render(first: bool = False) -> None:
        if not first:
            sys.stdout.write(f"\033[{len(rows)}A")
        active_row = selectable[cursor]
        for i, row in enumerate(rows):
            sys.stdout.write(CL)
            k = row["kind"]
            if k == "header":
                sys.stdout.write(f"  {BOLD}{row['text']}{RESET}\n")
            elif k == "note":
                sys.stdout.write(f"    {DIM}{row['text']}{RESET}\n")
            else:
                is_active = (i == active_row)
                cur_mark = f"{TEAL}✓{RESET} " if row.get("current") and not is_active else "  "
                prefix   = f"  {TEAL}→{RESET} " if is_active else "     "
                primary  = f"{TEAL}{BOLD}{row['primary']}{RESET}" if is_active else row["primary"]
                sec      = row.get("secondary", "")
                secondary = f"  {DIM}{sec}{RESET}" if sec else ""
                sys.stdout.write(f"{prefix}{cur_mark}{primary}{secondary}\n")
        sys.stdout.write(CL)
        sys.stdout.write(f"  {DIM}↑↓ · Enter select · Esc cancel{RESET}")
        sys.stdout.flush()

    _render(first=True)

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    result = None
    try:
        tty.setraw(fd)
        while True:
            ch = os.read(fd, 1)          # os.read bypasses Python IO buffering
            if ch == b"\x1b":
                rlist, _, _ = _select.select([fd], [], [], 0.05)
                if rlist:
                    nxt = os.read(fd, 1)
                    if nxt == b"[":
                        arrow = os.read(fd, 1)
                        if arrow == b"A":
                            cursor = max(0, cursor - 1)
                            _render()
                        elif arrow == b"B":
                            cursor = min(len(selectable) - 1, cursor + 1)
                            _render()
                else:
                    break  # plain Escape — cancel
            elif ch in (b"\r", b"\n"):
                result = rows[selectable[cursor]]["value"]
                break
            elif ch == b"\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\n")
        sys.stdout.flush()

    return result


def pick_transcript(transcripts: list[Path]) -> Path | None:
    """Arrow-key transcript picker."""
    if not transcripts:
        console.print("[dim]No transcripts found.[/]")
        return None

    rows: list[dict] = []
    last_header = None
    for path in transcripts:
        meta = _parse_transcript_meta(path)
        dt   = meta["datetime"]
        if dt:
            header = _date_header(dt)
            if header != last_header:
                rows.append({"kind": "header", "text": header})
                last_header = header
            has_notes = (NOTES_DIR / (meta["stem"] + "_notes.md")).exists()
            title     = format_transcript_title(path.stem[:16], meta["name"])
            time_str  = dt.strftime("%-I:%M %p")
            primary   = f"{'  ' if has_notes else '* '}{title}"
            rows.append({
                "kind":    "item",
                "primary": primary,
                "secondary": time_str,
                "value":   path,
                "current": False,
            })
        else:
            rows.append({"kind": "item", "primary": path.stem, "secondary": "", "value": path, "current": False})

    console.print("[dim]  * = notes not yet generated[/]")
    try:
        return _arrow_select(rows)
    except KeyboardInterrupt:
        return None


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    secs = int(seconds)
    h, remainder = divmod(secs, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _print_duration_bar(session_path: Path) -> None:
    """Print a duration bar for a session if it has recordings."""
    duration = session_duration_s(session_path)
    if duration <= 0:
        return
    bar_max = 40
    filled = min(bar_max, max(1, int(duration / 30)))
    bar = "━" * filled + "╌" * (bar_max - filled)
    console.print(f"  [dim]{bar}[/]  [bold]{_format_duration(duration)}[/] [dim]total[/]")


def pick_session(sessions: list[Path], active: Path | None = None) -> Path | None:
    """Arrow-key session picker. Returns selected session path or None."""
    if not sessions:
        console.print("[dim]No sessions found.[/]")
        return None

    rows: list[dict] = [{"kind": "header", "text": "Sessions"}]
    for path in sessions:
        try:
            data = read_session(path)
            name = display_title(data).replace("_", " ")
            created_str = data["meta"].get("created", "")
            n_recordings = len(data["transcripts"])
            has_notes = len(data["notes"]) > 0

            try:
                dt = datetime.fromisoformat(created_str)
                date_str = dt.strftime("%-d %b %Y")
            except ValueError:
                date_str = ""

            indicator = "✓" if has_notes else "*"
            rec_label = f"{n_recordings} recording{'s' if n_recordings != 1 else ''}"
            primary = f"{indicator} {name}"
            secondary = f"{date_str}  {rec_label}"
            is_active = active is not None and path == active
            if is_active:
                primary = f"→ {name}"
        except Exception:
            primary = path.stem
            secondary = ""
            is_active = False

        rows.append({
            "kind": "item",
            "primary": primary,
            "secondary": secondary,
            "value": path,
            "current": is_active,
        })

    console.print("[dim]  ✓ = has notes  * = no notes yet  → = active session[/]")
    try:
        return _arrow_select(rows)
    except KeyboardInterrupt:
        return None


def _get_ollama_models() -> list[tuple[str, str]]:
    """Return list of (model_name, size_label) from running Ollama instance."""
    import urllib.request
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
        return [
            (m["name"], f"{m.get('size', 0) / 1_073_741_824:.1f}GB")
            for m in data.get("models", [])
        ]
    except Exception:
        return []


def pick_model() -> tuple[str, str] | None:
    """Arrow-key model picker. Returns (provider, model_id) or None."""
    current_provider = os.environ.get("LLM_PROVIDER", "")
    current_model    = os.environ.get("LLM_MODEL", "")

    rows: list[dict] = []
    for pkey, info in MODEL_REGISTRY.items():
        models = _get_ollama_models() if pkey == "ollama" else info["models"]
        rows.append({"kind": "header", "text": info["label"]})
        if pkey == "ollama" and not models:
            rows.append({"kind": "note", "text": "Ollama not running or no models installed — visit ollama.ai"})
            continue
        for model_id, desc in models:
            rows.append({
                "kind":      "item",
                "primary":   model_id,
                "secondary": desc,
                "value":     (pkey, model_id),
                "current":   pkey == current_provider and model_id == current_model,
            })

    try:
        result = _arrow_select(rows)
    except KeyboardInterrupt:
        return None

    if result is None:
        return None

    pkey, model_id = result
    env_key = MODEL_REGISTRY[pkey]["env_key"]
    if env_key and not os.environ.get(env_key):
        try:
            api_key = input(f"  Enter {env_key}: ").strip()
            if api_key:
                os.environ[env_key] = api_key
                _update_env_file(env_key, api_key)
            else:
                console.print("[dim]API key required — cancelled.[/]")
                return None
        except KeyboardInterrupt:
            return None
    return pkey, model_id


def pick_template() -> str:
    """Arrow-key template picker. Returns selected template ID."""
    from openmic.templates import TemplateManager
    tm = TemplateManager()
    builtin      = sorted(tm.get_builtin_templates(), key=lambda t: t.id)
    user_tmpls   = sorted(tm.get_user_templates(), key=lambda t: t.id)
    all_templates = builtin + user_tmpls

    if not all_templates:
        return "default"

    rows: list[dict] = []
    for tmpl in all_templates:
        tag  = " (custom)" if not tmpl.is_builtin else ""
        rows.append({
            "kind":      "item",
            "primary":   tmpl.name,
            "secondary": f"{tmpl.description}{tag}",
            "value":     tmpl.id,
            "current":   tmpl.id == "default",
        })

    try:
        result = _arrow_select(rows)
    except KeyboardInterrupt:
        return "default"
    return result if result is not None else "default"


def pick_whisper_model() -> str | None:
    """Arrow-key whisper model picker. Returns model_id or None."""
    current_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

    rows: list[dict] = [{"kind": "header", "text": "Whisper model (whisper.cpp — local)"}]
    for model_id, desc in WHISPER_MODELS:
        rows.append({
            "kind":      "item",
            "primary":   model_id,
            "secondary": desc,
            "value":     model_id,
            "current":   model_id == current_model,
        })

    try:
        return _arrow_select(rows)
    except KeyboardInterrupt:
        return None


# ---------------------------------------------------------------------------
# Mention resolution
# ---------------------------------------------------------------------------

def _resolve_transcript_mention(mention_name: str) -> Path | None:
    """Resolve a @[Name] mention to a transcript path."""
    transcripts = list_transcripts()
    for path in transcripts:
        meta = _parse_transcript_meta(path)
        display = format_transcript_title(path.stem[:16], meta["name"])
        if display == mention_name:
            return path
    search = mention_name.lower()
    for path in transcripts:
        meta = _parse_transcript_meta(path)
        display = format_transcript_title(path.stem[:16], meta["name"])
        if search in display.lower():
            return path
    return None


async def _handle_mention_query(text: str, ctx: ReplContext) -> bool:
    """Parse @[Name] mention and run scoped query. Returns True if handled."""
    match = re.search(r'@\[([^\]]+)\]', text)
    if not match:
        return False
    mention_name = match.group(1)
    question = re.sub(r'@\[[^\]]+\]', '', text).strip()
    if not question:
        console.print("[dim]Please include a question with your @mention.[/]")
        return True
    path = _resolve_transcript_mention(mention_name)
    if path:
        await _run_query_on_path(question, path, ctx)
    else:
        console.print(f"[dim]Transcript not found: {mention_name}[/]")
    return True


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = "⣷⣯⣟⡿⢿⣻⣽⣾"   # counter-clockwise

_TEAL_ANSI = "\033[38;2;0;212;170m"
_DIM_ANSI  = "\033[2m"
_RST_ANSI  = "\033[0m"


async def _spinner_task(label: str, done: asyncio.Event) -> None:
    """Braille spinner that runs until done is set."""
    t0 = asyncio.get_event_loop().time()
    i  = 0
    while not done.is_set():
        elapsed = asyncio.get_event_loop().time() - t0
        frame   = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]

        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            m, s = divmod(int(elapsed), 60)
            time_str = f"{m}m {s:02d}s"

        sys.stdout.write(
            f"\r{_TEAL_ANSI}{frame}{_RST_ANSI}"
            f"    {label}"
            f"    {_DIM_ANSI}{time_str}{_RST_ANSI}"
            f"   "   # trailing spaces to overwrite any previous longer line
        )
        sys.stdout.flush()
        await asyncio.sleep(0.08)   # slightly faster = smoother
        i += 1
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


async def _with_spinner(label: str, fn):
    """Run fn() in a thread executor while showing a braille spinner.

    Escape or Ctrl+C cancels the await immediately (the underlying thread
    finishes in the background — see note in _run below).
    """
    done = asyncio.Event()
    spin = asyncio.create_task(_spinner_task(label, done))
    loop = asyncio.get_event_loop()

    # Wrap in a Task (not a bare Future) so task.cancel() interrupts the await
    # at the next event-loop tick, even while the thread is still running.
    async def _run():
        return await loop.run_in_executor(None, fn)

    task = asyncio.create_task(_run())

    # Listen for Escape on stdin using setcbreak (single-char reads; SIGINT still works).
    fd: int | None = None
    old_settings = None
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            fd = None
    except Exception:
        fd = None

    if fd is not None:
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def _on_key() -> None:
                try:
                    ch = os.read(fd, 32)
                except OSError:
                    return
                if b"\x1b" in ch:  # Escape
                    task.cancel()

            loop.add_reader(fd, _on_key)
        except Exception:
            fd = None

    try:
        return await task
    except (asyncio.CancelledError, KeyboardInterrupt):
        done.set()
        spin.cancel()
        try:
            await spin
        except asyncio.CancelledError:
            pass
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        raise
    finally:
        done.set()
        if not spin.done():
            await spin
        if fd is not None:
            loop.remove_reader(fd)
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_TABLE_LINE_RE = re.compile(r'^\s*\|')


def _parse_md_table(lines: list[str]) -> dict | None:
    """Parse markdown table lines into {headers, alignments, rows}.

    Returns None if lines don't form a valid table (must have header + separator rows).
    alignments: list of 'left' | 'center' | 'right' per column.
    """
    stripped = [l.strip() for l in lines if l.strip()]
    if len(stripped) < 2:
        return None

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = _cells(stripped[0])
    n = len(headers)

    sep = _cells(stripped[1])
    if not all(re.match(r'^:?-+:?$', c) for c in sep):
        return None

    alignments = []
    for cell in sep:
        if cell.startswith(':') and cell.endswith(':'):
            alignments.append('center')
        elif cell.endswith(':'):
            alignments.append('right')
        else:
            alignments.append('left')

    rows = [_cells(r) for r in stripped[2:]]
    rows = [(r + [''] * n)[:n] for r in rows]

    return {"headers": headers, "alignments": alignments, "rows": rows}


def _render_md_table(table: dict) -> None:
    """Render a parsed markdown table. Uses stacked format on narrow terminals."""
    import shutil

    cols    = shutil.get_terminal_size((80, 24)).columns
    headers = table["headers"]
    aligns  = table["alignments"]
    rows    = table["rows"]

    def _stacked() -> None:
        for i, row in enumerate(rows):
            if i > 0:
                console.print(f"  [{TEAL_DIM}]{'·' * 28}[/]")
            for h, v in zip(headers, row):
                console.print(f"  [bold dim]{h}:[/] {v}")

    if cols < 60:
        _stacked()
        return

    n = len(headers)
    max_widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows) if rows else [0])
        for i in range(n)
    ]

    available = cols - (n * 3) - 1
    total_content = sum(max_widths)

    MIN_READABLE = 15  # below this per-column, stacked is cleaner

    # If even minimum-width columns won't fit, stacked is more readable than cramped cells
    if n * MIN_READABLE > available:
        _stacked()
        return

    if total_content <= available:
        widths = max_widths
    else:
        widths = [max(MIN_READABLE, int(w * available / total_content)) for w in max_widths]

    RICH_JUSTIFY = {"left": "left", "center": "center", "right": "right"}
    t = Table(show_header=True, header_style=f"bold {TEAL}", box=box.SIMPLE_HEAD, padding=(0, 1))
    for h, a, w in zip(headers, aligns, widths):
        t.add_column(h, justify=RICH_JUSTIFY[a], no_wrap=False, max_width=w)
    for row in rows:
        t.add_row(*row)
    console.print(t)


def _strip_md_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (---\\n...\\n---) from the start of a markdown string."""
    if not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    # parts[0] == '' (before first ---), parts[1] == frontmatter, parts[2] == rest
    if len(parts) == 3:
        return parts[2].lstrip("\n")
    return content


def render_markdown(content: str) -> None:
    """Render markdown to the console with custom table handling.

    Table blocks use a width-aware Rich Table; everything else uses RichMarkdown.
    """
    lines = content.splitlines(keepends=True)
    segment_lines: list[str] = []
    in_table = False

    def _flush_text(buf: list[str]) -> None:
        text = "".join(buf).strip()
        if text:
            console.print(RichMarkdown(text))

    def _flush_table(buf: list[str]) -> None:
        parsed = _parse_md_table(buf)
        if parsed:
            _render_md_table(parsed)
        else:
            _flush_text(buf)

    for line in lines:
        if in_table:
            if _TABLE_LINE_RE.match(line):
                segment_lines.append(line)
            else:
                _flush_table(segment_lines)
                segment_lines = [line]
                in_table = False
        else:
            if _TABLE_LINE_RE.match(line):
                _flush_text(segment_lines)
                segment_lines = [line]
                in_table = True
            else:
                segment_lines.append(line)

    if in_table:
        _flush_table(segment_lines)
    else:
        _flush_text(segment_lines)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

async def _run_query_on_path(question: str, path: Path, ctx: ReplContext) -> None:
    """Run RAG query against a specific transcript."""
    try:
        answer = await _with_spinner(f"Searching {path.stem}", lambda: ctx.rag.query_file(question, path))
        ctx.usage.add_llm_call()
        console.print()
        console.print(f"[bold {TEAL}]  >[/] {question}")
        console.print()
        render_markdown(answer)
    except Exception as e:
        console.print(f"[red]Query error: {e}[/]")


async def _run_query_all(question: str, ctx: ReplContext) -> None:
    """Run RAG query across all sessions."""
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No sessions available. Use /start to record a meeting.[/]")
        return

    if not ctx.chatting:
        ctx.chatting = True
        ctx.rag.clear_chat_history()

    n     = len(sessions)
    label = f"Searching {n} session{'s' if n != 1 else ''}"
    try:
        result = await _with_spinner(label, lambda: ctx.rag.query(question))
        ctx.usage.add_llm_call()
        answer  = result["answer"]
        sources = result.get("sources", [])

        console.print()
        console.print(f"[bold {TEAL}]  >[/] {question}")
        console.print()
        render_markdown(answer)
        if sources:
            console.print(f"[dim]Sources: {', '.join(sources)}[/]")
        console.print()
    except Exception as e:
        console.print(f"[red]Query error: {e}[/]")


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

async def _generate_notes_for_session(
    session_path: Path,
    template_id: str = "default",
    force_regen: bool = False,
    ctx: ReplContext | None = None,
) -> None:
    """Generate (or show cached) notes for a session and append to its JSONL."""
    import tempfile

    data = read_session(session_path)
    existing_notes = data["notes"]

    if not force_regen and existing_notes:
        # Show the most recent notes entry from the session
        latest = existing_notes[-1]
        content = _strip_md_frontmatter(latest.get("content", ""))
        saved_template = latest.get("template", "default")
        console.print()
        render_markdown(content)
        console.print()
        console.print(f"[dim]Notes from session: {session_path.name}[/]")
        return

    # Build a temporary markdown file from the session transcript text so
    # generate_meeting_notes() can load and summarise it via LangChain.
    from openmic.session import session_to_text
    text = session_to_text(session_path)
    if not text.strip():
        console.print("[dim]Session has no transcript content yet.[/]")
        return

    meta = get_session_meta(session_path)
    session_name = meta.get("name", session_path.stem).replace("_", " ")
    header = f"# {session_name}\n\n"

    # Derive a properly-formatted filename for the temp file so that
    # generate_meeting_notes() can parse the date from the stem correctly.
    created_iso = meta.get("created", "")
    try:
        dt = datetime.fromisoformat(created_iso)
        ts_prefix = dt.strftime("%Y-%m-%d_%H-%M")
    except (ValueError, TypeError):
        ts_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")
    raw_name = meta.get("name", "")
    safe_name = re.sub(r"[^\w-]", "_", raw_name)[:40] if raw_name else ""
    tmp_filename = f"{ts_prefix}_{safe_name}.md" if safe_name else f"{ts_prefix}.md"

    try:
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / tmp_filename
        tmp_path.write_text(header + text, encoding="utf-8")

        label = f"Generating notes ({template_id})"
        result = await _with_spinner(label, lambda: generate_meeting_notes(tmp_path, template_id, force_regen=force_regen))
        notes_content, _, _ = result
        if ctx:
            ctx.usage.add_llm_call()

        # Append notes into the session JSONL (replaces the temp-file-based save)
        session_append_notes(session_path, notes_content, template_id)

        console.print()
        render_markdown(_strip_md_frontmatter(notes_content))
        console.print()
        console.print(f"[dim]Notes saved to session: {session_path.name}[/]")
    except Exception as e:
        console.print(f"[red]Error generating notes: {e}[/]")
    finally:
        try:
            tmp_path.unlink()
            tmp_dir.rmdir()
        except Exception:
            pass


async def _do_notes(ctx: ReplContext, force_regen: bool = False) -> None:
    """Handle /notes and /regen — use active session or pick one."""
    if ctx.active_session_path and ctx.active_session_path.exists():
        session_path = ctx.active_session_path
    else:
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No sessions available.[/]")
            return
        if len(sessions) == 1:
            session_path = sessions[0]
        else:
            console.print()
            session_path = pick_session(sessions, active=ctx.active_session_path)
            if session_path is None:
                return

    if force_regen:
        data = read_session(session_path)
        existing = data["notes"]
        template_id = existing[-1].get("template", "default") if existing else "default"
        if not existing:
            console.print()
            template_id = pick_template()
        await _generate_notes_for_session(session_path, template_id, force_regen=True, ctx=ctx)
        return

    # /notes: show cached if present, else pick template and generate
    data = read_session(session_path)
    if data["notes"]:
        template_id = data["notes"][-1].get("template", "default")
        await _generate_notes_for_session(session_path, template_id, ctx=ctx)
    else:
        console.print()
        template_id = pick_template()
        await _generate_notes_for_session(session_path, template_id, ctx=ctx)


async def _do_notes_with_template(ctx: ReplContext, template_id: str) -> None:
    """Handle /notes <template> — force-regenerate notes with the given template."""
    from openmic.templates import TemplateManager
    tm = TemplateManager()
    if tm.get_template(template_id) is None:
        available = [t.id for t in tm.get_builtin_templates() + tm.get_user_templates()]
        console.print(f"[red]Unknown template '{template_id}'. Available: {', '.join(available)}[/]")
        return

    if ctx.active_session_path and ctx.active_session_path.exists():
        session_path = ctx.active_session_path
    else:
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No sessions available.[/]")
            return
        if len(sessions) == 1:
            session_path = sessions[0]
        else:
            console.print()
            session_path = pick_session(sessions, active=ctx.active_session_path)
            if session_path is None:
                return

    await _generate_notes_for_session(session_path, template_id, force_regen=True, ctx=ctx)


# ---------------------------------------------------------------------------
# Notes copy / export helpers
# ---------------------------------------------------------------------------

def _get_notes_session(ctx: ReplContext) -> Path | None:
    """Return active session path, or None if unavailable (prints error)."""
    if ctx.active_session_path and ctx.active_session_path.exists():
        return ctx.active_session_path
    console.print("[red]No active session. Use /sessions to select one.[/]")
    return None


def _latest_notes_content(session_path: Path) -> str | None:
    """Return the most recent notes content for a session, or None if absent."""
    data = read_session(session_path)
    notes = data.get("notes", [])
    return notes[-1].get("content") if notes else None


_HTML_WRAPPER = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; font-size: 14px; color: #000; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
  h1, h2, h3 {{ color: #000; }}
  h2 {{ font-size: 15px; margin-top: 24px; margin-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
  th {{ text-align: left; padding: 6px 12px; border-bottom: 2px solid #000; font-weight: bold; }}
  td {{ padding: 5px 12px; border-bottom: 1px solid #ccc; vertical-align: top; }}
  ul, ol {{ margin: 4px 0 12px; padding-left: 20px; }}
  li {{ margin-bottom: 3px; }}
  p {{ margin: 4px 0 10px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _notes_to_html(markdown_content: str) -> str:
    """Convert markdown notes to an email-friendly HTML document."""
    import markdown as md
    # Strip YAML frontmatter if present
    content = markdown_content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].lstrip()
    body = md.markdown(content, extensions=["tables", "nl2br"])
    return _HTML_WRAPPER.format(body=body)


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using wl-copy (Wayland) or xclip (X11).

    Returns True on success, False if no clipboard tool is available.
    """
    import subprocess
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


# ---------------------------------------------------------------------------
# Session title generation helpers
# ---------------------------------------------------------------------------

def _current_model_name() -> str:
    """Return a readable model identifier based on current env vars."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("LLM_MODEL", "")
    return f"{provider}/{model}" if model else provider


async def _background_title_gen(session_path: Path) -> None:
    """Generate and persist an autoTitle for a session silently in the background.

    Uses a thread executor so the LLM call doesn't block the event loop.
    Any failure is silently swallowed — title is optional metadata.
    """
    loop = asyncio.get_event_loop()
    try:
        title = await loop.run_in_executor(None, generate_session_title, session_path)
        if title:
            append_title_update(session_path, title, _current_model_name())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Recording mode
# ---------------------------------------------------------------------------

async def recording_mode(session_name: str | None = None, ctx: ReplContext | None = None) -> Path | None:
    """
    Record audio and stream live transcript to terminal.
    Ctrl+C stops recording and triggers batch transcription.
    Returns the saved transcript path, or None on failure.
    """
    import time
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text as RichText

    lines: list[str] = []
    partial_holder: list[str] = [""]
    committed_queue: list[tuple[str, str]] = []  # (text, elapsed) for display
    realtime_segments: list[dict] = []           # structured segments for session JSONL
    _dirty: list[bool] = [False]
    _start_time: list[float] = [0.0]
    _last_seg_end: list[float] = [0.0]           # tracks end time of previous segment

    # Offset elapsed display by existing session duration so timer reads cumulatively
    _offset_s = session_duration_s(ctx.active_session_path) if (ctx and ctx.active_session_path and ctx.active_session_path.exists()) else 0.0

    def _elapsed() -> str:
        return _format_duration(_offset_s + (time.monotonic() - _start_time[0]))

    usage = ctx.usage if ctx else UsageTracker()
    verbose = ctx.verbose if ctx else False

    def on_audio_chunk(audio_bytes: bytes) -> None:
        usage.add_audio_bytes(len(audio_bytes))
        transcriber.send_audio_chunk(audio_bytes)

    def on_partial(text: str) -> None:
        partial_holder[0] = text
        _dirty[0] = True

    def on_committed(text: str) -> None:
        partial_holder[0] = ""
        lines.append(text)
        now = time.monotonic() - _start_time[0]
        realtime_segments.append({
            "speaker": "Speaker",
            "text": text.strip(),
            "start": round(_last_seg_end[0], 2),
            "end": round(now, 2),
        })
        _last_seg_end[0] = now
        committed_queue.append((text, _elapsed()))
        _dirty[0] = True

    def on_error(msg: str) -> None:
        console.print(f"[dim][{msg}][/]")

    def on_debug(msg: str) -> None:
        if verbose:
            console.print(f"[dim]  dbg: {msg}[/]")

    RealtimeTranscriberCls, _ = _get_transcribers()

    recorder = AudioRecorder(
        output_dir=RECORDINGS_DIR,
        on_audio_chunk=on_audio_chunk,
        on_limit_reached=lambda: console.print("\n[yellow]⚠ 6-hour limit reached — stopping.[/]"),
    )
    transcriber = RealtimeTranscriberCls(
        on_partial=on_partial,
        on_committed=on_committed,
        on_error=on_error,
        on_debug=on_debug,
    )

    wav_path = recorder.start()
    await transcriber.connect()
    _start_time[0] = time.monotonic()

    session_info = f" [{session_name}]" if session_name else ""
    console.print()
    console.print(f"[bold #ff4757]◉ RECORDING{session_info}[/]  [dim]Ctrl+C to stop[/]")
    console.print()

    def _make_chunk_row(text: str, ts: str) -> Table:
        row = Table.grid(expand=True)
        row.add_column(ratio=1)
        row.add_column(width=6, justify="right")
        row.add_row(RichText(text), RichText(ts, style="dim"))
        return row

    with Live(RichText(""), console=console, refresh_per_second=8, transient=False) as live:
        try:
            while True:
                await asyncio.sleep(0.1)
                while committed_queue:
                    text, ts = committed_queue.pop(0)
                    live.console.print(_make_chunk_row(text, ts))
                    live.console.print()
                if _dirty[0]:
                    partial = partial_holder[0]
                    live.update(RichText(partial, style="dim italic") if partial else RichText(""))
                    _dirty[0] = False
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

    console.print()
    console.print("[dim]Stopping...[/]")

    was_paused = recorder.is_paused
    returned_wav = recorder.stop()
    if not was_paused:
        await transcriber.disconnect()

    if not returned_wav:
        console.print("[dim]No audio recorded.[/]")
        return None

    # Use the realtime transcript segments directly — no batch pass needed.
    # Whisper.cpp already transcribed the audio chunk-by-chunk during recording;
    # running it again on the full WAV would eat CPU after recording stops with
    # no quality improvement (local mode has no diarization either way).
    try:
        duration_s = returned_wav.stat().st_size / (16000 * 2)  # int16 mono 16kHz
        segments = realtime_segments

        _SPEAKER_COLORS = [
            "#00d4aa",  # teal
            "#ff6b6b",  # coral
            "#4dabf7",  # blue
            "#ffd43b",  # yellow
            "#a9e34b",  # lime
            "#da77f2",  # purple
            "#ff922b",  # orange
            "#66d9e8",  # cyan
        ]
        speaker_color_map: dict[str, str] = {}
        prev_speaker: str | None = None

        console.print()
        for seg in segments:
            speaker = seg.get("speaker", "Speaker")
            text = seg.get("text", "")
            if speaker not in speaker_color_map:
                speaker_color_map[speaker] = _SPEAKER_COLORS[len(speaker_color_map) % len(_SPEAKER_COLORS)]
            color = speaker_color_map[speaker]
            if prev_speaker and prev_speaker != speaker:
                console.print()
            console.print(f"[bold {color}][{speaker}][/] {text}")
            prev_speaker = speaker

        console.print()

        # Session save: append to active session or create a new one
        active = ctx.active_session_path if ctx else None
        if active and active.exists():
            session_append_transcript(active, segments, duration_s)
            data = read_session(active)
            console.print(f"[dim]Appended to session: {display_title(data).replace('_', ' ')}[/]")
            # Fire background title generation if no autoTitle yet
            if data.get("autoTitle") is None:
                asyncio.create_task(_background_title_gen(active))
            return active
        else:
            # Prompt for a name if none supplied
            if not session_name:
                try:
                    name = input("Name this session (Enter to skip): ").strip()
                    session_name = name if name else None
                except KeyboardInterrupt:
                    pass

            session_path = create_session(session_name)
            session_append_transcript(session_path, segments, duration_s)
            if ctx:
                ctx.active_session_path = session_path
                ctx.active_session_name = session_name or session_path.stem
            data = read_session(session_path)
            console.print(f"[dim]Session created: {display_title(data).replace('_', ' ')}[/]")
            # Fire background title generation
            asyncio.create_task(_background_title_gen(session_path))
            return session_path

    except Exception as e:
        console.print(f"[red]Transcription error: {e}[/]")
        return None


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

async def handle_command(cmd: str, ctx: ReplContext) -> bool:
    """
    Handle one REPL command. Returns False if the REPL should exit.
    """
    cmd = cmd.strip()
    if not cmd:
        return True

    # --- Recording ---
    if cmd in ("/start", "/record", "/recording") or cmd.startswith("/start ") or cmd.startswith("/record ") or cmd.startswith("/recording "):
        # Extract optional name argument from /start, /record, or /recording
        if cmd.startswith("/start "):
            session_name = cmd[7:].strip().replace(" ", "_") or None
        elif cmd.startswith("/record "):
            session_name = cmd[8:].strip().replace(" ", "_") or None
        elif cmd.startswith("/recording "):
            session_name = cmd[11:].strip().replace(" ", "_") or None
        else:
            session_name = None
        path = await recording_mode(session_name, ctx)
        if path:
            ctx.latest_transcript_path = path
            ctx.chatting = False
            # Show active session reminder after recording
            if ctx.active_session_path:
                meta = get_session_meta(ctx.active_session_path)
                console.print(f"[dim]Active session: {meta.get('name', ctx.active_session_path.stem)}[/]")
        return True

    if cmd == "/stop":
        console.print("[dim]Not currently recording. Use /start to begin.[/]")
        return True

    # --- Clear session + chat history ---
    if cmd == "/clear":
        ctx.active_session_path = None
        ctx.active_session_name = None
        ctx.latest_transcript_path = None
        ctx.chatting = False
        ctx.rag.clear_chat_history()
        console.clear()
        _print_welcome()
        return True

    # --- Query ---
    if cmd.startswith("/query"):
        question = cmd[6:].strip()
        ctx.chatting = False
        ctx.rag.clear_chat_history()
        if not question:
            console.print("[dim]Usage: /query <your question>[/]")
            return True
        handled = await _handle_mention_query(question, ctx)
        if not handled:
            await _run_query_all(question, ctx)
        return True

    # --- Notes ---
    if cmd == "/notes copy":
        session_path = _get_notes_session(ctx)
        if session_path:
            content = _latest_notes_content(session_path)
            if not content:
                console.print("[dim]No notes yet. Run /notes to generate them first.[/]")
            elif _copy_to_clipboard(content):
                console.print("[dim]Notes copied to clipboard.[/]")
            else:
                console.print("[red]No clipboard tool found (wl-copy or xclip required).[/]")
        return True

    if cmd == "/notes export html":
        session_path = _get_notes_session(ctx)
        if session_path:
            content = _latest_notes_content(session_path)
            if not content:
                console.print("[dim]No notes yet. Run /notes to generate them first.[/]")
            else:
                meta = get_session_meta(session_path)
                slug = meta.get("slug") or meta.get("name") or session_path.stem
                export_path = Path.home() / f"{slug}_notes.html"
                export_path.write_text(_notes_to_html(content), encoding="utf-8")
                console.print(f"[dim]Notes exported to: {export_path}[/]")
                import subprocess
                subprocess.Popen(["xdg-open", str(export_path)])
        return True

    if cmd == "/notes export":
        session_path = _get_notes_session(ctx)
        if session_path:
            content = _latest_notes_content(session_path)
            if not content:
                console.print("[dim]No notes yet. Run /notes to generate them first.[/]")
            else:
                meta = get_session_meta(session_path)
                slug = meta.get("slug") or meta.get("name") or session_path.stem
                export_path = Path.home() / f"{slug}_notes.md"
                export_path.write_text(content, encoding="utf-8")
                console.print(f"[dim]Notes exported to: {export_path}[/]")
        return True

    if cmd == "/notes":
        await _do_notes(ctx, force_regen=False)
        return True

    if cmd.startswith("/notes "):
        template_arg = cmd[7:].strip()
        if template_arg and template_arg not in ("copy", "export", "export html"):
            await _do_notes_with_template(ctx, template_arg)
            return True

    if cmd == "/regen":
        await _do_notes(ctx, force_regen=True)
        return True

    # --- Transcript management ---
    if cmd.startswith("/name "):
        new_name = cmd[6:].strip()
        if not new_name:
            console.print("[dim]Usage: /name <transcript name>[/]")
        elif ctx.latest_transcript_path and ctx.latest_transcript_path.exists():
            new_path = rename_transcript(ctx.latest_transcript_path, new_name.replace(" ", "_"))
            ctx.latest_transcript_path = new_path
            console.print(f"[dim]Renamed to: {new_path.name}[/]")
        else:
            console.print("[dim]No recent transcript to rename.[/]")
        return True

    if cmd in ("/resume", "/sessions", "/history", "/transcripts", "/transcript"):
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No sessions found. Use /start to record a meeting.[/]")
            return True
        selected = pick_session(sessions, active=ctx.active_session_path)
        if selected:
            ctx.active_session_path = selected
            meta = get_session_meta(selected)
            ctx.active_session_name = meta.get("name") or selected.stem
            data = read_session(selected)
            n = len(data["transcripts"])
            name = meta.get("name", selected.stem)
            console.print(f"[dim]Active session: {name} ({n} recording{'s' if n != 1 else ''})[/]")
            _print_duration_bar(selected)
        return True

    if cmd.startswith("/transcript ") or cmd.startswith("/history ") or cmd.startswith("/sessions "):
        prefix_len = len(cmd.split()[0]) + 1
        identifier = cmd[prefix_len:].strip()
        if not identifier:
            console.print("[dim]Usage: /sessions <name or number>[/]")
            return True
        sessions = list_sessions()
        target = None
        try:
            idx = int(identifier) - 1
            if 0 <= idx < len(sessions):
                target = sessions[idx]
        except ValueError:
            search = identifier.lower().replace(" ", "_")
            for p in sessions:
                if search in p.stem.lower():
                    target = p
                    break
        if target:
            ctx.active_session_path = target
            meta = get_session_meta(target)
            ctx.active_session_name = meta.get("name") or target.stem
            data = read_session(target)
            n = len(data["transcripts"])
            name = meta.get("name", target.stem)
            console.print(f"[dim]Active session: {name} ({n} recording{'s' if n != 1 else ''})[/]")
            _print_duration_bar(target)
        else:
            console.print(f"[dim]Session not found: {identifier}[/]")
        return True

    # --- Delete session ---
    if cmd == "/delete":
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No sessions to delete.[/]")
            return True
        selected = pick_session(sessions, active=ctx.active_session_path)
        if not selected:
            return True
        meta = get_session_meta(selected)
        name = meta.get("name") or selected.stem
        console.print(f"[bold red]Delete '{name}'? This cannot be undone.[/] [y/N] ", end="")
        try:
            answer = input("").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return True
        if answer != "y":
            console.print("[dim]Cancelled.[/]")
            return True
        selected.unlink()
        console.print(f"[dim]Deleted: {name}[/]")
        if ctx.active_session_path == selected:
            ctx.active_session_path = None
            ctx.active_session_name = None
        ctx.rag.refresh()
        return True

    # --- Model ---
    if cmd == "/model":
        result = pick_model()
        if result:
            provider, model = result
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_MODEL"] = model
            _update_env_file("LLM_PROVIDER", provider)
            _update_env_file("LLM_MODEL", model)
            config = _load_config()
            config["llm_provider"] = provider
            config["llm_model"] = model
            _save_config(config)
            ctx.rag._vectorstore = None
            ctx.rag._qa_chain = None
            label = MODEL_REGISTRY[provider]["label"]
            console.print(f"[dim]Model set to {label}: {model}[/]")
        return True

    if cmd == "/transcribe":
        model_id = pick_whisper_model()
        if model_id:
            os.environ["WHISPER_MODEL"] = model_id
            _update_env_file("WHISPER_MODEL", model_id)
            config = _load_config()
            config["whisper_model"] = model_id
            _save_config(config)
            console.print(f"[dim]Whisper model set to {model_id} (takes effect on next recording)[/]")
        return True

    # --- Rename active session ---
    if cmd.startswith("/rename"):
        new_title = cmd[7:].strip()
        if not new_title:
            console.print("[dim]Usage: /rename New Title[/]")
        elif not ctx.active_session_path:
            console.print("[dim]No active session. Select one with /sessions first.[/]")
        else:
            append_rename(ctx.active_session_path, new_title)
            ctx.active_session_name = new_title
            console.print(f"[dim]Session renamed to: {new_title}[/]")
        return True

    # --- Misc ---
    if cmd == "/cleanup-recordings":
        recordings = list_recordings()
        if not recordings:
            console.print("[dim]No recordings to clean up.[/]")
            return True
        count, total_bytes = delete_all_recordings()
        size_str = f"{total_bytes / 1_048_576:.1f} MB" if total_bytes >= 1_048_576 else f"{total_bytes / 1024:.1f} KB"
        console.print(f"[dim]Deleted {count} recording{'s' if count != 1 else ''} ({size_str} freed).[/]")
        return True

    if cmd == "/verbose":
        ctx.verbose = not ctx.verbose
        console.print(f"[dim]Verbose mode: {'ON' if ctx.verbose else 'OFF'}[/]")
        return True

    if cmd == "/version":
        from openmic.version import get_version, get_latest_version, detect_install_method
        current = get_version()
        config = _load_config()
        latest = get_latest_version(config)
        _save_config(config)
        method = detect_install_method()
        console.print(f"\nopenmic [bold]v{current}[/]")
        if method != "unknown":
            console.print(f"[dim]Installed via: {method}[/]")
        if latest:
            if latest == current:
                console.print("[dim]Up to date[/]")
            else:
                console.print(f"[yellow]Update available: v{latest}[/]")
                console.print("[dim]Run 'openmic update' to upgrade.[/]")
        console.print()
        return True

    if cmd == "/help":
        console.print()
        print_help()
        return True

    if cmd == "/exit":
        return False

    # Bare text (no slash) = query
    if not cmd.startswith("/"):
        handled = await _handle_mention_query(cmd, ctx)
        if not handled:
            await _run_query_all(cmd, ctx)
        return True

    console.print(f"[dim]Unknown command: {cmd}  (type /help for commands)[/]")
    return True


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# openmic color scheme
# ---------------------------------------------------------------------------

TEAL        = "#00d4aa"   # primary accent (from banner)
TEAL_DIM    = "#007a63"   # muted teal for separators
GHOST_TEXT  = "#6c7a8a"   # inline auto-suggestion (light enough to read on dark bg)
META_DIM    = "#7a8899"   # description text (unselected) — readable on dark bg
META_BRIGHT = "#c0ccd8"   # description text (selected)

OPENMIC_STYLE = {
    # Separator lines and prompt character
    "separator":    f"{TEAL_DIM}",
    "prompt":       f"{TEAL} bold",
    "session-label": f"bg:{TEAL} #000000 bold",

    # Ghost-text auto-suggestion (appears inline after cursor)
    "auto-suggestion": GHOST_TEXT,

    # Completion popup — no background, no border, uses terminal bg
    "completion-menu":                           "noinherit",
    "completion-menu.completion":                f"fg:{META_DIM} noinherit",
    "completion-menu.completion.current":        f"fg:{TEAL} bold noinherit",
    "completion-menu.meta.completion":           f"fg:{META_DIM} noinherit",
    "completion-menu.meta.completion.current":   f"fg:{META_BRIGHT} noinherit",
    "scrollbar.background":                      "noinherit",
    "scrollbar.button":                          f"fg:{TEAL_DIM} noinherit",
}


class _CommandAutoSuggest:
    """
    Fish-shell style: first matching command appears as ghost text after the cursor.
    Right-arrow or End to accept. Tab to show full popup with descriptions.
    """
    def get_suggestion(self, buffer, document):
        from prompt_toolkit.auto_suggest import Suggestion
        text = document.text_before_cursor
        if not text:
            return None

        if text.startswith("/"):
            for cmd, _ in HELP_COMMANDS:
                if cmd and cmd.lower().startswith(text.lower()) and len(cmd) > len(text):
                    return Suggestion(cmd[len(text):])

        return None


class _CommandCompleter:
    """Full popup completer — shows matching commands with descriptions."""

    def get_completions(self, document, complete_event):
        from prompt_toolkit.completion import Completion
        text = document.text_before_cursor

        if text.startswith("/"):
            for cmd, desc in HELP_COMMANDS:
                if cmd and cmd.lower().startswith(text.lower()):
                    # start_position=-len(text) anchors the popup to the
                    # left edge of the input text, not the cursor position.
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )
            return

        at_idx = text.rfind("@")
        if at_idx != -1:
            prefix = text[at_idx + 1:]
            try:
                for path in list_transcripts():
                    meta    = _parse_transcript_meta(path)
                    display = format_transcript_title(path.stem[:16], meta["name"])
                    if prefix.lower() in display.lower():
                        yield Completion(
                            f"[{display}]",
                            start_position=-len(prefix),
                            display=f"@[{display}]",
                            display_meta="transcript",
                        )
            except Exception:
                pass


def _print_welcome() -> None:
    model_label = UsageTracker.current_model_label()
    hint = f"  {model_label}" if model_label else ""
    console.print(f"\n[bold {TEAL}]openmic[/][dim]{hint}  —  /help for commands[/]\n")


async def repl_loop(ctx: ReplContext) -> None:
    """Interactive REPL — reads commands until /exit or EOF."""
    import shutil
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.key_binding.bindings.emacs import load_emacs_bindings
    from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window, ConditionalContainer
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.processors import BeforeInput, AppendAutoSuggestion
    from prompt_toolkit.auto_suggest import AutoSuggest
    from prompt_toolkit.completion import Completer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.styles import Style

    _print_welcome()

    history   = InMemoryHistory()
    _suggest  = _CommandAutoSuggest()
    _completer = _CommandCompleter()
    _comp_idx    = [0]   # highlighted completion (absolute index)
    _view_offset = [0]   # first visible row in 6-item viewport
    COMP_WINDOW  = 6

    class _AutoSuggest(AutoSuggest):
        def get_suggestion(self, buffer, document):
            from prompt_toolkit.auto_suggest import Suggestion
            s = _suggest.get_suggestion(buffer, document)
            return s

    class _Completer(Completer):
        def get_completions(self, document, complete_event):
            yield from _completer.get_completions(document, complete_event)

    buf = Buffer(
        history=history,
        auto_suggest=_AutoSuggest(),
        name="default",
    )

    def _on_text_changed(_event) -> None:
        """Reset highlight to top whenever the user edits the input."""
        _comp_idx[0]    = 0
        _view_offset[0] = 0

    buf.on_text_changed += _on_text_changed

    # ── Completion display above separator ────────────────────────────────────

    def _get_template_commands() -> list[tuple[str, str]]:
        """Return ('/notes <id>', description) pairs for all available templates."""
        try:
            from openmic.templates import TemplateManager
            tm = TemplateManager()
            return [
                (f"/notes {t.id}", t.description or t.name)
                for t in tm.get_builtin_templates() + tm.get_user_templates()
            ]
        except Exception:
            return []

    def _slash_matches() -> list[tuple[str, str]]:
        text = buf.document.text_before_cursor
        if not text.startswith("/"):
            return []
        tl = text.lower()
        # When typing "/notes " show template completions instead of subcommand list
        if tl.startswith("/notes "):
            return [
                (cmd, desc) for cmd, desc in _get_template_commands()
                if cmd.lower().startswith(tl)
            ]
        return [
            (cmd, desc) for cmd, desc in HELP_COMMANDS
            if cmd and cmd.lower().startswith(tl)
        ]

    def _completions_text():
        matches = _slash_matches()
        if not matches:
            return []
        n = len(matches)
        # Clamp idx/offset in case text change shrank the list
        if _comp_idx[0] >= n:
            _comp_idx[0] = 0
        _view_offset[0] = min(_view_offset[0], max(0, n - COMP_WINDOW))
        idx     = _comp_idx[0]
        visible = matches[_view_offset[0]:_view_offset[0] + COMP_WINDOW]
        lines   = []
        for i, (cmd, desc) in enumerate(visible):
            abs_i = i + _view_offset[0]
            if abs_i == idx:
                lines.append(("class:completion-menu.completion.current",     f"   {cmd:<22}"))
                lines.append(("class:completion-menu.meta.completion.current", f"   {desc}\n"))
            else:
                lines.append(("class:completion-menu.completion",              f"   {cmd:<22}"))
                lines.append(("class:completion-menu.meta.completion",          f"   {desc}\n"))
        return lines

    def _sep_text():
        cols = shutil.get_terminal_size((80, 24)).columns
        return [("class:separator", "─" * cols)]

    # ── Key bindings ──────────────────────────────────────────────────────────

    kb = KeyBindings()

    @kb.add("enter")
    def _accept(event):
        matches = _slash_matches()
        result = matches[_comp_idx[0]][0] if matches else buf.text
        event.app.exit(result=result)

    def _has_completions() -> bool:
        return bool(_slash_matches())

    def _cycle(delta: int) -> None:
        """Move highlight by delta rows; scroll viewport to keep it visible."""
        matches = _slash_matches()
        if not matches:
            return
        n = len(matches)
        _comp_idx[0] = (_comp_idx[0] + delta) % n
        # Scroll viewport so the highlighted row stays within the window
        if _comp_idx[0] < _view_offset[0]:
            _view_offset[0] = _comp_idx[0]
        elif _comp_idx[0] >= _view_offset[0] + COMP_WINDOW:
            _view_offset[0] = _comp_idx[0] - COMP_WINDOW + 1

    @kb.add("tab")
    def _tab(event):
        matches = _slash_matches()
        if matches:
            # Fill highlighted completion into the buffer
            cmd = matches[_comp_idx[0]][0]
            buf.set_document(Document(cmd, len(cmd)))
        else:
            buf.start_completion(select_first=False)

    @kb.add("s-tab")
    def _shift_tab(event):
        if _has_completions():
            _cycle(-1)
        else:
            # Quick-start recording shortcut
            event.app.exit(result="/start")

    # Arrow keys navigate completions when visible; fall through to emacs
    # history bindings (via load_emacs_bindings) when no completions shown.
    @kb.add("down", filter=Condition(_has_completions))
    def _down(event):
        _cycle(+1)

    @kb.add("up", filter=Condition(_has_completions))
    def _up(event):
        _cycle(-1)

    @kb.add("escape")
    def _escape(event):
        # Clear input on Escape (no completion popup to dismiss)
        buf.reset()

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit(exception=KeyboardInterrupt)

    @kb.add("c-d")
    def _ctrl_d(event):
        event.app.exit(exception=EOFError)

    # ── Layout: input → completions ──────────────────────────────────────────
    # Separator is printed by the REPL loop (outside pt) so erase_when_done
    # only clears input + completions, leaving the separator visible during
    # command execution.

    layout = Layout(
        HSplit([
            Window(
                content=BufferControl(
                    buffer=buf,
                    input_processors=[
                        BeforeInput("›  ", style="class:prompt"),
                        AppendAutoSuggestion(),
                    ],
                    include_default_input_processors=False,
                ),
                height=1,
                wrap_lines=False,
            ),
            ConditionalContainer(
                Window(
                    content=FormattedTextControl(_completions_text, focusable=False),
                    height=Dimension(min=0, max=COMP_WINDOW),
                ),
                filter=Condition(lambda: bool(_slash_matches())),
            ),
        ]),
        focused_element=buf,
    )

    app = Application(
        layout=layout,
        key_bindings=merge_key_bindings([
            load_basic_bindings(),
            load_emacs_bindings(),
            kb,
        ]),
        style=Style.from_dict(OPENMIC_STYLE),
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )

    _ctrl_c_warned = False

    while True:
        _comp_idx[0]    = 0
        _view_offset[0] = 0
        buf.reset()
        # Print separator here (outside pt) so erase_when_done only clears
        # input + completions — separator stays visible during command execution.
        cols = shutil.get_terminal_size((80, 24)).columns
        if ctx.active_session_name:
            clean = ctx.active_session_name.replace("_", " ").strip()
            if len(clean) > 28:
                clean = clean[:27] + "…"
            badge = f" {clean} "
            dashes = max(0, cols - len(badge))
            console.print(
                f"[{TEAL_DIM}]{'─' * dashes}[/]"
                f"[bold black on {TEAL}]{badge}[/]",
                end="\n",
            )
        else:
            console.print(f"[{TEAL_DIM}]{'─' * cols}[/]")
        try:
            cmd = await app.run_async()
            _ctrl_c_warned = False
        except KeyboardInterrupt:
            if _ctrl_c_warned:
                console.print("\n[dim]Goodbye![/]")
                break
            _ctrl_c_warned = True
            console.print("\n[dim]Press Ctrl+C again to exit, or type /exit[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/]")
            break
        try:
            running = await handle_command(cmd.strip(), ctx)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Cancelled.[/]")
            continue
        if not running:
            console.print("[dim]Goodbye![/]")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_KNOWN_SUBCOMMANDS = {"record", "query", "notes", "list", "model", "update", "setup", "resume"}

_HELP_TEXT = """\
Usage: openmic [command] [args]

Commands:
  openmic                        Interactive REPL
  openmic resume                 Pick a session and enter REPL
  openmic record [name]          Record a meeting, then enter REPL
  openmic "query text"           Run a one-shot query and exit
  openmic query "query text"     Same (explicit form)
  openmic notes                  Show or generate notes for latest transcript
  openmic list                   List saved transcripts
  openmic model                  Interactive model picker
  openmic model <provider> <id>  Set model directly (e.g. anthropic claude-sonnet-4-6)
  openmic update                 Self-update
  openmic setup                  Re-run setup wizard
  openmic --version              Show version

Transcription is handled locally via whisper.cpp — no API key required.
Set WHISPER_MODEL in .env to change the model (default: large-v3-turbo).
"""


def main() -> None:
    """Entry point for the OpenMic application."""
    import sys

    argv = sys.argv[1:]

    # ── Fast-path: no config needed ─────────────────────────────────────────
    if not argv:
        _run_interactive()
        return

    first, rest = argv[0], argv[1:]

    if first in ("--version", "-V"):
        from openmic.version import get_version
        print(f"openmic {get_version()}")
        return

    if first in ("--help", "-h"):
        print(_HELP_TEXT)
        return

    if first == "update":
        from openmic.version import run_update
        run_update()
        return

    if first == "setup":
        from openmic.setup import run_setup
        run_setup()
        return

    # ── Bare string = one-shot query ─────────────────────────────────────────
    if not first.startswith("-") and first not in _KNOWN_SUBCOMMANDS:
        _run_oneshot_query(" ".join(argv))
        return

    # ── Subcommands ──────────────────────────────────────────────────────────
    if first == "resume":
        _run_interactive(resume=True)
        return

    if first in ("record", "--record", "-r"):
        session_name = "_".join(rest) if rest else None
        _run_interactive(record=True, session_name=session_name)
        return

    if first in ("query", "--query", "-q"):
        query_text = " ".join(rest)
        if query_text:
            _run_oneshot_query(query_text)
        else:
            _run_interactive()
        return

    if first == "notes":
        _run_oneshot_notes()
        return

    if first == "list":
        _run_list_transcripts()
        return

    if first == "model":
        _run_set_model(rest)
        return

    console.print(f"[dim]Unknown command: {first}[/]\n")
    print(_HELP_TEXT)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> dict | None:
    """Load config, run setup if needed, restore LLM env. Returns config or None."""
    config = _load_config()
    if not config.get("setup_complete"):
        from openmic.setup import run_setup
        run_setup()
        config = _load_config()
        if not config.get("setup_complete"):
            return None
    # UI-picked settings override .env so choices persist across sessions.
    if config.get("llm_provider"):
        os.environ["LLM_PROVIDER"] = config["llm_provider"]
    if config.get("llm_model"):
        os.environ["LLM_MODEL"] = config["llm_model"]
    if config.get("whisper_model"):
        os.environ["WHISPER_MODEL"] = config["whisper_model"]
    return config


def _run_interactive(
    record: bool = False,
    session_name: str | None = None,
    resume: bool = False,
) -> None:
    """Start interactive session (optional recording or resume first)."""
    config = _bootstrap()
    if config is None:
        return

    print_banner()
    _check_for_updates_sync(config)

    rag = TranscriptRAG()
    ctx = ReplContext(rag=rag)

    async def _run():
        if resume:
            await handle_command("/resume", ctx)
        elif record:
            path = await recording_mode(session_name=session_name, ctx=ctx)
            if path:
                ctx.latest_transcript_path = path
        await repl_loop(ctx)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def _run_oneshot_query(query_text: str) -> None:
    """Run a single RAG query, print the answer, and exit."""
    config = _bootstrap()
    if config is None:
        return

    transcripts = list_transcripts()
    if not transcripts:
        console.print("[dim]No transcripts available. Use 'openmic record' to record a meeting.[/]")
        return

    rag = TranscriptRAG()
    ctx = ReplContext(rag=rag)

    async def _run():
        await _run_query_all(query_text, ctx)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def _run_oneshot_notes() -> None:
    """Generate (or show cached) notes for the latest transcript and exit."""
    config = _bootstrap()
    if config is None:
        return

    transcript_path = get_latest_transcript()
    if transcript_path is None:
        console.print("[dim]No transcripts available.[/]")
        return

    rag = TranscriptRAG()
    ctx = ReplContext(rag=rag)

    async def _run():
        await _generate_notes_for_path(transcript_path, ctx=ctx)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def _run_list_transcripts() -> None:
    """Print the transcript list and exit."""
    config = _bootstrap()
    if config is None:
        return

    transcripts = list_transcripts()
    if not transcripts:
        console.print("[dim]No transcripts found.[/]")
        return

    t = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    last_header = None
    for i, path in enumerate(transcripts, 1):
        meta = _parse_transcript_meta(path)
        dt = meta["datetime"]
        if dt:
            header = _date_header(dt)
            if header != last_header:
                t.add_row("", f"[bold]{header}[/]", "")
                last_header = header
            has_notes = (NOTES_DIR / (meta["stem"] + "_notes.md")).exists()
            star = "" if has_notes else "[bold #00d4aa]*[/]"
            title = format_transcript_title(path.stem[:16], meta["name"])
            time_str = dt.strftime("%-I:%M %p")
            t.add_row(f"[dim]{i}[/]", f"{star} {title}", f"[dim]{time_str}[/]")
        else:
            t.add_row(f"[dim]{i}[/]", path.stem, "")
    console.print(t)
    console.print("[dim]  * = notes not yet generated[/]")


def _run_set_model(args: list[str]) -> None:
    """Set model from CLI args or launch interactive picker."""
    config = _bootstrap()
    if config is None:
        return

    if len(args) >= 2:
        provider, model_id = args[0], args[1]
        if provider not in MODEL_REGISTRY:
            console.print(f"[dim]Unknown provider: {provider}[/]")
            console.print(f"[dim]Available: {', '.join(MODEL_REGISTRY)}[/]")
            return
        # For ollama, accept any model string (user controls their install)
        if provider != "ollama":
            valid_ids = [m for m, _ in MODEL_REGISTRY[provider]["models"]]
            if model_id not in valid_ids:
                console.print(f"[dim]Unknown model: {model_id}[/]")
                console.print(f"[dim]Available for {provider}: {', '.join(valid_ids)}[/]")
                return
        env_key = MODEL_REGISTRY[provider]["env_key"]
        if env_key and not os.environ.get(env_key):
            console.print(f"[dim]{env_key} not set — add it to .env[/]")
            return
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL"] = model_id
        _update_env_file("LLM_PROVIDER", provider)
        _update_env_file("LLM_MODEL", model_id)
        cfg = _load_config()
        cfg["llm_provider"] = provider
        cfg["llm_model"] = model_id
        _save_config(cfg)
        label = MODEL_REGISTRY[provider]["label"]
        console.print(f"Model set to [bold]{label}[/]: [bold #00d4aa]{model_id}[/]")
        return

    if len(args) == 1:
        # Provider only — pick model interactively from that provider
        provider = args[0]
        if provider not in MODEL_REGISTRY:
            console.print(f"[dim]Unknown provider: {provider}[/]")
            return
        info = MODEL_REGISTRY[provider]
        console.print(f"[bold]{info['label']}[/] models:")
        for i, (mid, desc) in enumerate(info["models"], 1):
            console.print(f"  {i}. [bold]{mid}[/]  [dim]{desc}[/]")
        try:
            choice = input("Pick number (or Enter to cancel): ").strip()
            if not choice:
                return
            idx = int(choice) - 1
            if 0 <= idx < len(info["models"]):
                model_id = info["models"][idx][0]
                os.environ["LLM_PROVIDER"] = provider
                os.environ["LLM_MODEL"] = model_id
                cfg = _load_config()
                cfg["llm_provider"] = provider
                cfg["llm_model"] = model_id
                _save_config(cfg)
                console.print(f"Model set to [bold]{info['label']}[/]: [bold #00d4aa]{model_id}[/]")
        except (ValueError, KeyboardInterrupt):
            pass
        return

    # No args — full interactive picker
    result = pick_model()
    if result:
        provider, model_id = result
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL"] = model_id
        cfg = _load_config()
        cfg["llm_provider"] = provider
        cfg["llm_model"] = model_id
        _save_config(cfg)
        label = MODEL_REGISTRY[provider]["label"]
        console.print(f"Model set to [bold]{label}[/]: [bold #00d4aa]{model_id}[/]")


def _check_for_updates_sync(config: dict) -> None:
    """Quick synchronous update check (cached — no network if recently checked)."""
    try:
        from openmic.version import get_version, get_latest_version
        current = get_version()
        latest = get_latest_version(config)
        _save_config(config)
        if latest and latest != current:
            console.print(f"[dim]Update available: v{current} → v{latest}  run 'openmic update'[/]\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
