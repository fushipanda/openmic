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

from dotenv import load_dotenv

CONFIG_DIR = Path.home() / ".config" / "openmic"
CONFIG_FILE = CONFIG_DIR / "settings.json"

console = Console()


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

load_dotenv()

TRANSCRIPTION_REGISTRY: dict[str, dict] = {
    "elevenlabs": {
        "label": "ElevenLabs Scribe (cloud)",
        "env_key": "ELEVENLABS_API_KEY",
        "models": [
            ("scribe_v1", "Standard"),
        ],
    },
    "local": {
        "label": "Whisper / whisper.cpp (local — private)",
        "env_key": None,
        "models": [
            ("tiny.en",        "Tiny — fastest"),
            ("base.en",        "Base"),
            ("small.en",       "Small — recommended"),
            ("medium.en",      "Medium"),
            ("large-v3",       "Large — most accurate"),
            ("large-v3-turbo", "Large Turbo — fast + accurate"),
        ],
    },
}


def _get_transcribers():
    """Return (RealtimeTranscriber, BatchTranscriber) based on current TRANSCRIPTION_BACKEND."""
    backend = os.environ.get("TRANSCRIPTION_BACKEND", "elevenlabs").lower()
    if backend == "local":
        from openmic.local_transcribe import LocalRealtimeTranscriber, LocalBatchTranscriber
        return LocalRealtimeTranscriber, LocalBatchTranscriber
    from openmic.transcribe import RealtimeTranscriber, BatchTranscriber
    return RealtimeTranscriber, BatchTranscriber

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
from openmic.rag import TranscriptRAG
from openmic.notes import generate_meeting_notes, get_existing_notes

BANNER = """\
 ██████  ██████  ███████ ███    ██ ███    ███ ██  ██████
██    ██ ██   ██ ██      ████   ██ ████  ████ ██ ██
██    ██ ██████  █████   ██ ██  ██ ██ ████ ██ ██ ██
██    ██ ██      ██      ██  ██ ██ ██  ██  ██ ██ ██
 ██████  ██      ███████ ██   ████ ██      ██ ██  ██████"""

HELP_COMMANDS = [
    ("/start [name]",    "Start recording (optionally with session name)"),
    ("/stop",            "Stop recording and run batch transcription"),
    ("/pause",           "Pause recording"),
    ("/resume",          "Resume a paused recording"),
    ("/history",         "List saved transcripts"),
    ("/transcript <n>",  "View transcript by number or name"),
    ("/query <question>","Ask a question across all transcripts"),
    ("/notes",           "Generate notes (with template selection)"),
    ("/regen",           "Regenerate notes using the saved template"),
    ("/model",           "Select LLM provider and model"),
    ("/transcribe",      "Select transcription backend (ElevenLabs or local Whisper)"),
    ("/name <name>",     "Rename the latest transcript"),
    ("/cleanup-recordings", "Delete all saved recordings"),
    ("/verbose",         "Toggle debug output"),
    ("/version",         "Show version and check for updates"),
    ("/exit",            "Quit OpenMic"),
]


class UsageTracker:
    """Tracks ElevenLabs and LLM usage for the current session."""

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
            sys.stdout.write(f"\033[{len(rows) + 1}A")
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


def pick_transcription_backend() -> tuple[str, str] | None:
    """Arrow-key transcription backend picker. Returns (backend_key, model_id) or None."""
    current_backend = os.environ.get("TRANSCRIPTION_BACKEND", "elevenlabs").lower()
    current_model   = os.environ.get("WHISPER_MODEL", "")

    rows: list[dict] = []
    for bkey, info in TRANSCRIPTION_REGISTRY.items():
        rows.append({"kind": "header", "text": info["label"]})
        for model_id, desc in info["models"]:
            is_current = bkey == current_backend and (
                model_id == current_model or (bkey == "elevenlabs" and not current_model)
            )
            rows.append({
                "kind":      "item",
                "primary":   model_id,
                "secondary": desc,
                "value":     (bkey, model_id),
                "current":   is_current,
            })

    try:
        result = _arrow_select(rows)
    except KeyboardInterrupt:
        return None

    if result is None:
        return None

    bkey, model_id = result
    env_key = TRANSCRIPTION_REGISTRY[bkey]["env_key"]
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
    return bkey, model_id


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
# Query
# ---------------------------------------------------------------------------

async def _run_query_on_path(question: str, path: Path, ctx: ReplContext) -> None:
    """Run RAG query against a specific transcript."""
    console.print(f"[dim]Searching {path.stem}...[/]")
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, lambda: ctx.rag.query_file(question, path))
        ctx.usage.add_llm_call()
        console.print()
        console.print(f"[bold #00d4aa]  >[/] {question}")
        console.print()
        console.print(RichMarkdown(answer))
    except Exception as e:
        console.print(f"[red]Query error: {e}[/]")


async def _run_query_all(question: str, ctx: ReplContext) -> None:
    """Run RAG query across all transcripts."""
    transcripts = list_transcripts()
    if not transcripts:
        console.print("[dim]No transcripts available. Use /start to record a meeting.[/]")
        return

    if not ctx.chatting:
        ctx.chatting = True
        ctx.rag.clear_chat_history()

    console.print(f"[dim]Searching across {len(transcripts)} transcript{'s' if len(transcripts) != 1 else ''}...[/]")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: ctx.rag.query(question))
        ctx.usage.add_llm_call()
        answer = result["answer"]
        sources = result.get("sources", [])

        console.print()
        console.print(f"[bold #00d4aa]  >[/] {question}")
        console.print()
        console.print(RichMarkdown(answer))
        if sources:
            console.print(f"[dim]Sources: {', '.join(sources)}[/]")
        console.print()
    except Exception as e:
        console.print(f"[red]Query error: {e}[/]")


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

async def _generate_notes_for_path(
    path: Path, template_id: str = "default", force_regen: bool = False, ctx: ReplContext | None = None
) -> None:
    """Generate (or load cached) meeting notes for a transcript."""
    if force_regen:
        notes_path = NOTES_DIR / (path.stem + "_notes.md")
        if notes_path.exists():
            notes_path.unlink()

    existing = get_existing_notes(path)
    will_use_cache = existing is not None and existing[2] == template_id

    if will_use_cache:
        console.print("[dim]Loading saved notes...[/]")
    else:
        console.print(f"[dim]Generating notes ({template_id})...[/]")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: generate_meeting_notes(path, template_id))
        notes_content, notes_path, used_cache = result
        if not used_cache and ctx:
            ctx.usage.add_llm_call()
        console.print()
        console.print(RichMarkdown(notes_content))
        console.print()
        console.print(f"[dim]Saved to: {notes_path}[/]")
    except Exception as e:
        console.print(f"[red]Error generating notes: {e}[/]")


async def _do_notes(ctx: ReplContext, force_regen: bool = False) -> None:
    """Handle /notes and /regen — pick transcript, pick template, generate."""
    transcripts = list_transcripts()
    if not transcripts:
        console.print("[dim]No transcripts available.[/]")
        return

    if len(transcripts) == 1:
        path = transcripts[0]
    else:
        console.print()
        path = pick_transcript(transcripts)
        if path is None:
            return

    if force_regen:
        # For /regen: check if existing notes have a saved template
        existing = get_existing_notes(path)
        if existing is not None:
            _, _, saved_template = existing
            template_id = saved_template or "default"
        else:
            console.print()
            template_id = pick_template()
        await _generate_notes_for_path(path, template_id, force_regen=True, ctx=ctx)
        return

    # For /notes: show cached if present, otherwise pick template
    existing = get_existing_notes(path)
    if existing is not None:
        _, _, saved_template = existing
        template_id = saved_template or "default"
        await _generate_notes_for_path(path, template_id, ctx=ctx)
    else:
        console.print()
        template_id = pick_template()
        await _generate_notes_for_path(path, template_id, ctx=ctx)


# ---------------------------------------------------------------------------
# Recording mode
# ---------------------------------------------------------------------------

async def recording_mode(session_name: str | None = None, ctx: ReplContext | None = None) -> Path | None:
    """
    Record audio and stream live transcript to terminal.
    Ctrl+C stops recording and triggers batch transcription.
    Returns the saved transcript path, or None on failure.
    """
    lines: list[str] = []
    partial_holder: list[str] = [""]  # mutable for callback closure

    usage = ctx.usage if ctx else UsageTracker()
    verbose = ctx.verbose if ctx else False

    def on_audio_chunk(audio_bytes: bytes) -> None:
        usage.add_audio_bytes(len(audio_bytes))
        transcriber.send_audio_chunk(audio_bytes)

    def on_partial(text: str) -> None:
        partial_holder[0] = text

    def on_committed(text: str) -> None:
        if partial_holder[0]:
            partial_holder[0] = ""
        lines.append(text)
        console.print(text)

    def on_error(msg: str) -> None:
        console.print(f"[dim][{msg}][/]")

    def on_debug(msg: str) -> None:
        if verbose:
            console.print(f"[dim]  dbg: {msg}[/]")

    RealtimeTranscriberCls, BatchTranscriberCls = _get_transcribers()

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

    session_info = f" [{session_name}]" if session_name else ""
    console.print()
    console.print(f"[bold #ff4757]◉ RECORDING{session_info}[/]  [dim]Ctrl+C to stop[/]")
    console.print()

    try:
        while True:
            await asyncio.sleep(0.1)
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

    console.print("[dim]Running batch transcription with diarization...[/]")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: BatchTranscriberCls().transcribe_file(str(returned_wav))
        )
        segments = BatchTranscriberCls.parse_diarized_result(result)
        transcript_path = save_transcript(segments, session_name)

        console.print()
        for seg in segments:
            speaker = seg.get("speaker", "Speaker")
            text = seg.get("text", "")
            console.print(f"[bold #00d4aa][{speaker}][/] {text}")

        console.print()
        console.print(f"[dim]Saved to: {transcript_path.name}[/]")

        # Offer to name the session if none was given
        if not session_name:
            try:
                name = input("Name this session (Enter to skip): ").strip()
                if name:
                    transcript_path = rename_transcript(transcript_path, name.replace(" ", "_"))
                    console.print(f"[dim]Renamed to: {transcript_path.name}[/]")
            except KeyboardInterrupt:
                pass

        return transcript_path

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
    if cmd == "/start" or cmd.startswith("/start "):
        session_name = cmd[7:].strip().replace(" ", "_") if cmd.startswith("/start ") else None
        path = await recording_mode(session_name, ctx)
        if path:
            ctx.latest_transcript_path = path
            ctx.chatting = False
        return True

    if cmd in ("/stop", "/pause", "/resume"):
        console.print("[dim]Not currently recording. Use /start to begin.[/]")
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
    if cmd == "/notes":
        await _do_notes(ctx, force_regen=False)
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

    if cmd in ("/transcripts", "/history", "/transcript"):
        transcripts = list_transcripts()
        if not transcripts:
            console.print("[dim]No transcripts found.[/]")
            return True
        path = pick_transcript(transcripts)
        if path:
            content = path.read_text()
            console.print()
            console.print(RichMarkdown(content))
            console.print(f"[dim]— {path}[/]")
        return True

    if cmd.startswith("/transcript ") or cmd.startswith("/history "):
        prefix_len = len(cmd.split()[0]) + 1
        identifier = cmd[prefix_len:].strip()
        if not identifier:
            console.print("[dim]Usage: /transcript <name or number>[/]")
            return True
        transcripts = list_transcripts()
        target = None
        try:
            idx = int(identifier) - 1
            if 0 <= idx < len(transcripts):
                target = transcripts[idx]
        except ValueError:
            search = identifier.lower().replace(" ", "_")
            for p in transcripts:
                if search in p.stem.lower():
                    target = p
                    break
        if target:
            console.print()
            console.print(RichMarkdown(target.read_text()))
            console.print(f"[dim]— {target}[/]")
        else:
            console.print(f"[dim]Transcript not found: {identifier}[/]")
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
        result = pick_transcription_backend()
        if result:
            backend, model_id = result
            os.environ["TRANSCRIPTION_BACKEND"] = backend
            _update_env_file("TRANSCRIPTION_BACKEND", backend)
            config = _load_config()
            config["transcription_backend"] = backend
            if backend == "local":
                os.environ["WHISPER_MODEL"] = model_id
                _update_env_file("WHISPER_MODEL", model_id)
                config["whisper_model"] = model_id
            _save_config(config)
            label = TRANSCRIPTION_REGISTRY[backend]["label"]
            console.print(f"[dim]Transcription set to {label}[/]")
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
META_DIM    = "#3d4a5c"   # description text (unselected)
META_BRIGHT = "#8899aa"   # description text (selected)

OPENMIC_STYLE = {
    # Separator lines and prompt character
    "separator":  f"{TEAL_DIM}",
    "prompt":     f"{TEAL} bold",

    # Ghost-text auto-suggestion (appears inline after cursor)
    "auto-suggestion": GHOST_TEXT,

    # Completion popup — no background, uses terminal bg
    "completion-menu":                           "",
    "completion-menu.completion":                f"fg:{META_DIM}",
    "completion-menu.completion.current":        f"fg:{TEAL} bold",
    "completion-menu.meta.completion":           f"fg:{META_DIM}",
    "completion-menu.meta.completion.current":   f"fg:{META_BRIGHT}",
    "scrollbar.background":                      "",
    "scrollbar.button":                          f"fg:{TEAL_DIM}",
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
    """Full popup completer — Tab to trigger, shows all matches + descriptions."""

    def get_completions(self, document, complete_event):
        from prompt_toolkit.completion import Completion
        text = document.text_before_cursor

        if text.startswith("/"):
            for cmd, desc in HELP_COMMANDS:
                if cmd and cmd.lower().startswith(text.lower()):
                    yield Completion(cmd, start_position=-len(text), display_meta=desc)
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


async def repl_loop(ctx: ReplContext) -> None:
    """Interactive REPL — reads commands until /exit or EOF."""
    import shutil
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggest
    from prompt_toolkit.completion import Completer
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style

    class _PTAutoSuggest(AutoSuggest):
        _inner = _CommandAutoSuggest()
        def get_suggestion(self, buffer, document):
            return self._inner.get_suggestion(buffer, document)

    class _PTCompleter(Completer):
        _inner = _CommandCompleter()
        def get_completions(self, document, complete_event):
            yield from self._inner.get_completions(document, complete_event)

    model_label = UsageTracker.current_model_label()
    hint = f"  {model_label}" if model_label else ""
    console.print(f"\n[bold {TEAL}]openmic[/][dim]{hint}  —  /help for commands[/]\n")

    # Separator is printed by Rich before each prompt so prompt_toolkit's
    # scroll/reserve logic only needs to account for the single-line input + menu.
    def _sep() -> None:
        cols = shutil.get_terminal_size((80, 24)).columns
        console.print(f"[{TEAL_DIM}]{'─' * cols}[/]", end="")
        sys.stdout.write("\n")
        sys.stdout.flush()

    session: PromptSession = PromptSession(
        message=FormattedText([("class:prompt", "›  ")]),
        completer=_PTCompleter(),
        complete_while_typing=True,    # popup appears while typing
        auto_suggest=_PTAutoSuggest(), # ghost text inline too
        style=Style.from_dict(OPENMIC_STYLE),
        reserve_space_for_menu=8,
    )

    _ctrl_c_warned = False

    while True:
        _sep()
        try:
            cmd = await session.prompt_async()
            _ctrl_c_warned = False   # reset after a successful input
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
        console.print()
        running = await handle_command(cmd.strip(), ctx)
        if not running:
            console.print("[dim]Goodbye![/]")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_KNOWN_SUBCOMMANDS = {"record", "query", "notes", "list", "model", "transcribe", "update", "setup"}

_HELP_TEXT = """\
Usage: openmic [command] [args]

Commands:
  openmic                        Interactive REPL
  openmic record [name]          Record a meeting, then enter REPL
  openmic "query text"           Run a one-shot query and exit
  openmic query "query text"     Same (explicit form)
  openmic notes                  Show or generate notes for latest transcript
  openmic list                   List saved transcripts
  openmic model                  Interactive model picker
  openmic model <provider> <id>  Set model directly (e.g. anthropic claude-sonnet-4-6)
  openmic transcribe             Select transcription backend (cloud or local Whisper)
  openmic update                 Self-update
  openmic setup                  Re-run setup wizard
  openmic --version              Show version
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

    if first == "transcribe":
        _run_set_transcription(rest)
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
    if config.get("llm_provider"):
        os.environ.setdefault("LLM_PROVIDER", config["llm_provider"])
    if config.get("llm_model"):
        os.environ.setdefault("LLM_MODEL", config["llm_model"])
    if config.get("transcription_backend"):
        os.environ.setdefault("TRANSCRIPTION_BACKEND", config["transcription_backend"])
    if config.get("whisper_model"):
        os.environ.setdefault("WHISPER_MODEL", config["whisper_model"])
    return config


def _run_interactive(record: bool = False, session_name: str | None = None) -> None:
    """Start interactive session (optional recording first)."""
    config = _bootstrap()
    if config is None:
        return

    print_banner()
    _check_for_updates_sync(config)

    rag = TranscriptRAG()
    ctx = ReplContext(rag=rag)

    async def _run():
        if record:
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


def _run_set_transcription(args: list[str]) -> None:
    """Set transcription backend from CLI args or launch interactive picker."""
    config = _bootstrap()
    if config is None:
        return

    # No args — full interactive picker
    result = pick_transcription_backend()
    if result:
        backend, model_id = result
        os.environ["TRANSCRIPTION_BACKEND"] = backend
        _update_env_file("TRANSCRIPTION_BACKEND", backend)
        cfg = _load_config()
        cfg["transcription_backend"] = backend
        if backend == "local":
            os.environ["WHISPER_MODEL"] = model_id
            _update_env_file("WHISPER_MODEL", model_id)
            cfg["whisper_model"] = model_id
        _save_config(cfg)
        label = TRANSCRIPTION_REGISTRY[backend]["label"]
        console.print(f"Transcription set to [bold]{label}[/]")


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
