"""Session storage — append-only JSONL files, one per meeting."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from openmic.storage import _PROJECT_ROOT, _sanitize_name

SESSIONS_DIR = _PROJECT_ROOT / "sessions"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append(session_path: Path, entry: dict) -> None:
    """Append one JSON entry as a line to the session file."""
    with session_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def create_session(name: str | None = None) -> Path:
    """Create a new session JSONL file and write the meta entry.

    Args:
        name: Human-readable session name. If None, filename uses a timestamp.

    Returns:
        Path to the new session file.
    """
    SESSIONS_DIR.mkdir(exist_ok=True)

    if name:
        safe = _sanitize_name(name)
        # If a file with this name already exists, append a counter to avoid collision
        candidate = SESSIONS_DIR / f"{safe}.jsonl"
        counter = 1
        while candidate.exists():
            candidate = SESSIONS_DIR / f"{safe}_{counter}.jsonl"
            counter += 1
        session_path = candidate
        slug = safe
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        session_path = SESSIONS_DIR / f"{timestamp}.jsonl"
        slug = timestamp

    meta = {
        "type": "meta",
        "id": str(uuid.uuid4()),
        "slug": slug,
        "name": name or session_path.stem,
        "created": _now(),
    }
    _append(session_path, meta)
    return session_path


def append_transcript(
    session_path: Path,
    segments: list[dict],
    duration_s: float,
) -> None:
    """Append a transcript recording entry to the session.

    Args:
        session_path: Path to the session JSONL file.
        segments: List of {"speaker", "text", "start", "end"} dicts.
        duration_s: Duration of the recording in seconds.
    """
    entry = {
        "type": "transcript",
        "id": str(uuid.uuid4()),
        "timestamp": _now(),
        "duration_s": round(duration_s, 2),
        "segments": segments,
    }
    _append(session_path, entry)


def append_notes(session_path: Path, content: str, template: str) -> None:
    """Append a notes entry to the session.

    Args:
        session_path: Path to the session JSONL file.
        content: Full notes text (markdown).
        template: Template ID used to generate the notes.
    """
    entry = {
        "type": "notes",
        "id": str(uuid.uuid4()),
        "timestamp": _now(),
        "template": template,
        "content": content,
    }
    _append(session_path, entry)


def append_title_update(session_path: Path, auto_title: str, model: str) -> None:
    """Append an AI-generated title entry to the session.

    Args:
        session_path: Path to the session JSONL file.
        auto_title: Short AI-generated title string.
        model: Model identifier used to generate the title.
    """
    entry = {
        "type": "title_update",
        "id": str(uuid.uuid4()),
        "timestamp": _now(),
        "autoTitle": auto_title,
        "model": model,
    }
    _append(session_path, entry)


def append_rename(session_path: Path, custom_title: str) -> None:
    """Append a user-defined rename entry to the session.

    Args:
        session_path: Path to the session JSONL file.
        custom_title: User-provided display title.
    """
    entry = {
        "type": "rename",
        "id": str(uuid.uuid4()),
        "timestamp": _now(),
        "customTitle": custom_title,
    }
    _append(session_path, entry)


def display_title(session_data: dict) -> str:
    """Return the best available display title for a session.

    Precedence: customTitle > autoTitle > slug > id > 'unknown'
    """
    return (
        session_data.get("customTitle")
        or session_data.get("autoTitle")
        or session_data["meta"].get("slug")
        or session_data["meta"].get("name")
        or session_data["meta"].get("id", "unknown")
    )


def get_session_meta(session_path: Path) -> dict:
    """Read the meta entry (first line) from a session file.

    Returns:
        The meta dict, or a minimal fallback if the file is unreadable.
    """
    try:
        with session_path.open(encoding="utf-8") as f:
            first = f.readline()
        return json.loads(first)
    except Exception:
        return {"type": "meta", "name": session_path.stem, "created": ""}


def read_session(session_path: Path) -> dict:
    """Read all entries from a session file.

    Returns:
        Dict with keys:
          "meta", "transcripts" (list), "notes" (list),
          "autoTitle" (str | None), "customTitle" (str | None),
          "updatedAt" (float | None), "lastTranscriptAt" (str | None).
    """
    result: dict = {
        "meta": {},
        "transcripts": [],
        "notes": [],
        "autoTitle": None,
        "customTitle": None,
        "updatedAt": None,
        "lastTranscriptAt": None,
    }
    try:
        with session_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = entry.get("type")
                if t == "meta":
                    result["meta"] = entry
                elif t == "transcript":
                    result["transcripts"].append(entry)
                elif t == "notes":
                    result["notes"].append(entry)
                elif t == "title_update":
                    # Last title_update wins
                    result["autoTitle"] = entry.get("autoTitle")
                elif t == "rename":
                    # Last rename wins
                    result["customTitle"] = entry.get("customTitle")
        result["updatedAt"] = session_path.stat().st_mtime
        if result["transcripts"]:
            result["lastTranscriptAt"] = result["transcripts"][-1].get("timestamp")
    except FileNotFoundError:
        pass
    return result


def session_to_text(session_path: Path) -> str:
    """Return the full transcript text of a session as a single string.

    Concatenates speaker segments from all transcript entries in order.
    Used by the RAG loader to build embeddings.
    """
    parts: list[str] = []
    data = read_session(session_path)
    for recording in data["transcripts"]:
        for seg in recording.get("segments", []):
            speaker = seg.get("speaker", "Speaker")
            text = seg.get("text", "").strip()
            if text:
                parts.append(f"{speaker}: {text}")
    return "\n".join(parts)


def session_duration_s(session_path: Path) -> float:
    """Return total recorded duration in seconds across all transcript entries."""
    data = read_session(session_path)
    return sum(t.get("duration_s", 0.0) for t in data.get("transcripts", []))


def list_sessions() -> list[Path]:
    """List all session JSONL files, sorted newest first (by mtime)."""
    if not SESSIONS_DIR.exists():
        return []
    paths = list(SESSIONS_DIR.glob("*.jsonl"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
