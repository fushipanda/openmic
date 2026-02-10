"""Storage utilities for transcripts and notes."""

from datetime import datetime
from pathlib import Path


TRANSCRIPTS_DIR = Path("transcripts")
NOTES_DIR = Path("notes")
RECORDINGS_DIR = Path("recordings")


def format_transcript_title(timestamp: str, session_name: str | None = None) -> str:
    """Format a human-readable transcript title.

    Args:
        timestamp: Timestamp string in YYYY-MM-DD_HH-MM format
        session_name: Optional session name (underscores treated as spaces)

    Returns:
        Formatted title like "Meeting Transcript — Jan 15th 2026, 2:30 PM"
        or "Standup — Jan 15th 2026, 2:30 PM"
    """
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M")
        day = dt.day
        if 11 <= day <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        date_str = dt.strftime(f"%b {day}{suffix} %Y, %-I:%M %p")
    except ValueError:
        date_str = timestamp

    if session_name:
        name = session_name.replace("_", " ").strip()
        return f"{name} — {date_str}"
    return f"Meeting Transcript — {date_str}"


def ensure_dirs() -> None:
    """Create storage directories if they don't exist."""
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    RECORDINGS_DIR.mkdir(exist_ok=True)


def save_transcript(segments: list[dict], session_name: str | None = None) -> Path:
    """Save diarized transcript to markdown file.

    Args:
        segments: List of speaker-labeled segments
        session_name: Optional session name for the filename

    Returns:
        Path to the saved transcript file
    """
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if session_name:
        filename = f"{timestamp}_{session_name}.md"
    else:
        filename = f"{timestamp}.md"

    filepath = TRANSCRIPTS_DIR / filename

    title = format_transcript_title(timestamp, session_name)
    lines = [f"# {title}\n\n"]

    for segment in segments:
        speaker = segment.get("speaker", "Speaker")
        text = segment.get("text", "")
        lines.append(f"**{speaker}:** {text}\n\n")

    filepath.write_text("".join(lines))
    return filepath


def get_latest_transcript() -> Path | None:
    """Get the most recent transcript file.

    Returns:
        Path to the latest transcript, or None if no transcripts exist
    """
    ensure_dirs()
    transcripts = sorted(TRANSCRIPTS_DIR.glob("*.md"), reverse=True)
    return transcripts[0] if transcripts else None


def list_transcripts() -> list[Path]:
    """List all transcript files.

    Returns:
        List of transcript file paths, sorted newest first
    """
    ensure_dirs()
    return sorted(TRANSCRIPTS_DIR.glob("*.md"), reverse=True)


def rename_transcript(old_path: Path, new_name: str) -> Path:
    """Rename a transcript file with a new session name.

    Args:
        old_path: Current path to the transcript
        new_name: New session name to use

    Returns:
        Path to the renamed transcript file
    """
    # Extract timestamp prefix from existing filename
    stem = old_path.stem
    # Timestamp format is YYYY-MM-DD_HH-MM, which is 16 chars
    timestamp = stem[:16]
    new_name = new_name.strip().replace(" ", "_")
    new_filename = f"{timestamp}_{new_name}.md"
    new_path = old_path.parent / new_filename

    # Update the heading inside the file too
    content = old_path.read_text()
    first_line_end = content.index("\n")
    title = format_transcript_title(timestamp, new_name)
    content = f"# {title}" + content[first_line_end:]
    new_path.write_text(content)

    if new_path != old_path:
        old_path.unlink()

    return new_path


def save_notes(content: str, transcript_path: Path, template_id: str = "default") -> Path:
    """Save generated notes alongside source transcript.

    Args:
        content: The generated notes content (should include YAML frontmatter with template metadata)
        transcript_path: Path to the source transcript
        template_id: ID of the template used (included for API consistency, stored in frontmatter)

    Returns:
        Path to the saved notes file
    """
    ensure_dirs()

    # Keep simple filename - template is stored in frontmatter
    notes_filename = transcript_path.stem + "_notes.md"
    notes_path = NOTES_DIR / notes_filename
    notes_path.write_text(content)
    return notes_path


def list_recordings() -> list[Path]:
    """List all WAV recording files.

    Returns:
        List of recording file paths, sorted newest first
    """
    ensure_dirs()
    return sorted(RECORDINGS_DIR.glob("*.wav"), reverse=True)


def delete_all_recordings() -> tuple[int, int]:
    """Delete all saved recordings.

    Returns:
        Tuple of (count of files deleted, total bytes freed)
    """
    recordings = list_recordings()
    total_bytes = 0
    count = 0
    for rec in recordings:
        total_bytes += rec.stat().st_size
        rec.unlink()
        count += 1
    return count, total_bytes
