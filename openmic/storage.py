"""Storage utilities for transcripts and notes."""

from datetime import datetime
from pathlib import Path


TRANSCRIPTS_DIR = Path("transcripts")
NOTES_DIR = Path("notes")


def ensure_dirs() -> None:
    """Create storage directories if they don't exist."""
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)


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

    lines = [f"# Meeting Transcript - {timestamp}\n\n"]

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
    content = f"# Meeting Transcript - {timestamp} ({new_name})" + content[first_line_end:]
    new_path.write_text(content)

    if new_path != old_path:
        old_path.unlink()

    return new_path


def save_notes(content: str, transcript_path: Path) -> Path:
    """Save generated notes alongside source transcript.

    Args:
        content: The generated notes content
        transcript_path: Path to the source transcript

    Returns:
        Path to the saved notes file
    """
    ensure_dirs()

    notes_filename = transcript_path.stem + "_notes.md"
    notes_path = NOTES_DIR / notes_filename
    notes_path.write_text(content)
    return notes_path
