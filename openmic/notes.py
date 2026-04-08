"""Meeting notes generation using LangChain."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.prompts import PromptTemplate

from openmic.rag import get_llm
from openmic.storage import get_latest_transcript, save_notes, NOTES_DIR
from openmic.templates import TemplateManager


NOTES_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a meeting notes assistant. Based on the following meeting transcript,
generate structured meeting notes with the following sections:

## Agenda
- List the main topics discussed

## Key Points
- Summarize the important points from the meeting

## Decisions Made
- List any decisions that were made during the meeting

## Action Items
- List any action items, tasks, or follow-ups mentioned
- Include who is responsible if mentioned

---

TRANSCRIPT:
{transcript}

---

Generate the meeting notes now:"""
)


def get_existing_notes(transcript_path: Path) -> tuple[str, Path, Optional[str]] | None:
    """Check if notes already exist for a transcript.

    Returns:
        Tuple of (notes content, notes path, template_id) if found, None otherwise.
        template_id will be None for old notes without frontmatter.
    """
    notes_path = NOTES_DIR / (transcript_path.stem + "_notes.md")
    if not notes_path.exists():
        return None

    content = notes_path.read_text()
    template_id = None

    # Try to parse YAML frontmatter
    if content.startswith("---\n"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                if isinstance(frontmatter, dict):
                    template_id = frontmatter.get("template")
            except yaml.YAMLError:
                pass  # Old format without valid frontmatter

    return content, notes_path, template_id


def generate_meeting_notes(
    transcript_path: Path, template_id: str = "default"
) -> tuple[str, Path, bool]:
    """Generate structured meeting notes from a transcript.

    Returns cached notes if they already exist for this transcript with the same template.

    Args:
        transcript_path: Path to the transcript markdown file
        template_id: ID of the template to use (default: "default")

    Returns:
        Tuple of (generated notes content, path to saved notes file, used_cache)
        used_cache is True if existing notes were returned, False if new notes generated
    """
    existing = get_existing_notes(transcript_path)
    if existing is not None:
        existing_content, existing_path, existing_template = existing
        # If same template, return cached notes
        if existing_template == template_id:
            return existing_content, existing_path, True
        # Different template - caller should handle overwrite prompt
        # For now, return existing (caller will decide whether to overwrite)
        return existing_content, existing_path, True

    # Load template
    template_manager = TemplateManager()
    template = template_manager.get_template(template_id)
    if template is None:
        # Fall back to default template
        template = template_manager.default_template

    transcript_content = transcript_path.read_text()

    # Create LangChain prompt from template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template.prompt,
    )

    llm = get_llm()
    chain = prompt | llm

    notes_content = chain.invoke({"transcript": transcript_content}).content

    # Add header with formatted title
    stem = transcript_path.stem
    timestamp = stem[:16]
    session_name = stem[17:] if len(stem) > 16 else None

    # Format the title nicely
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

    # Build header based on whether we have a session name
    if session_name:
        session_display = session_name.replace("_", " ").strip()
        header = f"# {session_display}\n\n*{date_str}*"
    else:
        header = f"# Meeting Notes\n\n*{date_str}*"

    # Add YAML frontmatter with template metadata
    frontmatter = {
        "template": template.id,
        "generated": datetime.now().isoformat(),
    }
    frontmatter_str = yaml.dump(frontmatter, default_flow_style=False)
    full_notes = f"---\n{frontmatter_str}---\n\n{header}\n\n{notes_content}"

    notes_path = save_notes(full_notes, transcript_path, template_id)
    return full_notes, notes_path, False


def generate_notes_for_latest(template_id: str = "default") -> tuple[str, Path, bool] | None:
    """Generate notes for the most recent transcript.

    Args:
        template_id: ID of the template to use (default: "default")

    Returns:
        Tuple of (notes content, notes path, used_cache) or None if no transcripts exist
    """
    latest = get_latest_transcript()
    if latest is None:
        return None

    return generate_meeting_notes(latest, template_id)
