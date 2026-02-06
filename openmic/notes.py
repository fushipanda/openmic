"""Meeting notes generation using LangChain."""

from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

from openmic.rag import get_llm
from openmic.storage import get_latest_transcript, save_notes


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


def generate_meeting_notes(transcript_path: Path) -> tuple[str, Path]:
    """Generate structured meeting notes from a transcript.

    Args:
        transcript_path: Path to the transcript markdown file

    Returns:
        Tuple of (generated notes content, path to saved notes file)
    """
    transcript_content = transcript_path.read_text()

    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=NOTES_PROMPT)

    notes_content = chain.run(transcript=transcript_content)

    # Add header
    full_notes = f"# Meeting Notes\n\nSource: {transcript_path.name}\n\n{notes_content}"

    notes_path = save_notes(full_notes, transcript_path)
    return full_notes, notes_path


def generate_notes_for_latest() -> tuple[str, Path] | None:
    """Generate notes for the most recent transcript.

    Returns:
        Tuple of (notes content, notes path) or None if no transcripts exist
    """
    latest = get_latest_transcript()
    if latest is None:
        return None

    return generate_meeting_notes(latest)
