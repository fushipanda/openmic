"""Integration test for notes generation with mocked LLM."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openmic.notes import generate_meeting_notes, generate_notes_for_latest, NOTES_PROMPT


@pytest.fixture
def storage_dirs(tmp_path, monkeypatch):
    """Set up isolated transcript and notes directories."""
    transcripts = tmp_path / "transcripts"
    notes = tmp_path / "notes"
    transcripts.mkdir()
    notes.mkdir()
    monkeypatch.setattr("openmic.storage.TRANSCRIPTS_DIR", transcripts)
    monkeypatch.setattr("openmic.storage.NOTES_DIR", notes)
    monkeypatch.setattr("openmic.notes.get_latest_transcript",
                        lambda: _get_latest(transcripts))
    return transcripts, notes


def _get_latest(transcripts_dir):
    """Helper to find the latest transcript in a test dir."""
    files = sorted(transcripts_dir.glob("*.md"), reverse=True)
    return files[0] if files else None


FAKE_NOTES = """\
## Agenda
- Discuss database migration

## Key Points
- Migration to PostgreSQL planned for Q3

## Decisions Made
- Use PostgreSQL for the new database

## Action Items
- [ ] Complete schema migration by September 30th (Speaker 2)"""


class TestGenerateMeetingNotes:
    def test_generates_and_saves_notes(self, storage_dirs):
        """Notes are generated from transcript and saved to notes dir."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text(
            "# Meeting Transcript\n\n"
            "**Speaker 1:** We need to migrate to PostgreSQL.\n\n"
            "**Speaker 2:** I'll handle it by September 30th.\n\n"
        )

        mock_chain = MagicMock()
        mock_chain.run.return_value = FAKE_NOTES

        with patch("openmic.notes.get_llm") as mock_get_llm, \
             patch("openmic.notes.LLMChain", return_value=mock_chain):
            content, notes_path = generate_meeting_notes(transcript_path)

        assert notes_path.exists()
        assert notes_path.parent == notes_dir
        assert "2025-06-15_14-30_notes.md" == notes_path.name
        assert "# Meeting Notes" in content
        assert "Source: 2025-06-15_14-30.md" in content
        assert "database migration" in content.lower() or "PostgreSQL" in content

    def test_notes_content_has_header(self, storage_dirs):
        """Generated notes include a header and source reference."""
        transcripts_dir, _ = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\n**Speaker:** Hello.\n\n")

        mock_chain = MagicMock()
        mock_chain.run.return_value = "## Agenda\n- Nothing"

        with patch("openmic.notes.get_llm"), \
             patch("openmic.notes.LLMChain", return_value=mock_chain):
            content, _ = generate_meeting_notes(transcript_path)

        assert content.startswith("# Meeting Notes")
        assert "Source: 2025-06-15_14-30.md" in content

    def test_llm_chain_receives_transcript(self, storage_dirs):
        """The LLM chain is called with the transcript content."""
        transcripts_dir, _ = storage_dirs

        transcript_text = "# Transcript\n\n**Speaker 1:** Important discussion.\n\n"
        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text(transcript_text)

        mock_chain = MagicMock()
        mock_chain.run.return_value = "Notes"

        with patch("openmic.notes.get_llm"), \
             patch("openmic.notes.LLMChain", return_value=mock_chain):
            generate_meeting_notes(transcript_path)

        mock_chain.run.assert_called_once_with(transcript=transcript_text)


class TestGenerateNotesForLatest:
    def test_no_transcripts(self, storage_dirs):
        """Returns None when no transcripts exist."""
        # Override the monkeypatched get_latest_transcript to return None
        with patch("openmic.notes.get_latest_transcript", return_value=None):
            result = generate_notes_for_latest()
        assert result is None

    def test_generates_for_latest(self, storage_dirs):
        """Generates notes for the most recent transcript."""
        transcripts_dir, notes_dir = storage_dirs

        (transcripts_dir / "2025-01-01_10-00.md").write_text("Old transcript")
        latest = transcripts_dir / "2025-12-31_23-59.md"
        latest.write_text("# Latest\n\n**Speaker 1:** Latest content.\n\n")

        mock_chain = MagicMock()
        mock_chain.run.return_value = "Latest notes"

        with patch("openmic.notes.get_latest_transcript", return_value=latest), \
             patch("openmic.notes.get_llm"), \
             patch("openmic.notes.LLMChain", return_value=mock_chain):
            result = generate_notes_for_latest()

        assert result is not None
        content, path = result
        assert "# Meeting Notes" in content
        assert path.name == "2025-12-31_23-59_notes.md"


class TestNotesPrompt:
    def test_prompt_has_required_sections(self):
        """The prompt template asks for all required note sections."""
        template = NOTES_PROMPT.template
        assert "Agenda" in template
        assert "Key Points" in template
        assert "Decisions Made" in template
        assert "Action Items" in template

    def test_prompt_input_variable(self):
        """The prompt template uses 'transcript' as its input variable."""
        assert NOTES_PROMPT.input_variables == ["transcript"]
