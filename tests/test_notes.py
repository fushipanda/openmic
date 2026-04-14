"""Integration test for notes generation with mocked LLM."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from langchain_core.runnables import RunnableLambda

from openmic.notes import generate_meeting_notes, generate_notes_for_latest, get_existing_notes, NOTES_PROMPT


def _fake_llm(content: str):
    """Return a LangChain-compatible Runnable that produces a fake LLM response."""
    class _Resp:
        pass
    resp = _Resp()
    resp.content = content
    return RunnableLambda(lambda _: resp)


@pytest.fixture
def storage_dirs(tmp_path, monkeypatch):
    """Set up isolated transcript and notes directories."""
    transcripts = tmp_path / "transcripts"
    notes = tmp_path / "notes"
    transcripts.mkdir()
    notes.mkdir()
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    monkeypatch.setattr("openmic.storage.TRANSCRIPTS_DIR", transcripts)
    monkeypatch.setattr("openmic.storage.NOTES_DIR", notes)
    monkeypatch.setattr("openmic.storage.RECORDINGS_DIR", recordings)
    monkeypatch.setattr("openmic.notes.NOTES_DIR", notes)
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

        with patch("openmic.notes.get_llm", return_value=_fake_llm(FAKE_NOTES)):
            content, notes_path, used_cache = generate_meeting_notes(transcript_path)

        assert notes_path.exists()
        assert notes_path.parent == notes_dir
        assert "2025-06-15_14-30_notes.md" == notes_path.name
        assert "# Meeting Notes" in content
        assert "Jun 15th 2025, 2:30 PM" in content
        assert "database migration" in content.lower() or "PostgreSQL" in content
        assert used_cache is False

    def test_notes_content_has_header(self, storage_dirs):
        """Generated notes include a header and source reference."""
        transcripts_dir, _ = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\n**Speaker:** Hello.\n\n")

        with patch("openmic.notes.get_llm", return_value=_fake_llm("## Agenda\n- Nothing")):
            content, _, _ = generate_meeting_notes(transcript_path)

        assert "# Meeting Notes" in content
        assert "Jun 15th 2025, 2:30 PM" in content

    def test_llm_chain_receives_transcript(self, storage_dirs):
        """The LLM chain is called with the transcript content."""
        transcripts_dir, _ = storage_dirs

        transcript_text = "# Transcript\n\n**Speaker 1:** Important discussion.\n\n"
        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text(transcript_text)

        received = {}

        def _capturing_llm(prompt_value):
            # prompt_value is a StringPromptValue — the fully-rendered prompt
            received["text"] = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
            class _R:
                content = "Notes"
            return _R()

        with patch("openmic.notes.get_llm", return_value=RunnableLambda(_capturing_llm)):
            generate_meeting_notes(transcript_path)

        # The transcript content should appear inside the rendered prompt
        assert transcript_text.strip() in received.get("text", "")

    def test_notes_include_yaml_frontmatter(self, storage_dirs):
        """Generated notes include YAML frontmatter with template ID."""
        transcripts_dir, _ = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\nContent.\n\n")

        with patch("openmic.notes.get_llm", return_value=_fake_llm("## Notes")):
            content, _, _ = generate_meeting_notes(transcript_path)

        assert content.startswith("---\n")
        # Parse frontmatter
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["template"] == "default"
        assert "generated" in frontmatter

    def test_notes_with_specific_template(self, storage_dirs):
        """Notes can be generated with a specific template ID."""
        transcripts_dir, _ = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\nContent.\n\n")

        with patch("openmic.notes.get_llm", return_value=_fake_llm("## Concise Notes")):
            content, _, used_cache = generate_meeting_notes(transcript_path, template_id="concise")

        assert used_cache is False
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["template"] == "concise"


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

        with patch("openmic.notes.get_latest_transcript", return_value=latest), \
             patch("openmic.notes.get_llm", return_value=_fake_llm("Latest notes")):
            result = generate_notes_for_latest()

        assert result is not None
        content, path, used_cache = result
        assert "# Meeting Notes" in content
        assert path.name == "2025-12-31_23-59_notes.md"
        assert used_cache is False

    def test_generates_for_latest_with_template(self, storage_dirs):
        """Generates notes for latest transcript using specified template."""
        transcripts_dir, _ = storage_dirs

        latest = transcripts_dir / "2025-12-31_23-59.md"
        latest.write_text("# Latest\n\nContent.\n\n")

        with patch("openmic.notes.get_latest_transcript", return_value=latest), \
             patch("openmic.notes.get_llm", return_value=_fake_llm("Technical notes")):
            result = generate_notes_for_latest(template_id="technical")

        assert result is not None
        content, _, _ = result
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["template"] == "technical"


class TestNotesCaching:
    """BUG-3: Notes should not be regenerated if they already exist."""

    def test_returns_cached_notes_same_template(self, storage_dirs):
        """If notes exist with same template, return cached without LLM call."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\n**Speaker:** Hello.\n\n")

        cached_content = "---\ntemplate: default\ngenerated: '2025-06-15T14:30:00'\n---\n\n# Meeting Notes\n\nCached"
        notes_path = notes_dir / "2025-06-15_14-30_notes.md"
        notes_path.write_text(cached_content)

        with patch("openmic.notes.get_llm") as mock_llm:
            content, path, used_cache = generate_meeting_notes(transcript_path, template_id="default")

        mock_llm.assert_not_called()
        assert content == cached_content
        assert path == notes_path
        assert used_cache is True

    def test_returns_cached_when_different_template(self, storage_dirs):
        """If notes exist with different template, still returns cached (caller handles overwrite)."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\n**Speaker:** Hello.\n\n")

        cached_content = "---\ntemplate: default\ngenerated: '2025-06-15T14:30:00'\n---\n\n# Meeting Notes\n\nCached"
        notes_path = notes_dir / "2025-06-15_14-30_notes.md"
        notes_path.write_text(cached_content)

        with patch("openmic.notes.get_llm") as mock_llm:
            content, path, used_cache = generate_meeting_notes(transcript_path, template_id="concise")

        mock_llm.assert_not_called()
        assert used_cache is True

    def test_generates_when_no_cache(self, storage_dirs):
        """If no cached notes exist, generate them via LLM."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\n**Speaker:** Hello.\n\n")

        with patch("openmic.notes.get_llm", return_value=_fake_llm("Generated notes")):
            content, path, used_cache = generate_meeting_notes(transcript_path)

        assert "# Meeting Notes" in content
        assert used_cache is False

    def test_get_existing_notes_found(self, storage_dirs):
        """get_existing_notes returns content when notes file exists."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("transcript")

        notes_path = notes_dir / "2025-06-15_14-30_notes.md"
        notes_path.write_text("cached notes")

        result = get_existing_notes(transcript_path)
        assert result is not None
        content, path, template_id = result
        assert content == "cached notes"
        assert path == notes_path
        assert template_id is None  # Old format without frontmatter

    def test_get_existing_notes_with_frontmatter(self, storage_dirs):
        """get_existing_notes extracts template_id from YAML frontmatter."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("transcript")

        notes_content = "---\ntemplate: concise\ngenerated: '2025-06-15T14:30:00'\n---\n\n# Notes"
        notes_path = notes_dir / "2025-06-15_14-30_notes.md"
        notes_path.write_text(notes_content)

        result = get_existing_notes(transcript_path)
        assert result is not None
        content, path, template_id = result
        assert template_id == "concise"

    def test_get_existing_notes_not_found(self, storage_dirs):
        """get_existing_notes returns None when no notes file exists."""
        transcripts_dir, _ = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("transcript")

        result = get_existing_notes(transcript_path)
        assert result is None


class TestBackwardCompatibility:
    """Notes without YAML frontmatter (old format) continue to work."""

    def test_old_notes_treated_as_no_template(self, storage_dirs):
        """Old notes without frontmatter return template_id=None."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("transcript")

        old_notes = "# Meeting Notes\n\n*Jun 15th 2025, 2:30 PM*\n\nOld content"
        notes_path = notes_dir / "2025-06-15_14-30_notes.md"
        notes_path.write_text(old_notes)

        result = get_existing_notes(transcript_path)
        assert result is not None
        _, _, template_id = result
        assert template_id is None

    def test_old_notes_returned_as_cached(self, storage_dirs):
        """Old notes are still returned as cached (not regenerated)."""
        transcripts_dir, notes_dir = storage_dirs

        transcript_path = transcripts_dir / "2025-06-15_14-30.md"
        transcript_path.write_text("# Transcript\n\nContent.\n\n")

        old_notes = "# Meeting Notes\n\nOld cached notes"
        notes_path = notes_dir / "2025-06-15_14-30_notes.md"
        notes_path.write_text(old_notes)

        with patch("openmic.notes.get_llm") as mock_llm:
            content, path, used_cache = generate_meeting_notes(transcript_path)

        mock_llm.assert_not_called()
        assert content == old_notes
        assert used_cache is True


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
