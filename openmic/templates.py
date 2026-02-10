"""
Template management for meeting notes generation.

Handles loading, parsing, and managing note templates from both built-in
and user-defined template files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class NoteTemplate:
    """Represents a single note template."""

    id: str  # Filename stem (e.g., "default", "concise")
    name: str  # Display name from YAML frontmatter
    description: str  # Short description from YAML frontmatter
    prompt: str  # The actual prompt template with {transcript} placeholder
    is_builtin: bool  # Whether this is a built-in template


class TemplateManager:
    """Manages loading and accessing note templates."""

    def __init__(self, user_templates_dir: Optional[Path] = None):
        """
        Initialize the template manager.

        Args:
            user_templates_dir: Path to user templates directory.
                               Defaults to ~/.config/openmic/templates/
        """
        self.builtin_dir = Path(__file__).parent / "templates"
        self.user_dir = user_templates_dir or (
            Path.home() / ".config" / "openmic" / "templates"
        )
        self._templates: Dict[str, NoteTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load templates from built-in and user directories."""
        # Load built-in templates first
        if self.builtin_dir.exists():
            self._load_from_directory(self.builtin_dir, is_builtin=True)

        # Load user templates (these override built-ins with same ID)
        if self.user_dir.exists():
            self._load_from_directory(self.user_dir, is_builtin=False)

        # Ensure we always have at least a default template
        if not self._templates:
            logger.warning("No templates found, creating fallback default template")
            self._create_fallback_template()

    def _load_from_directory(self, directory: Path, is_builtin: bool) -> None:
        """
        Load all .md template files from a directory.

        Args:
            directory: Path to the directory containing template files
            is_builtin: Whether these are built-in templates
        """
        for template_file in directory.glob("*.md"):
            try:
                template = self._parse_template_file(template_file, is_builtin)
                if template:
                    self._templates[template.id] = template
                    logger.debug(
                        f"Loaded {'built-in' if is_builtin else 'user'} template: {template.id}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load template {template_file.name}: {e}"
                )

    def _parse_template_file(
        self, file_path: Path, is_builtin: bool
    ) -> Optional[NoteTemplate]:
        """
        Parse a template file with YAML frontmatter.

        Expected format:
        ---
        name: Template Name
        description: Short description
        ---

        Template prompt with {transcript} placeholder...

        Args:
            file_path: Path to the template file
            is_builtin: Whether this is a built-in template

        Returns:
            NoteTemplate if successful, None if parsing fails
        """
        content = file_path.read_text(encoding="utf-8")

        # Split on YAML frontmatter delimiters
        parts = content.split("---", 2)
        if len(parts) < 3:
            logger.warning(
                f"Template {file_path.name} missing YAML frontmatter"
            )
            return None

        # Parse YAML frontmatter
        try:
            frontmatter = yaml.safe_load(parts[1])
            if not isinstance(frontmatter, dict):
                logger.warning(
                    f"Template {file_path.name} has invalid frontmatter"
                )
                return None
        except yaml.YAMLError as e:
            logger.warning(
                f"Template {file_path.name} has malformed YAML: {e}"
            )
            return None

        # Extract required fields
        name = frontmatter.get("name")
        description = frontmatter.get("description")
        prompt = parts[2].strip()

        if not name or not description:
            logger.warning(
                f"Template {file_path.name} missing required fields (name, description)"
            )
            return None

        if "{transcript}" not in prompt:
            logger.warning(
                f"Template {file_path.name} missing {{transcript}} placeholder"
            )
            return None

        template_id = file_path.stem
        return NoteTemplate(
            id=template_id,
            name=name,
            description=description,
            prompt=prompt,
            is_builtin=is_builtin,
        )

    def _create_fallback_template(self) -> None:
        """Create a hardcoded fallback template if no templates found."""
        fallback_prompt = """Generate comprehensive meeting notes from the following transcript.

## Agenda
Main topics discussed

## Key Points
Important information and updates

## Decisions Made
Key decisions and their rationale

## Action Items
Tasks assigned with owners

---

TRANSCRIPT:
{transcript}

---

Generate the notes now:"""

        self._templates["default"] = NoteTemplate(
            id="default",
            name="Standard Meeting Notes",
            description="Comprehensive notes with agenda, key points, decisions, and action items",
            prompt=fallback_prompt,
            is_builtin=True,
        )

    def get_template(self, template_id: str) -> Optional[NoteTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: The template ID (filename stem)

        Returns:
            NoteTemplate if found, None otherwise
        """
        return self._templates.get(template_id)

    def list_templates(self) -> List[NoteTemplate]:
        """
        Get all available templates, sorted with built-ins first.

        Returns:
            List of NoteTemplate objects
        """
        # Sort: built-ins first, then by ID
        return sorted(
            self._templates.values(),
            key=lambda t: (not t.is_builtin, t.id),
        )

    def get_builtin_templates(self) -> List[NoteTemplate]:
        """Get only built-in templates."""
        return [t for t in self._templates.values() if t.is_builtin]

    def get_user_templates(self) -> List[NoteTemplate]:
        """Get only user-defined templates."""
        return [t for t in self._templates.values() if not t.is_builtin]

    def has_template(self, template_id: str) -> bool:
        """Check if a template exists."""
        return template_id in self._templates

    @property
    def default_template(self) -> NoteTemplate:
        """Get the default template (guaranteed to exist)."""
        return self._templates.get("default") or list(self._templates.values())[0]
