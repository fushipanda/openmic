"""Unit tests for template management."""

from pathlib import Path

import pytest

from openmic.templates import NoteTemplate, TemplateManager


@pytest.fixture
def templates_dir(tmp_path):
    """Create a temporary templates directory with test templates."""
    d = tmp_path / "templates"
    d.mkdir()
    return d


@pytest.fixture
def user_dir(tmp_path):
    """Create a temporary user templates directory."""
    d = tmp_path / "user_templates"
    d.mkdir()
    return d


def _write_template(directory, filename, name, description, prompt):
    """Helper to write a template file with YAML frontmatter."""
    path = directory / filename
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{prompt}\n"
    )
    return path


class TestTemplateParsing:
    def test_parse_valid_template(self, templates_dir, user_dir):
        """A valid template with frontmatter and {transcript} loads correctly."""
        _write_template(
            templates_dir, "test.md",
            "Test Template", "A test template",
            "Generate notes:\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        # Override built-in dir
        manager._templates.clear()
        manager.builtin_dir = templates_dir
        manager._load_templates()

        tmpl = manager.get_template("test")
        assert tmpl is not None
        assert tmpl.id == "test"
        assert tmpl.name == "Test Template"
        assert tmpl.description == "A test template"
        assert "{transcript}" in tmpl.prompt
        assert tmpl.is_builtin is True

    def test_missing_frontmatter_skipped(self, templates_dir, user_dir):
        """Templates without YAML frontmatter are skipped."""
        (templates_dir / "bad.md").write_text("No frontmatter here\n{transcript}")
        manager = TemplateManager(user_templates_dir=user_dir)
        manager._templates.clear()
        manager.builtin_dir = templates_dir
        manager._load_templates()

        # Should have fallback only
        assert not manager.has_template("bad")

    def test_invalid_yaml_skipped(self, templates_dir, user_dir):
        """Templates with invalid YAML frontmatter are skipped."""
        (templates_dir / "bad.md").write_text(
            "---\n: invalid yaml: [broken\n---\n\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        manager._templates.clear()
        manager.builtin_dir = templates_dir
        manager._load_templates()

        assert not manager.has_template("bad")

    def test_missing_name_skipped(self, templates_dir, user_dir):
        """Templates missing the 'name' field are skipped."""
        (templates_dir / "noname.md").write_text(
            "---\ndescription: Has description but no name\n---\n\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        manager._templates.clear()
        manager.builtin_dir = templates_dir
        manager._load_templates()

        assert not manager.has_template("noname")

    def test_missing_description_skipped(self, templates_dir, user_dir):
        """Templates missing the 'description' field are skipped."""
        (templates_dir / "nodesc.md").write_text(
            "---\nname: Has name\n---\n\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        manager._templates.clear()
        manager.builtin_dir = templates_dir
        manager._load_templates()

        assert not manager.has_template("nodesc")

    def test_missing_transcript_placeholder_skipped(self, templates_dir, user_dir):
        """Templates without {transcript} placeholder are skipped."""
        (templates_dir / "noplaceholder.md").write_text(
            "---\nname: No Placeholder\ndescription: Missing placeholder\n---\n\nJust text"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        manager._templates.clear()
        manager.builtin_dir = templates_dir
        manager._load_templates()

        assert not manager.has_template("noplaceholder")


class TestTemplateLoading:
    def test_builtin_templates_load(self):
        """Built-in templates from the package are loaded."""
        manager = TemplateManager(user_templates_dir=Path("/nonexistent"))
        templates = manager.list_templates()
        assert len(templates) >= 6
        ids = {t.id for t in templates}
        assert "default" in ids
        assert "concise" in ids
        assert "team-meeting" in ids
        assert "technical" in ids
        assert "executive" in ids
        assert "one-on-one" in ids

    def test_all_builtins_are_builtin(self):
        """All loaded built-in templates have is_builtin=True."""
        manager = TemplateManager(user_templates_dir=Path("/nonexistent"))
        for tmpl in manager.list_templates():
            assert tmpl.is_builtin is True

    def test_user_templates_override_builtins(self, user_dir):
        """User templates with same ID override built-in templates."""
        _write_template(
            user_dir, "default.md",
            "My Custom Default", "Custom default template",
            "My custom prompt\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)

        tmpl = manager.get_template("default")
        assert tmpl is not None
        assert tmpl.name == "My Custom Default"
        assert tmpl.is_builtin is False

    def test_user_templates_add_new(self, user_dir):
        """User templates with new IDs are added alongside built-ins."""
        _write_template(
            user_dir, "custom.md",
            "My Custom Template", "A custom template",
            "Custom prompt\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)

        assert manager.has_template("custom")
        assert manager.has_template("default")  # Built-in still present

    def test_get_builtin_templates(self):
        """get_builtin_templates returns only built-in templates."""
        manager = TemplateManager(user_templates_dir=Path("/nonexistent"))
        builtins = manager.get_builtin_templates()
        assert all(t.is_builtin for t in builtins)
        assert len(builtins) >= 6

    def test_get_user_templates(self, user_dir):
        """get_user_templates returns only user-defined templates."""
        _write_template(
            user_dir, "custom.md",
            "Custom", "Custom desc",
            "Prompt\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        user_templates = manager.get_user_templates()
        assert len(user_templates) == 1
        assert user_templates[0].id == "custom"
        assert user_templates[0].is_builtin is False


class TestTemplateAccess:
    def test_get_nonexistent_template(self):
        """get_template returns None for unknown IDs."""
        manager = TemplateManager(user_templates_dir=Path("/nonexistent"))
        assert manager.get_template("nonexistent") is None

    def test_has_template(self):
        """has_template correctly checks template existence."""
        manager = TemplateManager(user_templates_dir=Path("/nonexistent"))
        assert manager.has_template("default") is True
        assert manager.has_template("nonexistent") is False

    def test_default_template(self):
        """default_template always returns a valid template."""
        manager = TemplateManager(user_templates_dir=Path("/nonexistent"))
        default = manager.default_template
        assert default is not None
        assert default.id == "default"

    def test_list_templates_builtins_first(self, user_dir):
        """list_templates returns built-ins before user templates."""
        _write_template(
            user_dir, "zzz-custom.md",
            "ZZZ Custom", "Custom at end",
            "Prompt\n{transcript}"
        )
        manager = TemplateManager(user_templates_dir=user_dir)
        templates = manager.list_templates()

        # Find where user templates start
        first_user_idx = next(
            (i for i, t in enumerate(templates) if not t.is_builtin), len(templates)
        )
        # All items before first user template should be built-in
        for t in templates[:first_user_idx]:
            assert t.is_builtin is True


class TestFallbackTemplate:
    def test_fallback_when_no_templates_found(self, tmp_path):
        """A fallback template is created when no template files exist."""
        empty_builtin = tmp_path / "empty_builtin"
        empty_builtin.mkdir()
        empty_user = tmp_path / "empty_user"
        empty_user.mkdir()

        manager = TemplateManager(user_templates_dir=empty_user)
        manager._templates.clear()
        manager.builtin_dir = empty_builtin
        manager._load_templates()

        assert manager.has_template("default")
        default = manager.default_template
        assert default.name == "Standard Meeting Notes"
        assert "{transcript}" in default.prompt


class TestBuiltinTemplateContent:
    """Verify each built-in template has correct structure."""

    @pytest.fixture
    def manager(self):
        return TemplateManager(user_templates_dir=Path("/nonexistent"))

    def test_default_template_content(self, manager):
        tmpl = manager.get_template("default")
        assert tmpl.name == "Standard Meeting Notes"
        assert "{transcript}" in tmpl.prompt

    def test_concise_template_content(self, manager):
        tmpl = manager.get_template("concise")
        assert tmpl.name == "Concise Notes"
        assert "{transcript}" in tmpl.prompt

    def test_team_meeting_template_content(self, manager):
        tmpl = manager.get_template("team-meeting")
        assert tmpl.name == "Team Meeting"
        assert "{transcript}" in tmpl.prompt

    def test_technical_template_content(self, manager):
        tmpl = manager.get_template("technical")
        assert tmpl.name == "Technical Discussion"
        assert "{transcript}" in tmpl.prompt

    def test_executive_template_content(self, manager):
        tmpl = manager.get_template("executive")
        assert tmpl.name == "Executive Summary"
        assert "{transcript}" in tmpl.prompt

    def test_one_on_one_template_content(self, manager):
        tmpl = manager.get_template("one-on-one")
        assert tmpl.name == "1:1 Meeting"
        assert "{transcript}" in tmpl.prompt
