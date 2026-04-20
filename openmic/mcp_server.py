"""MCP server exposing OpenMic session data via FastMCP (stdio transport)."""
import logging
from pathlib import Path

import anyio
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan

_CONFIG_DIR = Path.home() / ".config" / "openmic"
load_dotenv()
load_dotenv(_CONFIG_DIR / ".env", override=False)

logger = logging.getLogger(__name__)


def _apply_settings() -> None:
    """Apply llm_provider/llm_model/whisper_model from settings.json to env.

    Mirrors app.py:_bootstrap() so the MCP server respects the user's
    configured provider rather than falling back to the project .env defaults.
    """
    import json
    settings_path = _CONFIG_DIR / "settings.json"
    try:
        config = json.loads(settings_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    for key, env_var in (
        ("llm_provider", "LLM_PROVIDER"),
        ("llm_model", "LLM_MODEL"),
        ("whisper_model", "WHISPER_MODEL"),
    ):
        if config.get(key):
            os.environ[env_var] = config[key]


_apply_settings()


@lifespan
async def startup(server: FastMCP):
    from openmic.rag import TranscriptRAG

    rag = None
    try:
        rag = TranscriptRAG()
        await anyio.to_thread.run_sync(rag.refresh)
    except Exception as exc:
        logger.warning("TranscriptRAG unavailable: %s", exc)
    yield {"rag": rag}


mcp = FastMCP("openmic", lifespan=startup)


def _resolve_session(session_id: str) -> Path:
    from openmic.session import list_sessions

    for path in list_sessions():
        if path.stem == session_id:
            return path
    raise ValueError(f"Session not found: {session_id!r}")


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    if len(parts) == 3:
        return parts[2].lstrip("\n")
    return content


@mcp.tool
def query_transcripts(question: str, ctx: Context) -> str:
    """Search across all OpenMic session transcripts using RAG.

    Returns an AI-generated answer and the session sources it drew from.
    Requires the FAISS index to have been built — run /query inside openmic first.
    """
    rag = ctx.lifespan_context.get("rag")
    if rag is None:
        return (
            "RAG is unavailable. Possible causes: no sessions recorded yet, "
            "the FAISS index has not been built (run /query inside openmic first), "
            "or the LLM provider is not configured."
        )
    try:
        rag.clear_chat_history()
        result = rag.query(question)
        answer = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])
        if sources:
            source_list = "\n".join(f"  - {s}" for s in sources)
            return f"{answer}\n\nSources:\n{source_list}"
        return answer
    except Exception as exc:
        logger.exception("query_transcripts failed")
        return f"Query failed: {exc}"


@mcp.tool
def list_sessions(ctx: Context) -> list[dict]:
    """List all OpenMic sessions, newest first.

    Each entry contains:
      - id: pass this to other tools as session_id
      - title: best available display title
      - date: ISO creation timestamp
      - duration_s: total recorded duration in seconds
      - has_notes: whether notes have been generated for this session
    """
    from openmic.session import (
        display_title,
        list_sessions as _list_sessions,
        read_session,
        session_duration_s,
    )

    results = []
    for path in _list_sessions():
        try:
            data = read_session(path)
            results.append({
                "id": path.stem,
                "title": display_title(data),
                "date": data["meta"].get("created", ""),
                "duration_s": session_duration_s(path),
                "has_notes": len(data.get("notes", [])) > 0,
            })
        except Exception as exc:
            logger.warning("Could not read session %s: %s", path.name, exc)
    return results


@mcp.tool
def get_session_transcript(session_id: str, ctx: Context) -> str:
    """Return the full transcript for a session as 'Speaker: text' lines.

    Use list_sessions to get valid session IDs.
    """
    from openmic.session import session_to_text

    try:
        path = _resolve_session(session_id)
    except ValueError as exc:
        return str(exc)
    text = session_to_text(path)
    if not text.strip():
        return "This session has no transcript content."
    return text


@mcp.tool
def get_session_notes(session_id: str, ctx: Context) -> str | None:
    """Return existing notes for a session, or None if none have been generated.

    Use list_sessions to find sessions where has_notes is true.
    Use list_note_templates to see available template formats.
    """
    from openmic.session import read_session

    try:
        path = _resolve_session(session_id)
    except ValueError as exc:
        return str(exc)
    data = read_session(path)
    notes_list = data.get("notes", [])
    if not notes_list:
        return None
    return _strip_frontmatter(notes_list[-1].get("content", ""))


@mcp.tool
def list_note_templates(ctx: Context) -> list[dict]:
    """List all available note templates (built-in and user-defined).

    Pass a template 'id' to openmic's /notes command to generate notes in that format.
    """
    from openmic.templates import TemplateManager

    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "is_builtin": t.is_builtin,
        }
        for t in TemplateManager().list_templates()
    ]


def main() -> None:
    import sys

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(levelname)s [openmic-mcp] %(message)s",
    )
    mcp.run()
