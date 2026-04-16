# openmic

A privacy-first CLI for capturing and structuring spoken thought. Start a session, speak freely — OpenMic turns what you say into searchable, AI-ready data, entirely on your machine. Audio transcription is always local; the LLM layer (querying, notes) is configurable from fully offline Ollama to any cloud provider.

## Architecture

```
During session:
  Mic ──► faster-whisper (local, GPU) ──► live text in terminal
           └── webrtcvad gates segments on speech boundaries
  Mic ──► local .wav file ──► audio buffer on disk

On stop:
  .wav ──► final segment flush ──► transcript saved to transcripts/
                                ──► .wav deleted (KEEP_RECORDINGS=true to retain)

Post-session:
  /query  ──► LangChain RAG over transcripts/  ──► answer in REPL
  /notes  ──► LangChain summarisation chain    ──► structured notes saved to notes/
```

## Privacy Model

Audio transcription always runs locally — recordings are never uploaded regardless of LLM choice.

| Mode | Transcription | LLM | Audio uploaded |
|------|--------------|-----|----------------|
| Local + cloud LLM | faster-whisper (local, GPU) | Anthropic/OpenAI/Gemini/OpenRouter | Never |
| Fully local | faster-whisper (local, GPU) | Ollama | Never |

## Tech Stack

- Python 3.12+
- Rich — terminal output and formatting
- prompt_toolkit — REPL input, completions, key bindings
- faster-whisper — GPU-accelerated local transcription via CTranslate2
- webrtcvad-wheels — voice activity detection for speech boundary gating
- LangChain — RAG querying and notes generation (model-agnostic, provider swappable)
- Ollama (via langchain-ollama) — local LLM and embeddings
- sounddevice — mic capture
- FAISS — local vector store for RAG

## CLI Architecture Notes

- `app.py` is a lightweight CLI (Rich + prompt_toolkit). The Textual TUI has been removed.
- The REPL uses a custom prompt_toolkit `Application` with a fixed `Layout` (HSplit:
  separator → input → completions). This replaced `PromptSession` to allow the completions
  window to be a managed peer of the input, not a floating popup.
- **Completion navigation**: highlight index (`_comp_idx`) and viewport offset (`_view_offset`)
  are decoupled from the buffer. Arrow keys move the highlight only; Enter submits the
  highlighted command; Tab fills it into the buffer. `on_text_changed` resets the highlight
  when the user types.
- **Recording mode owns the terminal**: `recording_mode()` runs a bare `asyncio.sleep` loop —
  the prompt_toolkit Application is NOT active during recording. REPL keybindings (including
  Shift+Tab) cannot fire while recording. Any future "stop from shortcut" feature would require
  recording_mode to use the pt Application for its input loop instead.

## .env Keys

```
# Transcription (always local — no API key required)
WHISPER_MODEL=large-v3-turbo    # tiny.en, base.en, small.en, medium.en, large-v3, large-v3-turbo
WHISPER_DEVICE=auto             # auto, cuda, cpu — auto selects GPU when available
WHISPER_COMPUTE_TYPE=float16    # float16 (fastest on GPU), int8_float16, int8
WHISPER_VAD_ENABLED=true        # voice activity detection (default: true)
WHISPER_VAD_SILENCE_MS=600      # silence before flushing a segment (ms)
KEEP_RECORDINGS=true            # retain WAV files after transcription (default: deleted)

# LLM provider (for /query and /notes)
LLM_PROVIDER=anthropic          # anthropic, openai, gemini, openrouter, ollama
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...              # also used for embeddings with cloud providers
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...

# Fully local (no cloud calls at all)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Local Install

Install globally via uv from the repo root:

```bash
uv tool install ".[local]"
```

Re-run this after adding or removing dependencies in `pyproject.toml`. The `[local]` extra installs `faster-whisper` and `webrtcvad-wheels`. Extras use inline bracket syntax with uv — `--extra` flag is not supported.

## Versioning and Releases

Current version is tracked in `pyproject.toml` and surfaced via `openmic/version.py`.

Release process:
1. Bump `version` in `pyproject.toml`
2. Commit: `git commit -m "Bump to vX.Y.Z; <summary>"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin <branch> && git push origin vX.Y.Z`
5. Create release: `gh release create vX.Y.Z --title "vX.Y.Z — <title>" --notes "<notes>"`

Versioning follows semver loosely:
- Patch (0.x.Y) — bug fixes, small improvements
- Minor (0.X.0) — new features or significant backend changes (e.g. faster-whisper switch)
- Major (X.0.0) — reserved for breaking changes or major architectural shifts

## Strict Instructions

- Commit to git after each TODO item is completed. Message should reference the item.
- Never commit `.env`, `*.wav`, or anything in `transcripts/` or `notes/`. These must be in `.gitignore` from the first commit.
- Do not install a dependency if something already in `pyproject.toml` covers the need.
- All LLM calls must route through LangChain. Do not call any provider SDK directly.
- Use keys from `.env` via environment variables. Never hardcode API keys.
- If a task is blocked, skip it and move to the next one. Do not retry indefinitely.
- Run any existing tests before marking a task complete.
- WAV files are deleted after transcription by default. Do not change this behaviour unless explicitly asked.

---

## Active TODO

No active items. See git log for history.
