# OpenMic

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Built with Claude Code](https://img.shields.io/badge/built_with-Claude_Code-blueviolet.svg)](https://github.com/anthropics/claude-code)
[![Tests: 278/278](https://img.shields.io/badge/tests-278%2F278-brightgreen.svg)](#development)

> A privacy-first CLI for meeting transcription. Runs entirely on your machine — no cloud API keys required for transcription.

[Installation](#installation) • [Commands](#commands) • [Features](#features) • [Development](#development)

---

## What It Does

OpenMic is a terminal-based meeting transcription tool that:

- **Records audio** from your microphone during meetings
- **Streams live transcription** locally via faster-whisper with voice activity detection
- **Saves transcripts** as markdown files with timestamps
- **Answers questions** about past meetings using RAG (retrieval-augmented generation)
- **Generates meeting notes** with structured summaries, action items, and decisions
- **Audio never leaves your machine** — transcription is fully local, no cloud calls

---

## Installation

### Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/fushipanda/openmic/main/install.sh | bash
```

This checks for Python 3.12+, installs OpenMic via pipx/uv/pip, and you're ready to go. The first time you run `openmic`, a setup wizard walks you through LLM provider selection and API key configuration.

### Manual Install

```bash
pip install git+https://github.com/fushipanda/openmic.git
openmic setup  # interactive setup wizard
```

### From Source (Development)

```bash
git clone https://github.com/fushipanda/openmic.git
cd openmic
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,anthropic,openai,local]"
openmic setup
```

The `local` extra installs `faster-whisper` (GPU-accelerated transcription via CTranslate2) and `webrtcvad-wheels` (voice activity detection).

---

## Configuration

### Transcription (no API key required)

Transcription is handled locally by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The model downloads automatically on first use from Hugging Face and is cached at `~/.cache/huggingface/hub/`.

```bash
# In your .env or shell environment
WHISPER_MODEL=large-v3-turbo  # tiny.en, base.en, small.en, medium.en, large-v3, large-v3-turbo
WHISPER_DEVICE=auto           # auto, cuda, cpu — auto selects GPU when available
WHISPER_COMPUTE_TYPE=float16  # float16 (fastest on GPU), int8_float16, int8
WHISPER_VAD_ENABLED=true      # voice activity detection (default: true)
WHISPER_VAD_SILENCE_MS=600    # silence duration before flushing a segment (default: 600ms)
KEEP_RECORDINGS=true          # keep WAV files after transcription (default: deleted for privacy)
```

By default, recorded WAV files are deleted immediately after transcription. Set `KEEP_RECORDINGS=true` to retain them.

### LLM Provider (for /query and /notes)

The setup wizard (`openmic setup`) handles LLM configuration interactively. It will:

1. Let you choose an LLM provider (Anthropic, OpenAI, Gemini, OpenRouter, or Ollama)
2. Install the required LangChain packages
3. Prompt for API keys (skips any already set in your environment)
4. Save everything to `~/.config/openmic/settings.json` and `.env`

You can re-run `openmic setup` at any time to reconfigure.

### API Keys

| Key | Purpose | Required |
|-----|---------|----------|
| `OPENAI_API_KEY` | Embeddings for RAG search | If using cloud LLM provider |
| `ANTHROPIC_API_KEY` | Claude LLM | If using Anthropic |
| `GEMINI_API_KEY` | Gemini LLM | If using Gemini |
| `OPENROUTER_API_KEY` | OpenRouter LLM | If using OpenRouter |

> **Fully offline**: Set `LLM_PROVIDER=ollama` and no API keys are required at all. Transcription is already local — this makes the LLM local too.

> **Why OpenAI key for embeddings?** Anthropic doesn't provide an embeddings API. OpenMic uses OpenAI embeddings for RAG search (`/query`). This isn't needed with `LLM_PROVIDER=ollama`.

---

## Usage

### Start the Application

```bash
openmic              # launch the CLI
openmic --version    # show installed version
openmic update       # self-update to latest release
openmic setup        # re-run the setup wizard
```

### Quick Example Workflow

```bash
# Start recording a meeting
> /start standup

# [speak for a few minutes... live transcript appears as you talk]

# Stop and save
> /stop

# Ask a question about the meeting
> /query What action items were mentioned?

# Generate structured meeting notes
> /notes
```

---

## Commands

| Command | Description |
|---------|-------------|
| `/start [name]` | Start recording (optionally with session name) |
| `/stop [name]` | Stop recording and save transcript |
| `/pause` | Pause recording (resume with `/start`) |
| `/history` | Browse saved transcripts in a date-grouped list |
| `/transcript <n>` | View a specific transcript by number or name |
| `/query <question>` | Ask a question about a transcript (uses RAG) |
| `/notes` | Generate structured notes from a transcript |
| `/name <name>` | Rename the most recent transcript |
| `/model` | Switch LLM provider or model |
| `/help` | Show help with all commands and shortcuts |
| `/verbose` | Toggle debug output |
| `/version` | Show version and check for updates |
| `/exit` | Quit the application |

**Aliases**: `/transcripts`, `/history`, `/transcript` (no args) all open the transcript browser.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Shift+Tab` | Start recording (from REPL) |
| `Ctrl+C` | Stop recording / quit application |
| `Tab` | Autocomplete commands |
| `Up / Down` | Navigate command history / completions |
| `Esc` | Return to home screen (when viewing transcript or notes) |

---

## Features

### Live Transcription — Local, Private

Audio is transcribed locally by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend). No audio is sent to any server.

GPU acceleration is used automatically when CUDA is available — on an RTX-class GPU, inference runs several times faster than CPU. If CUDA libraries aren't present, it falls back to CPU automatically.

Voice activity detection (webrtcvad) gates transcription on speech boundaries, reducing latency from ~10s (fixed interval) to ~1–3s after you stop speaking.

### RAG-Powered Querying

Ask natural language questions about past meetings. OpenMic uses LangChain with FAISS vector search to find relevant transcript segments, then generates accurate answers using your chosen LLM.

### Structured Meeting Notes

Generate professional meeting notes with `/notes`. Notes are saved alongside the transcript and include agenda, discussion points, decisions, and action items.

### Pause/Resume

Pause recording mid-meeting without ending the session. Audio accumulates across pauses — the final transcript covers the full session.

### Transcript Browser

Browse all your saved transcripts organised by date (Today, Yesterday, or formatted date). Indicators show which transcripts still need notes generated.

### Privacy Modes

| Mode | Transcription | LLM | Cloud calls |
|------|--------------|-----|-------------|
| Local + cloud LLM | faster-whisper (local, GPU) | Anthropic/OpenAI/etc | LLM only |
| Fully local | faster-whisper (local, GPU) | Ollama (local) | None |

---

## File Storage

Transcripts and notes are saved as markdown files:

```
transcripts/
  ├── 2026-02-11_14-30.md              # Unnamed session
  ├── 2026-02-11_15-45_standup.md     # Named session
  └── ...

notes/
  ├── 2026-02-11_14-30_notes.md
  ├── 2026-02-11_15-45_standup_notes.md
  └── ...
```

Recorded WAV files are deleted immediately after transcription. Set `KEEP_RECORDINGS=true` to retain them.

---

## Architecture

### Data Flow

```
During session:
  Mic ──► faster-whisper (local, GPU) ──► live text in terminal
           └── webrtcvad gates segments on speech boundaries
  Mic ──► local .wav file ──► audio buffer on disk

On stop:
  .wav ──► final segment flush ──► transcript saved to transcripts/
                                ──► .wav deleted (privacy default)

Post-session:
  /query ──► LangChain RAG over transcripts/ ──► answer in REPL
  /notes ──► LangChain summarization chain   ──► structured notes in notes/
```

### Module Layout

```
openmic/
├── app.py              # CLI entry point (Rich + prompt_toolkit)
├── audio.py            # Mic capture via sounddevice — writes 16kHz mono WAV
├── local_transcribe.py # faster-whisper transcription (realtime VAD + batch, GPU-accelerated)
├── storage.py          # File I/O for transcripts/ and notes/ markdown files
├── rag.py              # LangChain RAG — FAISS vector store + RetrievalQA chain
├── notes.py            # LangChain summarization chain
├── setup.py            # Interactive setup wizard
└── version.py          # Version management and self-update
```

---

## Requirements

- **Python 3.12+**
- **faster-whisper** — GPU-accelerated local transcription via CTranslate2 (installed with `pip install "openmic[local]"`)
- **Whisper model** — auto-downloads on first run from Hugging Face (~75MB for `small.en`, ~1.6GB for `large-v3-turbo`)
- **LLM API key** — only needed for `/query` and `/notes`, not for transcription

---

## Development

### Running Tests

```bash
pip install -e ".[dev,anthropic,openai,local]"
pytest
pytest -v                               # verbose
pytest tests/test_local_transcribe.py  # specific file
```

OpenMic has **278 passing tests** covering:
- Storage operations (transcripts, notes)
- Local transcription (VAD loop, parse logic)
- RAG pipeline (vector search, LLM provider selection)
- Notes generation (template system, caching)
- Setup wizard (provider selection, key mapping)
- Version management (self-update, install detection)

### Test Coverage

- `tests/test_storage.py` — Storage layer
- `tests/test_local_transcribe.py` — Local transcription and VAD
- `tests/test_rag.py` — RAG pipeline integration
- `tests/test_notes.py` — Notes generation
- `tests/test_app.py` — CLI and REPL
- `tests/test_setup.py` — Setup wizard
- `tests/test_version.py` — Version management

### Known Limitations

- **No speaker diarization in local mode** — All speech is attributed to "Speaker". Diarization requires pyannote-audio (torch dependency), which conflicts with the lightweight-CLI goal.
- **Embeddings require OpenAI key** — Anthropic has no embeddings API. Use `LLM_PROVIDER=ollama` to avoid this.
- **FAISS index rebuilt each session** — Not persisted between restarts.

---

## Platform Support

| OS | Status | Notes |
|----|----|-------|
| Linux | Supported | Tested on Arch Linux |
| macOS | Should work | Intel and Apple Silicon (untested) |
| Windows | Untested | webrtcvad may require extra build steps |

---

## How This Was Built

I built OpenMic by looping [Claude Code](https://github.com/anthropics/claude-code) — giving it a feature list and bugs to fix in [CLAUDE.md](./CLAUDE.md), then iterating: Claude would implement a todo, I'd review it, and once tested it got pushed. The whole thing — from an empty directory to the current state (architecture, 278 tests, this README) — came out of that process.

The pivot to fully local transcription came from wanting a tool I'd actually trust with sensitive meeting audio. faster-whisper + webrtcvad gives sub-3-second latency with zero cloud calls, and GPU acceleration via CTranslate2 keeps inference fast without any model quality trade-off.

I use this daily for recording meetings. Inspired in part by TUIs I've worked with recently, primarily [OpenCode](https://github.com/anomalyco/opencode/).

---

## Acknowledgments

- **[Claude Code](https://github.com/anthropics/claude-code)** — Used to build this project iteratively
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — GPU-accelerated Whisper inference via CTranslate2
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** — Original fast local speech recognition by Georgi Gerganov
- **[webrtcvad](https://github.com/wiseman/py-webrtcvad)** — Voice activity detection (Google WebRTC VAD)
- **[LangChain](https://github.com/langchain-ai/langchain)** — Flexible framework for RAG and LLM chains
- **Rich** and **prompt_toolkit** — Terminal output and REPL input

---

## License

MIT License — See [LICENSE](./LICENSE) file for details.

---

## Learn More

- **See the development process**: [CLAUDE.md](./CLAUDE.md) — Complete TODO history and architecture decisions
- **Read the code**: All modules are documented with clear data flow and extension points
- **Run the tests**: 278 automated tests demonstrate usage patterns
