# openmic

A CLI/TUI meeting transcription tool built with Python and Textual. Records audio, streams live transcription to screen via ElevenLabs Scribe realtime, then runs a diarized batch transcription on stop. Saved transcripts are queryable via LangChain RAG, with a command to generate structured meeting notes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/openmic.git
cd openmic

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[anthropic]"  # or [openai] for OpenAI
```

## Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```
ELEVENLABS_API_KEY=your_elevenlabs_api_key
LLM_PROVIDER=anthropic       # or openai
ANTHROPIC_API_KEY=your_anthropic_api_key
# OPENAI_API_KEY=your_openai_api_key
```

## Usage

Start the application:

```bash
openmic
```

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Start recording a meeting |
| `/start <name>` | Start recording with a session name |
| `/stop` | Stop recording and process with diarization |
| `/query <question>` | Ask a question about past meetings |
| `/notes` | Generate structured notes from the last meeting |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Toggle recording |
| `Ctrl+C` | Quit application |

## Features

- **Live Transcription**: See words appear as you speak via ElevenLabs Scribe realtime
- **Speaker Diarization**: Final transcripts include speaker labels (Speaker 1, Speaker 2, etc.)
- **RAG Querying**: Ask questions about any past meeting transcript
- **Meeting Notes**: Auto-generate structured notes with agenda, key points, decisions, and action items
- **Session Naming**: Name your meetings for easier organization

## File Storage

- Transcripts: `transcripts/YYYY-MM-DD_HH-MM.md` (or with session name)
- Meeting notes: `notes/YYYY-MM-DD_HH-MM_notes.md`

## Architecture

```
During session:
  Mic ──► WebSocket (Scribe realtime) ──► live text displayed in TUI
  Mic ──► local .wav file              ──► audio buffer on disk

On stop:
  .wav ──► Scribe batch API (diarize=True) ──► Speaker-labeled transcript
                                            ──► saved to transcripts/
                                            ──► displayed in TUI

Post-session:
  /query ──► LangChain RAG over transcripts/ ──► answer in TUI
  /notes ──► LangChain summarisation chain   ──► structured notes
```

## Requirements

- Python 3.12+
- ElevenLabs API key (for transcription)
- Anthropic or OpenAI API key (for RAG and notes generation)
