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

---

## Developer Guide

### Module layout

```
openmic/
├── app.py          # TUI entry point. Owns all widgets and orchestrates the others.
├── audio.py        # Mic capture via sounddevice. Writes 16kHz mono WAV to disk.
├── transcribe.py   # ElevenLabs client wrappers: RealtimeTranscriber (WebSocket)
│                   #   and BatchTranscriber (REST upload + diarization parsing).
├── storage.py      # File I/O: read/write transcripts/ and notes/ as markdown.
├── rag.py          # LangChain RAG stack: loads transcripts, chunks them,
│                   #   builds a FAISS index, runs RetrievalQA.
└── notes.py        # LangChain summarisation chain: prompt → LLM → structured notes.
```

### Data flow

```
/start
  app._start_recording()
    ├── audio.AudioRecorder.start()          → opens mic stream, accumulates frames
    └── transcribe.RealtimeTranscriber.connect()

  [while recording]
    audio callback                           → fires on_audio_chunk every 1024 samples
      └── transcribe.RealtimeTranscriber.send_audio_chunk()   ← stub, see below

/stop
  app._stop_recording()
    ├── audio.AudioRecorder.stop()           → flushes frames, writes .wav, returns path
    ├── transcribe.RealtimeTranscriber.disconnect()
    └── app._run_batch_transcription(wav_path)
          ├── transcribe.BatchTranscriber.transcribe_file()   → uploads to Scribe REST
          ├── transcribe.BatchTranscriber.parse_diarized_result()
          ├── storage.save_transcript(segments)               → writes transcripts/*.md
          ├── app._display_diarized_transcript()              → replaces TUI pane
          └── app._cleanup_wav()                              → deletes .wav

/query <question>
  app._run_query()
    └── rag.TranscriptRAG.query()
          ├── .refresh()  (if not yet built)
          │     ├── DirectoryLoader  → reads transcripts/*.md
          │     ├── FAISS.from_documents()
          │     └── RetrievalQA chain
          └── .invoke()  → returns answer string

/notes
  app._generate_notes()
    └── notes.generate_notes_for_latest()
          ├── storage.get_latest_transcript()
          ├── LLMChain(prompt=NOTES_PROMPT).run()
          └── storage.save_notes()  → writes notes/*_notes.md
```

### Known limitations / stubs

- **Realtime WebSocket is not wired up.** `RealtimeTranscriber.send_audio_chunk()`
  base64-encodes each chunk but discards it. The actual WebSocket streaming to
  ElevenLabs Scribe has not been implemented. The live preview pane will remain
  empty during recording; the diarized transcript appears only after `/stop`
  completes the batch call.

- **Embeddings always use OpenAI.** `rag.get_embeddings()` returns
  `OpenAIEmbeddings()` regardless of `LLM_PROVIDER`. Anthropic does not expose
  an embeddings API, so an `OPENAI_API_KEY` is required for `/query` even when
  `LLM_PROVIDER=anthropic`. The `.env.example` does not call this out.

- **Vector store is in-memory only.** The FAISS index is rebuilt from disk on
  every first `/query` in a session. It is not persisted between runs.

- **`LLMChain` / `chain.run()` in notes.py are deprecated** in newer LangChain
  versions. Works now, but will need updating when the library drops them.

### Extending

- **Swap the vector store:** replace `FAISS` in `rag.py` with ChromaDB or another
  `langchain_community.vectorstores` backend. The rest of the code is unaware of
  the choice.

- **Swap the LLM provider:** add a new branch in `rag.get_llm()` and install the
  matching `langchain-<provider>` package. Nothing else needs to change.

- **Wire up realtime streaming:** fill in `RealtimeTranscriber.send_audio_chunk()`
  using the ElevenLabs SDK's WebSocket client, and call `self.on_partial` /
  `self.on_committed` from the message handler. The TUI callbacks in `app.py`
  are already connected.
