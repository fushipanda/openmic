# openmic

A CLI/TUI meeting transcription tool built with Python and Textual. Records audio, streams
live transcription to screen via ElevenLabs Scribe realtime, then runs a diarized batch
transcription on stop. Saved transcripts are queryable via LangChain RAG, with a command
to generate structured meeting notes.

## Architecture

```
During session:
  Mic ──► WebSocket (Scribe realtime) ──► live text displayed in TUI (no speaker labels)
  Mic ──► local .wav file              ──► audio buffer on disk

On stop:
  .wav ──► Scribe batch API (diarize=True) ──► Speaker 1/2/etc labeled transcript
                                            ──► saved to transcripts/ as markdown
                                            ──► final labeled transcript replaces live preview on screen

Post-session:
  /query ──► LangChain RAG over transcripts/ ──► answer in TUI
  /notes ──► LangChain summarisation chain   ──► structured notes saved to notes/
```

## Tech Stack

- Python 3.12+
- Textual — TUI framework
- ElevenLabs Python SDK — Scribe realtime (WebSocket) + batch transcription
- LangChain — RAG querying and notes generation (model-agnostic, provider swappable via .env)
- sounddevice — mic capture
- FAISS or ChromaDB — local vector store for RAG

## .env Keys

```
ELEVENLABS_API_KEY=...       # already present
LLM_PROVIDER=anthropic       # or openai — controls which LangChain provider is used
ANTHROPIC_API_KEY=...        # add whichever provider you set above
# OPENAI_API_KEY=...
```

## Strict Instructions

- Commit to git after each TODO item is completed. Message should reference the item.
- Never commit `.env`, `*.wav`, or anything in `transcripts/` or `notes/`. These must be in `.gitignore` from the first commit.
- Do not install a dependency if something already in `pyproject.toml` covers the need.
- All LLM calls must route through LangChain. Do not call any provider SDK directly.
- Use keys from `.env` via environment variables. Never hardcode API keys.
- If a task is blocked, skip it and move to the next one. Do not retry indefinitely.
- Run any existing tests before marking a task complete.
- Temporary `.wav` files should be cleaned up after a successful batch transcription.

## TODO

### Phase 1 — Project foundation
- [x] Init project: `pyproject.toml`, `.gitignore`, `.env.example` (template with key names, no values)
- [x] Textual app skeleton: status bar (idle / recording), main transcript pane, command input at the bottom
- [x] Mic capture with sounddevice: record audio continuously to a local `.wav` while session is active

### Phase 2 — Realtime transcription (live preview)
- [x] Connect to Scribe realtime WebSocket: stream mic audio chunks as `input_audio_chunk` messages
- [x] Receive `partial_transcript` and `committed_transcript` events and display them in the main pane as words arrive

### Phase 3 — Batch transcription and saving
- [x] On stop: upload the recorded `.wav` to Scribe batch API with `diarize=True` and `num_speakers` defaulting to a reasonable max
- [x] Parse the batch response into speaker-labeled segments
- [x] Save the diarized transcript to `transcripts/` as a markdown file named `YYYY-MM-DD_HH-MM.md`
- [x] Replace the live preview in the TUI with the final diarized transcript once batch completes
- [x] Delete the temporary `.wav` file after successful batch transcription

### Phase 4 — RAG querying
- [x] Wire up LangChain: document loader for markdown files in `transcripts/`, local vector store (FAISS or ChromaDB), embeddings via the configured provider
- [x] Implement `/query` command: user types a question in the command input, relevant chunks are retrieved, LLM generates an answer, result displayed in the main pane

### Phase 5 — Meeting notes
- [x] Implement `/notes` command: LangChain chain that takes the most recent session transcript and produces structured notes (agenda, key points, decisions, action items)
- [x] Save generated notes to `notes/` alongside the source transcript
- [x] Display the generated notes in the main pane

### Phase 6 — Polish
- [x] Add optional session naming: prompt at start, used in the transcript filename
- [x] Update README with setup, `.env` config, and command reference

### Phase 7 — Testing & Bug Fixes

#### 7.1 — Wire up Realtime WebSocket
- [x] Implement `RealtimeTranscriber.send_audio_chunk()` to actually stream audio to ElevenLabs Scribe WebSocket
- [x] Handle incoming `partial_transcript` and `committed_transcript` messages from WebSocket
- [ ] Verify live text appears in TUI during recording (before `/stop`) *(requires manual testing)*

#### 7.2 — Automated Tests (pytest)
- [x] Add pytest + pytest-asyncio to dev dependencies
- [x] Unit tests for `storage.py`: save/load transcripts and notes
- [x] Unit tests for `transcribe.py`: parse_diarized_result with sample responses
- [x] Integration test for RAG pipeline with mocked embeddings
- [x] Integration test for notes generation with mocked LLM

#### 7.3 — Manual TUI Verification
- [ ] `/start` — recording starts, status changes to RECORDING, mic captures audio
- [ ] `/start <name>` — session name appears in status/filename
- [ ] `/stop` — batch transcription runs, diarized output displays, .wav deleted
- [ ] `/query <question>` — RAG returns relevant answer from transcripts/
- [ ] `/notes` — generates structured notes, saves to notes/
- [ ] `Ctrl+R` — toggles recording on/off
- [ ] `Ctrl+C` — quits cleanly

#### 7.4 — Documentation Audit
- [x] Update README: clarify that OPENAI_API_KEY is required for embeddings even with `LLM_PROVIDER=anthropic`
- [x] Update .env.example with all required keys and comments
- [x] Verify all commands in README match actual implementation

---

## Feature Requests

### FR-17: Combine themes and palettes ✅
- [x] Removed duplicate custom nord theme (conflicted with Textual's built-in nord)
- [x] Consolidated to single custom theme (openmic) + curated Textual built-in themes
- [x] All themes use Textual's Theme API — title/banner colors derive from current_theme
- [x] Theme choice persists through sessions via settings.json

### FR-18: Visual indicator for generated notes ✅
- [x] Add green `*` indicator in transcript picker for transcripts with existing notes
- [x] Indicator checks notes/ directory using the same naming convention as storage

### FR-19: Rich text formatting for markdown ✅
- [x] Added `set_markdown()` method to TranscriptPane using Rich's Markdown renderer
- [x] Notes display now renders with proper headings, bold, lists, and formatting
- [x] Transcript viewing renders markdown formatting (headings, bold speaker names, etc.)

### FR-20: Improve notes title formatting ✅
- [x] Titles now use readable format: "Meeting Transcript — Jan 1st 2026, 12:00 PM"
- [x] Named sessions show: "standup — Jan 1st 2026, 12:00 PM" (no brackets)
- [x] Applied to saved transcripts, renamed transcripts, and notes headers

### FR-21: Easy navigation back from transcript/notes view ✅
- [x] Esc key returns to home screen (banner) when viewing a transcript, notes, or query results
- [x] Documented in help screen shortcuts

## Bugs

### BUG-5: Command autocomplete not working ✅
- [x] Fixed: autocomplete dropdown now uses overlay layer to render above other widgets
- [x] Added bottom margin to clear command input/footer
- [x] Screen CSS defines overlay layer for proper z-ordering

### BUG-6: Note summaries regeneration issue
- Note summaries that have already been generated with AI are being regenerated on subsequent calls
- Should save and reuse the existing meeting summary instead of making new LLM calls
- Only regenerate if explicitly requested or if the transcript has changed

---

When all TODO items are checked off, output: `<promise>COMPLETE</promise>`
