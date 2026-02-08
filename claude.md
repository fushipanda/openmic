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

### FR-1: Help popup on `?` key ✅
- Pressing `?` opens a centered popup listing all available commands with descriptions (similar to Claude Code's help)
- Should show command name, usage, and brief description
- Dismiss with `Esc` or `?` again

### FR-2: Rename Nord theme and remove Theme footer button ✅
- Rename `openmic-nord` theme to `nord`
- Remove `Ctrl+T` / "Theme" from the footer bindings (keep the binding functional, just hide from footer)

### FR-3: Scope `/query` and `/notes` to a specific transcript ✅
- `/query` and `/notes` should operate on a chosen transcript, not all transcripts / latest
- Ideally use the transcript picker popup (FR-5) to select which transcript

### FR-4: `/history` alias for `/transcripts` ✅
- Add `/history` as an alias that behaves identically to `/transcripts`
- `/transcript` (singular) should also work as an alias

### FR-5: Transcript history popup (redesign `/transcripts` and `/history`) ✅
- Replace the current numbered list with a centered modal popup (similar to opencode sessions)
- Group transcripts by date with headers: "Today", "Yesterday", or formatted date (e.g. "Jan 1st 2026")
- Each entry shows: `[Meeting Name]` on the left, time (e.g. `10:00 AM`) aligned to the right
- No numbered list (1. 2. 3.) — use selectable rows instead
- Navigable with arrow keys, select with Enter, dismiss with Esc

### FR-6: `/notes` transcript picker ✅
- When running `/notes`, show the transcript picker popup (FR-5) to choose which transcript to generate notes for
- If only one transcript exists, skip the picker and use it directly

### FR-7: Scrollable live transcript ✅
- During live transcription, the user should be able to scroll up/down through the transcript text
- Auto-scroll to bottom on new text, but respect manual scroll position (don't force-scroll if user scrolled up)

### FR-8: Structured live transcript (paragraph breaks) ✅
- Live transcript currently outputs as a single paragraph
- Use ElevenLabs `committed_transcript` events (triggered by VAD silence detection) as natural paragraph/line breaks
- Each committed segment should start on a new line rather than being concatenated inline

### FR-9: Prompt for session name after `/stop` ✅
- After `/stop` finishes batch transcription, prompt the user to name the session
- Could use the command input with a placeholder like "Name this session (Enter to skip)"
- If skipped (empty Enter), keep the timestamp-only filename

### FR-10: Pause command (`/pause`) ✅
- `/pause` stops mic capture and stops sending audio chunks to the WebSocket
- Keep the WebSocket connection alive (reconnect transparently if it times out)
- Keep the WAV file handle open for appending
- Transcript stays on screen, no batch processing triggered
- `/start` after pause resumes: restarts mic capture, continues sending chunks, appends to same WAV
- Status bar should show a "PAUSED" state (distinct from IDLE and RECORDING)
- `/stop` after pause should work normally — processes the full WAV with all segments

### FR-11 Shadow on title text ✅
- Openmic text should have a clear shadow down and left behind the text, ensure the shadow is aligned near the text and colours match theme

### FR-12: `/exit` command ✅
- Typing `/exit` in the command input exits the application cleanly
- Should behave the same as `Ctrl+C` / normal quit

### FR-13: Session credit usage display
- Show ElevenLabs and LLM credit/token usage for the current session
- Display in the top-right corner of the TUI
- Track ElevenLabs usage (transcription time/characters consumed)
- Track LLM token usage (Anthropic/OpenAI tokens for RAG queries and notes generation)
- No need to show percentage unless total remaining credits can be fetched from the API
- Reset counters on app start

### FR-14: Command autocomplete dropdown ✅
- When typing a `/` command, show a dropdown popup above the command bar with matching commands
- Filter as the user types: e.g. typing `/st` shows `/start`, `/stop` etc.
- Results sorted alphabetically
- Pressing Enter selects the top match
- Arrow keys to navigate, Esc to dismiss
- Each entry shows command name and brief description

### FR-15: Replace `?` help with `Ctrl+?` keyboard shortcut ✅
- Remove the `?` key binding for help
- Add `Ctrl+?` as the keyboard shortcut to toggle the help menu
- Help menu behaviour otherwise unchanged

### FR-16: Theme auto-save and persistence ✅
- When the user changes theme, automatically save the selection
- Persist theme choice across sessions (via `~/.config/openmic/settings.json`)
- On app start, load and apply the saved theme

## Bugs

### BUG-1: Opening text doesn't match theme ✅
- The initial/opening text displayed when the app launches does not use the current theme colors
- Should respect the active theme on startup

### BUG-2: History popup and help menu text blends with background ✅
- Text in the history popup (transcript picker) blends into the background depending on the theme
- Same issue with the help menu popup
- Text colors should contrast properly with the popup background across all themes

### BUG-3: Note summaries are regenerated unnecessarily ✅
- If `/notes` has already been run for a transcript and a summary exists in `notes/`, it should not regenerate
- Load and display the existing saved summary instead of making another LLM call
- Only regenerate if explicitly requested (or if the transcript has changed)

### BUG-4: Command bar lacks padding ✅
- All commands typed in the command bar start at the very left edge, touching the border
- Add left padding/space in the command input so text doesn't touch the border

---

When all TODO items are checked off, output: `<promise>COMPLETE</promise>`
