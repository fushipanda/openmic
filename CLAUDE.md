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

---

## Active TODO

### ✅ FR-27: Chat UI Redesign
- [x] User messages: `  > question` with accent-colored `>` prefix, no "You:" label
- [x] AI responses: rendered as Rich Markdown via `RichGroup`, no "AI:" label
- [x] Sources displayed as muted line after markdown
- [x] `_run_query_on_path` uses same styled format
- [x] `_rebuild_rich_display` uses `RichGroup` to support mixed renderables

### ✅ FR-28: RAG Quality Improvements
- [x] Retrieval switched to MMR with k=8, fetch_k=20, lambda_mult=0.7
- [x] History-aware prompt updated to preserve specific names/dates when reformulating
- [x] QA prompt rewritten with structured rules (quote, partial, not-found guidance)
- [x] Same improvements applied to `query_file()` for single-transcript queries
- [x] Extended thinking support via `LLM_EXTENDED_THINKING=true` env var (Anthropic)

### ✅ FR-29: Multi-Provider Model Support + `/model` Command
- [x] `get_llm()` supports `LLM_MODEL` env var, Gemini, and OpenRouter providers
- [x] `MODEL_REGISTRY` constant with curated models for each provider
- [x] `ModelPickerScreen` two-step modal (provider → model → optional API key)
- [x] `/model` command added to SLASH_COMMANDS, HELP_COMMANDS, and command handler
- [x] On confirm: updates `os.environ`, writes to `.env`, saves to `settings.json`, rebuilds RAG
- [x] `LLM_PROVIDER`/`LLM_MODEL` loaded from `settings.json` on startup
- [x] Gemini optional dependency added to `pyproject.toml`
- [x] `.env.example` updated with all new keys and comments

### ✅ FR-30: ElevenLabs Scribe Enhancements
- [x] Realtime WebSocket URL includes `language_code=en`
- [x] Batch transcription adds `language_code="en"` and `tag_audio_events=True`
- [x] `ELEVENLABS_SCRIBE_MODEL` env var controls model (default `scribe_v1`)

### ✅ BUG-8: Tab cycling for command autocomplete
- [x] Tab cycles forward through matching commands without hiding the dropdown
- [x] Shift+Tab cycles backward through matches
- [x] Dropdown stays visible while cycling and updates the highlight
- [x] Enter executes the currently shown command
- [x] Escape hides the dropdown
- [x] `_tab_cycling` flag prevents `on_input_changed` from resetting matches during cycling
- [x] Replaced Rich `Text` with Textual's native `Content` class in `_render_content` to fix invisible dropdown text

### ✅ BUG-9: TranscriptPane not scrollable with mouse wheel or Page Up/Down
- [x] Changed `overflow-y: auto` to `overflow-y: scroll` in TranscriptPane CSS
- [x] Added app-level `pageup`/`pagedown` bindings that delegate to `transcript_pane.scroll_page_up()`/`scroll_page_down()`
- [x] Re-added `allow_vertical_scroll` override — required because `scrollbar-size-vertical: 0` makes `show_vertical_scrollbar` false, which disables scroll events
- [x] Set `can_focus = True` on TranscriptPane for mouse wheel event handling
- [x] Scrollbar remains hidden (`scrollbar-size-vertical: 0`)

---

## ✅ PROJECT STATUS: ALL ITEMS COMPLETED (prior to BUG-8/BUG-9)

All feature requests and bugs have been successfully implemented, tested, and pushed to GitHub!

**Recent Commits:**
- `a649606` - BUG-7: Fix command autocomplete dropdown display and add Tab support
- `f09fe4a` - FR-23: Reverse star indicator to highlight transcripts without notes
- `e183fe8` - FR-25: Improve notes title formatting for better readability
- `5794ccc` - FR-22: Enhance session credit/usage display with clearer labeling

**Tests:** 163/163 passing ✅

---

## Completed Feature Requests

### ✅ FR-17: Combine themes and palettes
- [x] Removed duplicate custom nord theme (conflicted with Textual's built-in nord)
- [x] Consolidated to single custom theme (openmic) + curated Textual built-in themes
- [x] All themes use Textual's Theme API — title/banner colors derive from current_theme
- [x] Theme choice persists through sessions via settings.json

### ✅ FR-18: Visual indicator for generated notes
- [x] Add green `*` indicator in transcript picker for transcripts with existing notes
- [x] Indicator checks notes/ directory using the same naming convention as storage

### ✅ FR-19: Rich text formatting for markdown
- [x] Added `set_markdown()` method to TranscriptPane using Rich's Markdown renderer
- [x] Notes display now renders with proper headings, bold, lists, and formatting
- [x] Transcript viewing renders markdown formatting (headings, bold speaker names, etc.)

### ✅ FR-20: Improve notes title formatting
- [x] Titles now use readable format: "Meeting Transcript — Jan 1st 2026, 12:00 PM"
- [x] Named sessions show: "standup — Jan 1st 2026, 12:00 PM" (no brackets)
- [x] Applied to saved transcripts, renamed transcripts, and notes headers

### ✅ FR-21: Easy navigation back from transcript/notes view
- [x] Esc key returns to home screen (banner) when viewing a transcript, notes, or query results
- [x] Documented in help screen shortcuts

### ✅ FR-22: Session Credit Usage Display
- [x] Display credit usage for the current session in top right corner
- [x] If possible, show total remaining credits (otherwise just session usage)
- [x] No need to show percentage unless total credits available

**Implementation:** Enhanced the display to clearly show "Session:" prefix to make it more obvious that credit/usage tracking is active. Display shows:
- Audio usage (time in seconds/minutes)
- LLM calls (with token counts)
- Example: "Session: Audio: 2.5m · LLM: 3 calls (450 tok)"

### ✅ FR-23: Reverse Star Indicator for Notes
- [x] Change visual indicator to star notes that **haven't** been generated yet
- [x] Currently stars notes that have been generated - reverse this behavior

### ✅ FR-24: Rich Text Formatting for Markdown
- [x] Add rich formatting for headings (currently shows as `#`)
- [x] Add rich formatting for bold text (currently shows as `**text**`)
- [x] Improve markdown display readability

**Note:** This was already implemented in FR-19. The TranscriptPane uses RichMarkdown from the rich library which automatically formats headings, bold text, lists, and other markdown elements when viewing transcripts and notes.

### ✅ FR-25: Improve Notes Title Formatting
- [x] Current format: `Meeting Transcript - 2026-01-01_12-00`
- [x] Format this more nicely
- [x] Don't need to put name in brackets

**New Format:**
- With session name: `# Session Name` + `*Jan 15th 2026, 2:30 PM*`
- Without session name: `# Meeting Notes` + `*Jan 15th 2026, 2:30 PM*`
- Date displayed in italics below the heading for cleaner appearance

---

## Completed Bugs

### ✅ BUG-5: Command autocomplete not working
- [x] Fixed: autocomplete dropdown now uses overlay layer to render above other widgets
- [x] Added bottom margin to clear command input/footer
- [x] Screen CSS defines overlay layer for proper z-ordering

### ✅ BUG-6: Note summaries regeneration issue
- [x] App now checks for cached notes before calling LLM (via get_existing_notes)
- [x] Usage tracker only counts LLM calls when notes are actually generated (not cached)
- [x] UI shows "Loading saved notes..." for cached notes vs "Generating..." for new

### ✅ BUG-7: Command Popup Not Displaying
- [x] Command popup autocomplete box is not visually showing commands
- [x] Currently has autocomplete with Enter key
- [x] Need to add Tab key support for autocomplete
- [x] Fix commands not loading/appearing in the dropdown box

---

## Original TODO (Completed)

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
- [x] Implement `RealtimeTranscriber.send_audio_chunk()` to actually stream audio to ElevenLabs Scribe WebSocket
- [x] Handle incoming `partial_transcript` and `committed_transcript` messages from WebSocket
- [x] Add pytest + pytest-asyncio to dev dependencies
- [x] Unit tests for `storage.py`: save/load transcripts and notes
- [x] Unit tests for `transcribe.py`: parse_diarized_result with sample responses
- [x] Integration test for RAG pipeline with mocked embeddings
- [x] Integration test for notes generation with mocked LLM
- [x] Update README: clarify that OPENAI_API_KEY is required for embeddings even with `LLM_PROVIDER=anthropic`
- [x] Update .env.example with all required keys and comments
- [x] Verify all commands in README match actual implementation
