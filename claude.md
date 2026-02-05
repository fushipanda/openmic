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
- [ ] Connect to Scribe realtime WebSocket: stream mic audio chunks as `input_audio_chunk` messages
- [ ] Receive `partial_transcript` and `committed_transcript` events and display them in the main pane as words arrive

### Phase 3 — Batch transcription and saving
- [ ] On stop: upload the recorded `.wav` to Scribe batch API with `diarize=True` and `num_speakers` defaulting to a reasonable max
- [ ] Parse the batch response into speaker-labeled segments
- [ ] Save the diarized transcript to `transcripts/` as a markdown file named `YYYY-MM-DD_HH-MM.md`
- [ ] Replace the live preview in the TUI with the final diarized transcript once batch completes
- [ ] Delete the temporary `.wav` file after successful batch transcription

### Phase 4 — RAG querying
- [ ] Wire up LangChain: document loader for markdown files in `transcripts/`, local vector store (FAISS or ChromaDB), embeddings via the configured provider
- [ ] Implement `/query` command: user types a question in the command input, relevant chunks are retrieved, LLM generates an answer, result displayed in the main pane

### Phase 5 — Meeting notes
- [ ] Implement `/notes` command: LangChain chain that takes the most recent session transcript and produces structured notes (agenda, key points, decisions, action items)
- [ ] Save generated notes to `notes/` alongside the source transcript
- [ ] Display the generated notes in the main pane

### Phase 6 — Polish
- [ ] Add optional session naming: prompt at start, used in the transcript filename
- [ ] Update README with setup, `.env` config, and command reference

---

When all TODO items are checked off, output: `<promise>COMPLETE</promise>`
