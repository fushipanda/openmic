"""OpenMic TUI application."""

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Input, Static
from textual.binding import Binding

from dotenv import load_dotenv

from openmic.audio import AudioRecorder
from openmic.transcribe import BatchTranscriber, RealtimeTranscriber
from openmic.storage import save_transcript
from openmic.rag import TranscriptRAG
from openmic.notes import generate_notes_for_latest

load_dotenv()


class StatusBar(Static):
    """Status bar showing recording state."""

    def __init__(self) -> None:
        super().__init__("Status: IDLE")
        self.recording = False

    def set_recording(self, recording: bool) -> None:
        self.recording = recording
        status = "RECORDING" if recording else "IDLE"
        self.update(f"Status: {status}")


class TranscriptPane(Static):
    """Main pane displaying transcript text."""

    def __init__(self) -> None:
        super().__init__("")

    def append_text(self, text: str) -> None:
        current = str(self.renderable)
        self.update(current + text)

    def set_text(self, text: str) -> None:
        self.update(text)

    def clear(self) -> None:
        self.update("")


class CommandInput(Input):
    """Command input at the bottom of the screen."""

    def __init__(self) -> None:
        super().__init__(placeholder="Type /start, /stop, /query, or /notes...")


class OpenMicApp(App):
    """OpenMic TUI application."""

    CSS = """
    StatusBar {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    StatusBar.recording {
        background: $error;
    }

    TranscriptPane {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        overflow-y: auto;
    }

    CommandInput {
        dock: bottom;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+r", "toggle_recording", "Record"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.status_bar = StatusBar()
        self.transcript_pane = TranscriptPane()
        self.command_input = CommandInput()
        self.audio_recorder = AudioRecorder(
            output_dir=Path("."),
            on_audio_chunk=self._on_audio_chunk,
        )
        self.transcriber = RealtimeTranscriber(
            on_partial=self._on_partial_transcript,
            on_committed=self._on_committed_transcript,
        )
        self._live_text = ""
        self.batch_transcriber = BatchTranscriber()
        self._current_wav_path: Path | None = None
        self._latest_transcript_path: Path | None = None
        self.rag = TranscriptRAG()
        self._session_name: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.status_bar
        yield Container(self.transcript_pane)
        yield self.command_input
        yield Footer()

    async def action_toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.status_bar.recording:
            await self._stop_recording()
        else:
            await self._start_recording()

    async def _start_recording(self) -> None:
        """Start audio recording and transcription."""
        self._live_text = ""
        self._current_wav_path = self.audio_recorder.start()
        await self.transcriber.connect()
        self.status_bar.set_recording(True)
        self.status_bar.add_class("recording")
        session_info = f" [{self._session_name}]" if self._session_name else ""
        self.transcript_pane.set_text(f"Recording started...{session_info} ({self._current_wav_path.name})\n\n")

    async def _stop_recording(self) -> None:
        """Stop audio recording and transcription."""
        wav_path = self.audio_recorder.stop()
        await self.transcriber.disconnect()
        self.status_bar.set_recording(False)
        self.status_bar.remove_class("recording")
        if wav_path:
            self.transcript_pane.append_text(f"\n\nRecording stopped. Processing with diarization...\n")
            self._current_wav_path = wav_path
            await self._run_batch_transcription(wav_path)

    def _on_audio_chunk(self, audio_bytes: bytes) -> None:
        """Handle audio chunk from recorder."""
        self.transcriber.send_audio_chunk(audio_bytes)

    def _on_partial_transcript(self, text: str) -> None:
        """Handle partial transcript update."""
        self.call_from_thread(self._update_partial, text)

    def _on_committed_transcript(self, text: str) -> None:
        """Handle committed transcript."""
        self.call_from_thread(self._update_committed, text)

    def _update_partial(self, text: str) -> None:
        """Update transcript pane with partial text."""
        display = self._live_text + text
        self.transcript_pane.set_text(display)

    def _update_committed(self, text: str) -> None:
        """Update transcript pane with committed text."""
        self._live_text += text + " "
        self.transcript_pane.set_text(self._live_text)

    async def _run_batch_transcription(self, wav_path: Path) -> None:
        """Run batch transcription with diarization."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.batch_transcriber.transcribe_file(str(wav_path)),
            )
            segments = BatchTranscriber.parse_diarized_result(result)
            self._latest_transcript_path = save_transcript(segments, self._session_name)
            self._display_diarized_transcript(segments)
            self._cleanup_wav(wav_path)
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError during transcription: {e}\n")

    def _display_diarized_transcript(self, segments: list[dict]) -> None:
        """Display diarized transcript in the pane."""
        lines = []
        for segment in segments:
            speaker = segment.get("speaker", "Speaker")
            text = segment.get("text", "")
            lines.append(f"[{speaker}]: {text}\n")

        self.transcript_pane.set_text("\n".join(lines))
        if self._latest_transcript_path:
            self.transcript_pane.append_text(f"\n\nSaved to: {self._latest_transcript_path}\n")

    def _cleanup_wav(self, wav_path: Path) -> None:
        """Delete temporary WAV file after successful transcription."""
        try:
            wav_path.unlink()
        except OSError:
            pass

    async def _run_query(self, question: str) -> None:
        """Run a RAG query against the transcripts."""
        self.transcript_pane.set_text(f"Querying: {question}\n\nSearching transcripts...\n")
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.rag.query(question),
            )
            self.transcript_pane.set_text(f"Q: {question}\n\nA: {answer}\n")
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError during query: {e}\n")

    async def _generate_notes(self) -> None:
        """Generate meeting notes from the latest transcript."""
        self.transcript_pane.set_text("Generating meeting notes...\n")
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                generate_notes_for_latest,
            )
            if result is None:
                self.transcript_pane.set_text("No transcripts available to generate notes from.\n")
            else:
                notes_content, notes_path = result
                self.transcript_pane.set_text(f"{notes_content}\n\nSaved to: {notes_path}\n")
        except Exception as e:
            self.transcript_pane.append_text(f"\n\nError generating notes: {e}\n")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input."""
        command = event.value.strip()
        self.command_input.value = ""

        if command == "/start":
            if not self.status_bar.recording:
                self._session_name = None
                await self._start_recording()
        elif command.startswith("/start "):
            if not self.status_bar.recording:
                self._session_name = command[7:].strip().replace(" ", "_")
                await self._start_recording()
        elif command == "/stop":
            if self.status_bar.recording:
                await self._stop_recording()
        elif command.startswith("/query"):
            query_text = command[6:].strip()
            if query_text:
                await self._run_query(query_text)
            else:
                self.transcript_pane.append_text("\nUsage: /query <your question>\n")
        elif command == "/notes":
            await self._generate_notes()
        elif command:
            self.transcript_pane.append_text(f"\nUnknown command: {command}\n")


def main() -> None:
    """Entry point for the OpenMic application."""
    app = OpenMicApp()
    app.run()


if __name__ == "__main__":
    main()
