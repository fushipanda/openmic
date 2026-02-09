# OpenMic TODO List

## Feature Requests

### ✅ FR-22: Session Credit Usage Display (COMPLETED - Enhanced)
- ✅ Display credit usage for the current session in top right corner
- ✅ If possible, show total remaining credits (otherwise just session usage)
- ✅ No need to show percentage unless total credits available

NOTE: This feature was already implemented. Enhanced the display to clearly show "Session:" prefix to make it more obvious that credit/usage tracking is active. Display shows:
- Audio usage (time in seconds/minutes)
- LLM calls (with token counts)
- Example: "Session: Audio: 2.5m · LLM: 3 calls (450 tok)"

### ✅ FR-23: Reverse Star Indicator for Notes (COMPLETED)
- ✅ Change visual indicator to star notes that **haven't** been generated yet
- ✅ Currently stars notes that have been generated - reverse this behavior

### ✅ FR-24: Rich Text Formatting for Markdown (COMPLETED - Already Implemented as FR-19)
- ✅ Add rich formatting for headings (currently shows as `#`)
- ✅ Add rich formatting for bold text (currently shows as `**text**`)
- ✅ Improve markdown display readability

NOTE: This was already implemented in FR-19. The TranscriptPane uses RichMarkdown from the rich library which automatically formats headings, bold text, lists, and other markdown elements when viewing transcripts and notes.

### ✅ FR-25: Improve Notes Title Formatting (COMPLETED)
- ✅ Current format: `Meeting Transcript - 2026-01-01_12-00`
- ✅ Format this more nicely
- ✅ Don't need to put name in brackets

NEW FORMAT:
- With session name: `# Session Name` + `*Jan 15th 2026, 2:30 PM*`
- Without session name: `# Meeting Notes` + `*Jan 15th 2026, 2:30 PM*`
- Date displayed in italics below the heading for cleaner appearance

## Bugs

### ✅ BUG-7: Command Popup Not Displaying (COMPLETED)
- ✅ Command popup autocomplete box is not visually showing commands
- ✅ Currently has autocomplete with Enter key
- ✅ Need to add Tab key support for autocomplete
- ✅ Fix commands not loading/appearing in the dropdown box
