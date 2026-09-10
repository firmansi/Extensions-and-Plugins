---
name: youtube-transcript
description: Fetch and summarize YouTube video transcripts. Use when asked to summarize, transcribe, or extract content from YouTube videos. Handles transcript fetching via residential IP proxy to bypass YouTube's cloud IP blocks.
---

# YouTube Transcript

Fetch transcripts from YouTube videos and optionally summarize them.

## Tool Usage

Use the `fetch_transcript` tool to get transcript data:

```
Tool: fetch_transcript
Parameters:
  - video_url: YouTube video URL or video ID
  - languages: (optional) Comma-separated language codes
```

**Examples:**
```
Tool: fetch_transcript
video_url: https://www.youtube.com/watch?v=dQw4w9WgXcQ

Tool: fetch_transcript
video_url: dQw4w9WgXcQ
languages: fr,en,de
```

**Output:** JSON with `video_id`, `title`, `author`, `full_text`, and timestamped `transcript` array.

## Workflow

1. Call `fetch_transcript` tool with video URL or ID
2. Tool handles VPN connection automatically
3. Returns JSON with full transcript text
4. Summarize the `full_text` field as needed

## Language Codes

Default priority: `en, fr, de, es, it, pt, nl`

Override with `languages` parameter: `fetch_transcript(video_url="ID", languages="ja,ko,zh")`

## Setup & Configuration

See [references/SETUP.md](references/SETUP.md) for:
- Python dependencies installation
- WireGuard VPN configuration (required for cloud VPS)
- Troubleshooting common errors
- Alternative proxy options