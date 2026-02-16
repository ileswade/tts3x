# TTS3 Rewrite Architecture

## Objective
A deterministic, local-first content pipeline for ReadStack preparation and playback sync.

## Pipeline
1. `import`
- Input: PDF/DOCX/TXT/MD
- Output: canonical `.md`
- Notes: OCR mode deferred to next pass.

2. `adapt`
- Input: canonical `.md`
- Output: derived markdown variants:
  - `-speakable.md`
  - `-explained.md`
  - `-summary.md` (optional)
- URL and image mention behavior is controlled here.
- `explained` is AI-generated paragraph-by-paragraph with local context window
  (previous N + current + next N, configurable; default N=2).
- Prompt shaping for `explained` is profile-driven:
  - built-in profiles in code (`academic_default`, `whitepaper_education`, `novel_literary`)
  - optional overrides via JSON prompt library file.
- `speakable` v2 is deterministic and structure-aware:
  - bullets become narrated sequence (`First... Second... Finally...`)
  - tables become placeholders (`A table is provided for this section.`)
  - blockquotes become quote-speaker lines (`[Quote]: ...`)
  - URL/image metadata is persisted for later enrichment (`<!--URL:...-->`, `<!--IMG:...-->`)

3. `synth`
- Input: base `.md` or a specific derived `.md`
- Output:
  - `.mp3`
  - `.synth.json` (sidecar)
- Rule set:
  - If input is derived (`-speakable.md`, `-explained.md`, `-summary.md`): synth only that file.
  - If input is base and `--format` omitted: synth all available formats.
- Sidecar stores the exact spoken plan (paragraphs/chunks/voices/policies/hashes).
- Quote speaker fallback voice defaults to `fable` unless overridden via `--voice` mapping.

4. `captions`
- Input: `.mp3`
- Output:
  - `.words.json` (word timing)
  - `.vtt`
  - optional enriched `.json`
- Local mode uses Apple Speech directly (pyobjc) and keeps optional cached `.wav` (`--keep-wav`, `--overwrite-wav`).

5. `enrich` (integrated into `captions` when `--md` provided)
- Merges timed words with structural metadata (headers/URLs/images)
- Preferred alignment source: `.synth.json` sidecar (exact mode)
- Fallback: markdown token estimation (approximate mode)

6. `read`
- Input: `.mp3` + `.words.json`
- Terminal playback with focused word rendering and controls:
  - space pause/resume
  - left/right seek
  - `+/-` adjust surrounding context window

## Why sidecar is required
STT provides accurate timestamps, but not guaranteed exact lexical identity with source text.
The sidecar preserves canonical spoken sequence and paragraph mapping from synth time, so
metadata events can be attached at exact paragraph boundaries without drift.

## OCR status
- `import --ocr auto|force|never` is part of the CLI contract.
- `auto`/`never` currently use text extraction (`pypdf` for PDF).
- `force` is reserved and intentionally errors until OCR backend is added.

## JSON outputs
- `*.words.json`: raw STT output with word timings
- `*.synth.json`: synth-time plan and policy metadata
- `*.json`: app playback timeline (`words`, `paragraphs`, `header_events`)

## UI implications (iPhone app)
- Word highlighting uses `words[]` only.
- Header fades use `header_events[]` and `cooldown_sec`.
- URL/image indicators are paragraph metadata and do not interrupt word highlighting.
