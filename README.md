# TTS3 Extended (tts3x)

TTS3 Extended (tts3x) is a standalone local-first content pipeline for TTS workflows.

## Goals
- Keep `import -> adapt -> synth -> captions -> read` as explicit phases.
- Preserve deterministic mapping between markdown structure and playback timeline.
- Emit a synth sidecar (`.synth.json`) so later enrichment is exact.
- Stay infrastructure-neutral (local processing first; cloud sync later).

## License + Attribution
- Licensed under Apache License 2.0 (`LICENSE`).
- Attribution must be preserved in redistributions and derivative works (`NOTICE`).
- Required credit: Iles Wade as the original author of TTS3 Extended (`tts3x`).

## Install
```bash
cd /Users/ileswade/ris/projects/tts3-extended
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# if you want local Apple captions/read support:
pip install pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-AVFoundation pyobjc-framework-Speech
```

## Config
Set `OPENAI_API_KEY` in environment or `.env`.

## Commands
```bash
# 1) Import (PDF/DOCX/TXT/MD -> MD)
tts3x import /path/file.pdf --ocr auto

# 2) Adapt
# outputs: <in>-speakable.md, <in>-explained.md, <in>-summary.md (as requested)
tts3x adapt /path/file.md --format speakable,explained,summary --url mention --img mention

# explained tuning (AI paragraph-by-paragraph with context window)
tts3x adapt /path/file.md --format explained --explained-model gpt-4o-mini --explained-window 2

# explicit back/forward context and listener persona
tts3x adapt /path/file.md --format explained \
  --explained-back 2 --explained-forward 2 \
  --explained-persona Iles

# list available explained prompt profiles
tts3x adapt /path/file.md --list-explained-profiles

# select profile (e.g., white paper mode)
tts3x adapt /path/file.md --format explained --explained-profile whitepaper_education

# default style is now conversational_companion
tts3x adapt /path/file.md --format explained --explained-profile conversational_companion

# use custom prompt library file
tts3x adapt /path/file.md --format explained --explained-prompts-file /path/explained_prompts.json --explained-profile novel_literary

# pronunciation catches for speakability (names/terms)
tts3x adapt /path/file.md --format explained \
  --pronunciation-file /Users/ileswade/ris/projects/tts3-extended/config/pronunciation.example.json

# 3) Synthesize
# base file defaults to all formats if --format not provided
tts3x synth /path/file.md --voice nova

# derived file only synthesizes that file
tts3x synth /path/file-speakable.md --voice voices.json

# speaker voice mapping (CLI)
tts3x synth /path/file.md --voice "Jennifer:orca,Quote:fable"

# 4) Captions (local default)
tts3x captions /path/file-speakable.mp3 --local --keep-wav --on-device

# with enrichment metadata from markdown + synth sidecar
tts3x captions /path/file-speakable.mp3 --md /path/file-speakable.md --cooldown 15

# cloud captions (explicit)
tts3x captions /path/file-speakable.mp3 --cloud --model whisper-1

# verify timing coverage (fails if below threshold)
tts3x captions /path/file-speakable.mp3 --local --verify --verify-threshold 85

# also fail if first word starts too late
tts3x captions /path/file-speakable.mp3 --local --verify --verify-threshold 85 --verify-max-start-seconds 10

# hybrid start gate (absolute + duration ratio, whichever is larger)
tts3x captions /path/file-speakable.mp3 --local --verify \
  --verify-threshold 85 \
  --verify-max-start-seconds 10 \
  --verify-max-start-ratio 0.02

# 5) Read in terminal with focus-word controls (+/- expands context)
tts3x read /path/file-speakable.mp3 --seconds 0 --window 8
```

## Output files (per format)
- `*.mp3`
- `*.words.json` (word timings)
- `*.vtt`
- `*.synth.json` (exact spoken plan from synth)
- `*.json` (enriched timeline for app playback UI)

## Current status
- Implemented: baseline import/adapt/synth/captions + sidecar + enriched timeline JSON.
- Implemented: native local Apple Speech captions path in tts3x (`--local`).
- OCR modes are wired (`--ocr auto|force|never`), but `force` is not yet implemented in this pass.
- Implemented: AI-driven `explained` generation using paragraph context window (`--explained-window`).
- Implemented: Speakable v2 deterministic transform:
  - bullet lists -> narrated sequence (First/Second/.../Finally)
  - table blocks -> placeholder line
  - blockquotes -> `[Quote]: ...` for alternate quote voice routing
  - URL/image metadata preserved via tags for downstream timeline enrichment
- Next: robust OCR backend, stronger paragraph/token alignment tests, full CloudKit upload tooling.

## Progress + Verbosity
- `adapt`, `synth`, and `captions` now show Rich progress bars by default.
- Detailed runtime info (format/chunks/chars/voices/estimated cost) is shown by default.
- Use `--quiet` to suppress rich/progress output and print minimal result paths.

## Programmatic API (manager/MCP-ready)
```python
from tts3x import TTS3X

t = TTS3X()
t.import_file("/path/doc.pdf", ocr="auto")
t.adapt("/path/doc.md", format="speakable,explained,summary")
t.adapt("/path/doc.md", format="explained", explained_profile="whitepaper_education")
t.synth("/path/doc.md", voice="nova")
t.captions("/path/doc-speakable.mp3", local=True, md_path="/path/doc-speakable.md", cooldown=15)
```

## Explained Prompt Profiles
- Built-in profiles:
  - `conversational_companion` (default)
  - `academic_default`
  - `whitepaper_education`
  - `novel_literary`
- Custom profile storage (auto-loaded if present):
  - `$HOME/.config/tts3x/explained_prompts.json`
  - `<cwd>/.tts3x/explained_prompts.json`
- Explicit file override:
  - `--explained-prompts-file /path/to/explained_prompts.json`
  - or env var: `TTS3X_PROMPTS_FILE=/path/to/explained_prompts.json`
- Example schema file:
  - `/Users/ileswade/ris/projects/tts3-extended/config/explained_prompts.example.json`

## Pronunciation Catches
- Purpose: force better spoken forms for names/terms (e.g., `Iles -> eye-ulls`).
- Auto-loaded locations (if present):
  - `$HOME/.config/tts3x/pronunciation.json`
  - `<cwd>/.tts3x/pronunciation.json`
- Explicit override:
  - `--pronunciation-file /path/pronunciation.json`
  - or env var: `TTS3X_PRONUN_FILE=/path/pronunciation.json`
- Example:
  - `/Users/ileswade/ris/projects/tts3-extended/config/pronunciation.example.json`

# tts3x
