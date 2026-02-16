from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VALID_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
VALID_FORMATS = {"literal", "speakable", "explained", "ted-talk", "summary"}
DEFAULT_EXPLAINED_PROFILE = "conversational_companion"
DEFAULT_QUOTE_VOICE = "fable"
VOICE_ALIASES = {
    "orca": "onyx",
}

DEFAULT_EXPLAINED_PROMPTS: Dict[str, Dict[str, Any]] = {
    "conversational_companion": {
        "description": "Casual conversational walkthrough in plain language.",
        "title": "AI Walkthrough",
        "system_prompt": (
            "You are talking directly to the listener in a natural, conversational voice. "
            "Explain one paragraph at a time in your own words, grounded only in the supplied text/context. "
            "Do not sound like a formal teacher or lecture script."
        ),
        "task": (
            "Give a casual explanation of what this paragraph is trying to say, "
            "why it matters in the flow of the paper, and how it connects to nearby paragraphs."
        ),
        "constraints": [
            "Use plain, human language.",
            "No external facts or hallucinated claims.",
            "No repetitive openings like 'in this section...'.",
            "No bullet list; one flowing paragraph.",
            "Paraphrase; do not copy long phrases from the source.",
        ],
    },
    "academic_default": {
        "description": "Balanced instructional explainer for academic/course readings.",
        "title": "Instructor Walkthrough",
        "system_prompt": (
            "You are an instructional assistant explaining one paragraph from an academic reading. "
            "Use only the supplied text and nearby context. Do not add external facts. "
            "Do not repeat section/chapter titles verbatim unless essential. "
            "Write one concise spoken paragraph that sounds natural to listen to."
        ),
        "task": "Explain the current paragraph for audio learners.",
        "constraints": [
            "Keep fidelity to author intent.",
            "Avoid repeating the same opening phrasing between consecutive outputs.",
            "No bullets; one paragraph only.",
            "No meta phrases like 'in this section the author says'.",
        ],
    },
    "whitepaper_education": {
        "description": "Explainer style for professional/whitepaper educational content.",
        "title": "Professional Walkthrough",
        "system_prompt": (
            "You are facilitating a professional briefing on an education-focused white paper. "
            "Explain what each paragraph is doing and why it matters, using only provided text/context."
        ),
        "task": "Explain this paragraph to professionals already familiar with the domain.",
        "constraints": [
            "Preserve nuance and terminology from the source.",
            "Point out argument structure and implications when present.",
            "No external references or added claims.",
            "Output exactly one spoken paragraph.",
        ],
    },
    "novel_literary": {
        "description": "Narrative-oriented explainer for literary/novel style content.",
        "title": "Narrative Companion",
        "system_prompt": (
            "You are a literary guide helping a listener understand tone, motivation, and progression "
            "in a narrative passage, using only the supplied passage and context."
        ),
        "task": "Explain the current paragraph in a natural narrative companion voice.",
        "constraints": [
            "Do not spoil beyond supplied context.",
            "Do not invent plot or character details not present.",
            "Keep language vivid but concise.",
            "Output exactly one paragraph.",
        ],
    },
    "ted_talk": {
        "description": "Narrative TED-talk style walkthrough that covers the whole document.",
        "title": "Ted Talk",
        "system_prompt": (
            "You are writing spoken TED-talk style script segments. "
            "Keep it engaging, clear, and authentic while staying faithful to the source text only. "
            "No external facts. No hallucinations."
        ),
        "task": (
            "Explain this part of the document in a TED-talk style voice, connecting ideas smoothly, "
            "and preserving the author's core intent."
        ),
        "constraints": [
            "Conversational and story-like, but accurate to source.",
            "No bullet lists.",
            "Avoid repetitive openings.",
            "Do not invent facts or references.",
            "One spoken paragraph output.",
        ],
    },
}

try:
    import AVFoundation  # type: ignore
    import Speech  # type: ignore
    from Foundation import NSDate, NSRunLoop, NSURL  # type: ignore

    APPLE_SPEECH_AVAILABLE = True
except Exception:
    AVFoundation = None
    Speech = None
    NSDate = None
    NSRunLoop = None
    NSURL = None
    APPLE_SPEECH_AVAILABLE = False


@dataclass
class HeaderState:
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "mac_roman", "cp1252"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or "")]


def _split_text(text: str, max_chars: int = 3800) -> List[str]:
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return [t] if t else []

    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in re.split(r"(?<=[.!?])\s+", t):
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            words = s.split()
            chunk: List[str] = []
            wlen = 0
            for w in words:
                add = len(w) + (1 if chunk else 0)
                if wlen + add > max_chars and chunk:
                    out.append(" ".join(chunk))
                    chunk = [w]
                    wlen = len(w)
                else:
                    chunk.append(w)
                    wlen += add
            if chunk:
                out.append(" ".join(chunk))
            continue

        add = len(s) + (1 if cur else 0)
        if cur_len + add > max_chars and cur:
            out.append(" ".join(cur))
            cur = [s]
            cur_len = len(s)
        else:
            cur.append(s)
            cur_len += add
    if cur:
        out.append(" ".join(cur))
    return out


def _sanitize_urls(text: str, policy: str, placeholder: str) -> str:
    url_re = r"https?://[^\s\]\)\}<>]+"
    if policy == "keep":
        return text
    if policy == "placeholder":
        return re.sub(url_re, placeholder, text)
    return re.sub(url_re, "", text)


def _extract_md_links(line: str) -> Tuple[str, List[str]]:
    urls: List[str] = []
    s = line or ""

    # Metadata tags emitted by adapt speakable: <!--URL:https://...-->
    for m in re.finditer(r"<!--\s*URL:([^>]+?)\s*-->", s):
        urls.append(m.group(1).strip())
    s = re.sub(r"<!--\s*URL:[^>]+?\s*-->", "", s)

    def _repl(m: re.Match[str]) -> str:
        label = m.group(1)
        url = m.group(2)
        urls.append(url)
        return label

    s = re.sub(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", _repl, s)
    for m in re.finditer(r"https?://[^\s\]\)\}<>]+", s):
        urls.append(m.group(0))
    dedup = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return s, dedup


def _extract_md_images(line: str) -> Tuple[str, List[str]]:
    imgs: List[str] = []
    s = line or ""

    # Metadata tags emitted by adapt speakable: <!--IMG:path_or_url-->
    for m in re.finditer(r"<!--\s*IMG:([^>]+?)\s*-->", s):
        imgs.append(m.group(1).strip())
    s = re.sub(r"<!--\s*IMG:[^>]+?\s*-->", "", s)

    def _repl(m: re.Match[str]) -> str:
        alt = (m.group(1) or "").strip()
        src = (m.group(2) or "").strip()
        if src:
            imgs.append(src)
        return alt

    s = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", _repl, s)
    return s, imgs


def _strip_inline_markdown(text: str) -> str:
    t = text or ""
    # Inline code, bold, italics -> keep text only.
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", t)
    return " ".join(t.split())


def _ordinal_intro(idx: int, total: int) -> str:
    if total <= 1:
        return ""
    if idx == total:
        return "Finally, "
    mapping = {
        1: "First, ",
        2: "Second, ",
        3: "Third, ",
        4: "Fourth, ",
        5: "Fifth, ",
    }
    return mapping.get(idx, "Next, ")


def _ensure_sentence(text: str) -> str:
    t = " ".join((text or "").split()).strip()
    if not t:
        return ""
    if t[-1] in ".!?":
        return t
    return t + "."


def _build_speakable_markdown_v2(
    source_markdown: str,
    *,
    url_mode: str,
    url_mention: str,
    img_mode: str,
    img_mention: str,
) -> str:
    """
    Speakable v2:
    - Preserve #/##/### headers (metadata anchors; typically not spoken at synth time).
    - Convert bullet lists to narrated sequence (First/Second/.../Finally).
    - Replace tables/code blocks with spoken placeholders.
    - Convert blockquotes to [Quote]: lines so synth can use alternate quote voice.
    - Preserve URL/image metadata via HTML comments: <!--URL:...-->, <!--IMG:...-->
    """
    lines = (source_markdown or "").splitlines()
    out: List[str] = []
    i = 0
    in_code = False

    url_policy = {"none": "drop", "mention": "placeholder", "full": "keep"}.get((url_mode or "mention").lower(), "placeholder")

    while i < len(lines):
        raw = lines[i]
        s = raw.rstrip()
        stripped = s.strip()

        # Fenced code blocks.
        if stripped.startswith("```"):
            in_code = not in_code
            if in_code:
                out.append("A code example is provided.")
                out.append("")
            i += 1
            continue
        if in_code:
            i += 1
            continue

        # Preserve headers for metadata anchoring.
        hm = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if hm:
            out.append(stripped)
            out.append("")
            i += 1
            continue

        # Blank line.
        if not stripped:
            out.append("")
            i += 1
            continue

        # Markdown table block detection.
        if "|" in stripped:
            j = i
            table_lines: List[str] = []
            while j < len(lines):
                t = lines[j].strip()
                if not t:
                    break
                if "|" not in t:
                    break
                table_lines.append(t)
                j += 1
            if len(table_lines) >= 2 and any(re.match(r"^\|?[\s:\-|\+]+\|?$", tl) for tl in table_lines[1:2]):
                out.append("A table is provided for this section.")
                out.append("")
                i = j
                continue

        # Blockquote block -> quote speaker line(s).
        if stripped.startswith(">"):
            q_lines: List[str] = []
            j = i
            while j < len(lines):
                t = lines[j].strip()
                if not t.startswith(">"):
                    break
                q_lines.append(re.sub(r"^\s*>\s?", "", t))
                j += 1
            q_text = _ensure_sentence(_strip_inline_markdown(" ".join(q_lines)))
            if q_text:
                out.append(f"[Quote]: {q_text}")
                out.append("")
            i = j
            continue

        # Bullet list block.
        bm = re.match(r"^\s*(?:[-\*\+]|\d+\.)\s+(.+)$", s)
        if bm:
            items: List[str] = []
            j = i
            while j < len(lines):
                bline = lines[j].rstrip()
                m = re.match(r"^\s*(?:[-\*\+]|\d+\.)\s+(.+)$", bline)
                if not m:
                    break
                item = m.group(1).strip()

                # Extract and preserve image/url metadata.
                no_img, imgs = _extract_md_images(item)
                for src in imgs:
                    no_img += f" <!--IMG:{src}-->"
                no_link, urls = _extract_md_links(no_img)
                for u in urls:
                    no_link += f" <!--URL:{u}-->"

                if img_mode == "mention" and imgs:
                    no_link = (no_link + " " + img_mention).strip()
                no_link = _sanitize_urls(no_link, url_policy, url_mention)
                no_link = _strip_inline_markdown(no_link)
                items.append(_ensure_sentence(no_link))
                j += 1

            total = len(items)
            if total:
                narrated_parts: List[str] = []
                for idx, it in enumerate(items, start=1):
                    narrated_parts.append(_ordinal_intro(idx, total) + it)
                out.append(" ".join([x for x in narrated_parts if x.strip()]).strip())
                out.append("")
            i = j
            continue

        # Regular prose line.
        no_img, imgs = _extract_md_images(stripped)
        prose = no_img
        for src in imgs:
            prose += f" <!--IMG:{src}-->"
        no_link, urls = _extract_md_links(prose)
        prose = no_link
        for u in urls:
            prose += f" <!--URL:{u}-->"
        if img_mode == "mention" and imgs:
            prose = (prose + " " + img_mention).strip()
        prose = _sanitize_urls(prose, url_policy, url_mention)
        prose = _strip_inline_markdown(prose)
        prose = _ensure_sentence(prose)
        if prose:
            out.append(prose)
        i += 1

    # Normalize extra blank lines.
    final_lines: List[str] = []
    blank = False
    for ln in out:
        if ln.strip():
            final_lines.append(ln.rstrip())
            blank = False
        else:
            if not blank:
                final_lines.append("")
            blank = True
    return "\n".join(final_lines).strip() + "\n"


def _load_explained_prompt_profiles(prompt_file: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = json.loads(json.dumps(DEFAULT_EXPLAINED_PROMPTS))
    candidates: List[Path] = []

    env_file = os.getenv("TTS3X_PROMPTS_FILE")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    if prompt_file:
        candidates.append(prompt_file.expanduser())
    candidates.append(Path.home() / ".config" / "tts3x" / "explained_prompts.json")
    candidates.append(Path.cwd() / ".tts3x" / "explained_prompts.json")

    seen: set[str] = set()
    for p in candidates:
        k = str(p.resolve()) if p.exists() else str(p)
        if k in seen:
            continue
        seen.add(k)
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for name, cfg in payload.items():
            if isinstance(cfg, dict):
                profiles[str(name).strip()] = cfg
    return profiles


def _load_pronunciation_map(pronunciation_file: Optional[Path] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    candidates: List[Path] = []
    env_file = os.getenv("TTS3X_PRONUN_FILE")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    if pronunciation_file:
        candidates.append(pronunciation_file.expanduser())
    candidates.append(Path.home() / ".config" / "tts3x" / "pronunciation.json")
    candidates.append(Path.cwd() / ".tts3x" / "pronunciation.json")

    seen: set[str] = set()
    for p in candidates:
        k = str(p.resolve()) if p.exists() else str(p)
        if k in seen:
            continue
        seen.add(k)
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for src, dst in payload.items():
            s = str(src).strip()
            d = str(dst).strip()
            if s and d:
                out[s] = d
    return out


def _apply_pronunciation_map(text: str, pronunciation_map: Dict[str, str]) -> str:
    t = text or ""
    if not pronunciation_map:
        return t
    # Replace longer keys first to avoid partial overlaps.
    for src in sorted(pronunciation_map.keys(), key=len, reverse=True):
        dst = pronunciation_map[src]
        t = re.sub(re.escape(src), dst, t, flags=re.IGNORECASE)
    return t


def list_explained_prompt_profiles(prompt_file: Optional[Path] = None) -> Dict[str, str]:
    profiles = _load_explained_prompt_profiles(prompt_file=prompt_file)
    out: Dict[str, str] = {}
    for name, cfg in profiles.items():
        out[name] = str(cfg.get("description", ""))
    return out


def _parse_markdown_for_explain(source_markdown: str) -> List[Dict[str, Any]]:
    """
    Parse markdown into explainable units while preserving header context.
    Headers are state markers; prose/bullets become paragraph units.
    """
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None
    units: List[Dict[str, Any]] = []
    cur: List[str] = []

    def is_plain_heading(line: str) -> bool:
        # Heuristic for non-markdown section titles like "Search History" / "Conclusion".
        t = line.strip()
        if not t:
            return False
        # Colon-led titles are common in imported academic docs.
        if ":" in t:
            words_colon = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&:/-]*", t)
            if 1 <= len(words_colon) <= 30:
                return True
        if len(t) > 140:
            return False
        if re.search(r"[.!?]$", t):
            return False
        # Must look like a title, not full prose sentence.
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&:/-]*", t)
        if not words or len(words) > 14:
            return False
        cap_like = sum(1 for w in words if w[:1].isupper() or w.isupper())
        if cap_like < max(1, int(len(words) * 0.6)):
            return False
        # If it has a colon and is short-ish, it is likely a heading.
        if ":" in t and len(words) <= 20:
            return True
        # Common title patterns (single/multi-word title case).
        return True

    def flush(kind: str = "paragraph") -> None:
        nonlocal cur
        if not cur:
            return
        raw = " ".join([x.strip() for x in cur if x.strip()]).strip()
        cur = []
        if not raw:
            return
        units.append(
            {
                "kind": kind,
                "text": raw,
                "h1": h1,
                "h2": h2,
                "h3": h3,
            }
        )

    for line in (source_markdown or "").splitlines():
        s = line.rstrip("\n")
        if not s.strip():
            flush("paragraph")
            continue

        stripped = s.strip()
        hm = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if hm:
            flush("paragraph")
            lvl = len(hm.group(1))
            title = hm.group(2).strip()
            if lvl == 1:
                h1, h2, h3 = title, None, None
            elif lvl == 2:
                h2, h3 = title, None
            else:
                h3 = title
            continue

        bm = re.match(r"^\s*(?:[-\*\+]|\d+\.)\s+(.+)$", s)
        if bm:
            flush("paragraph")
            units.append(
                {
                    "kind": "bullet",
                    "text": bm.group(1).strip(),
                    "h1": h1,
                    "h2": h2,
                    "h3": h3,
                }
            )
            continue

        # Plain-text heading support when source markdown came from import and lacks '#' headers.
        if is_plain_heading(stripped):
            flush("paragraph")
            # Treat plain headings as level-2 section markers by default.
            h2, h3 = stripped, None
            continue

        # If we already have paragraph text and this line appears to start a new sentence block,
        # split to avoid collapsing a whole document into one giant unit.
        if cur:
            prev = cur[-1].rstrip()
            if re.search(r"[.!?:]$", prev) and (s.startswith("  ") or s.startswith("\t")):
                flush("paragraph")

        cur.append(stripped)

    flush("paragraph")
    for i, u in enumerate(units, start=1):
        u["id"] = f"p_{i:04d}"
    return units


def _looks_like_ted_talk_markdown(text: str) -> bool:
    for line in (text or "").splitlines():
        t = line.strip()
        if not t:
            continue
        if re.match(r"^#\s*ted\s*talk\b", t, flags=re.IGNORECASE):
            return True
        if re.match(r"^ted\s*talk\b", t, flags=re.IGNORECASE):
            return True
        return False
    return False


def _build_explained_markdown_ai(
    source_markdown: str,
    *,
    model: str,
    back: int,
    forward: int,
    prompt_profile: str,
    prompt_file: Optional[Path],
    persona_name: str,
    pronunciation_map: Optional[Dict[str, str]] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    info_cb: Optional[Callable[[str], None]] = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for explained format generation.")

    units = _parse_markdown_for_explain(source_markdown)
    if not units:
        return "# Instructor Walkthrough\n\n(No explainable content found.)\n"

    profiles = _load_explained_prompt_profiles(prompt_file=prompt_file)
    selected = (prompt_profile or DEFAULT_EXPLAINED_PROFILE).strip()
    if selected not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown explained prompt profile '{selected}'. Available: {available}")
    cfg = profiles[selected]
    title = str(cfg.get("title", "Instructor Walkthrough")).strip() or "Instructor Walkthrough"
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    task_text = str(cfg.get("task", "Explain the current paragraph for audio learners.")).strip()
    constraints = cfg.get("constraints") or []
    if not isinstance(constraints, list):
        constraints = [str(constraints)]
    constraints = [str(x).strip() for x in constraints if str(x).strip()]

    client = OpenAI(api_key=api_key)
    out_lines: List[str] = [f"# {title}", ""]
    prev_headers = HeaderState()
    total_units = len(units)
    if info_cb:
        info_cb(
            f"explained profile={selected} model={model} back={back} forward={forward} "
            f"listener={persona_name} paragraphs={total_units}"
        )

    for idx, u in enumerate(units):
        h1, h2, h3 = u.get("h1"), u.get("h2"), u.get("h3")
        if h1 and h1 != prev_headers.h1:
            out_lines.append(f"## {h1}")
            out_lines.append("")
        if h2 and h2 != prev_headers.h2:
            out_lines.append(f"### {h2}")
            out_lines.append("")
        if h3 and h3 != prev_headers.h3:
            out_lines.append(f"#### {h3}")
            out_lines.append("")
        prev_headers = HeaderState(h1=h1, h2=h2, h3=h3)

        left = max(0, idx - max(0, int(back)))
        right = min(len(units), idx + max(0, int(forward)) + 1)
        prev_ctx = [x["text"] for x in units[left:idx]]
        next_ctx = [x["text"] for x in units[idx + 1:right]]
        cur_text = u["text"]

        if pronunciation_map:
            prev_ctx = [_apply_pronunciation_map(x, pronunciation_map) for x in prev_ctx]
            next_ctx = [_apply_pronunciation_map(x, pronunciation_map) for x in next_ctx]
            cur_text = _apply_pronunciation_map(cur_text, pronunciation_map)

        user_prompt = {
            "task": task_text,
            "prompt_profile": selected,
            "listener_name": persona_name,
            "current_paragraph_id": u["id"],
            "section_context": {"h1": h1, "h2": h2, "h3": h3},
            "previous_paragraphs": prev_ctx,
            "current_paragraph": cur_text,
            "next_paragraphs": next_ctx,
            "constraints": constraints,
        }

        resp = client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        )
        explanation = (resp.choices[0].message.content or "").strip()
        if not explanation:
            explanation = cur_text
        if pronunciation_map:
            explanation = _apply_pronunciation_map(explanation, pronunciation_map)
        explanation = " ".join(explanation.split())

        out_lines.append(f"<!--SOURCE:{u['id']}-->")
        out_lines.append(explanation)
        out_lines.append("")
        if progress_cb:
            progress_cb("explained", idx + 1, total_units, f"Explained {idx+1}/{total_units}")

    return "\n".join(out_lines).rstrip() + "\n"


def _build_literal_markdown_parsed(source_markdown: str) -> str:
    units = _parse_markdown_for_explain(source_markdown)
    if not units:
        return source_markdown.strip() + "\n"

    out_lines: List[str] = []
    prev_headers = HeaderState()
    for u in units:
        h1, h2, h3 = u.get("h1"), u.get("h2"), u.get("h3")
        if h1 and h1 != prev_headers.h1:
            out_lines.append(f"# {h1}")
            out_lines.append("")
        if h2 and h2 != prev_headers.h2:
            out_lines.append(f"## {h2}")
            out_lines.append("")
        if h3 and h3 != prev_headers.h3:
            out_lines.append(f"### {h3}")
            out_lines.append("")
        prev_headers = HeaderState(h1=h1, h2=h2, h3=h3)

        out_lines.append(u["text"].strip())
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


def _parse_voice_spec(voice: Optional[str]) -> Tuple[Optional[str], Dict[str, str], Optional[Path]]:
    if not voice:
        return None, {}, None
    v = voice.strip()
    p = Path(v).expanduser()
    if v.lower().endswith(".json") and p.exists():
        return None, {}, p
    if ":" in v or "," in v:
        mapping: Dict[str, str] = {}
        for part in [x.strip() for x in v.split(",") if x.strip()]:
            if ":" not in part:
                continue
            name, vv = part.split(":", 1)
            mapping[name.strip().lower()] = vv.strip()
        return None, mapping, None
    return v, {}, None


def _load_voice_map(voice: Optional[str], voices_file: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
    default_voice, inline_map, inline_file = _parse_voice_spec(voice)
    vf = Path(voices_file).expanduser() if voices_file else inline_file
    mapping = dict(inline_map)
    if vf:
        payload = json.loads(vf.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for k, v in payload.items():
                mapping[str(k).strip().lower()] = str(v).strip()
    return default_voice, mapping


def _normalize_voice_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    n = str(name).strip().lower()
    if n in VALID_VOICES:
        return n
    return VOICE_ALIASES.get(n, n)


def detect_derived(path: Path) -> Optional[str]:
    n = path.name.lower()
    if n.endswith("-speakable.md"):
        return "speakable"
    if n.endswith("-explained.md"):
        return "explained"
    if n.endswith("-ted-talk.md") or n.endswith("-tedtalk.md"):
        return "ted-talk"
    if n.endswith("-summary.md"):
        return "summary"
    return None


def parse_formats(spec: Optional[str]) -> List[str]:
    s = (spec or "all").strip().lower()
    aliases = {
        "a": "all",
        "l": "literal",
        "s": "speakable",
        "e": "explained",
        "t": "ted-talk",
        "m": "summary",
    }
    s = aliases.get(s, s)
    if s == "all":
        return ["literal", "speakable", "explained", "ted-talk", "summary"]
    parts = [aliases.get(x.strip().lower(), x.strip().lower()) for x in s.split(",") if x.strip()]
    out = [p for p in parts if p in VALID_FORMATS]
    return out or ["literal"]


def _strip_variant_suffix(stem: str) -> str:
    s = stem
    for suf in ("-literal", "-speakable", "-explained", "-explain", "-ted-talk", "-tedtalk", "-summary", "-instructions"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def _document_root_stem(path: Path) -> str:
    return _strip_variant_suffix(path.stem)


def _artifact_work_dir_for_path(path: Path) -> Path:
    root = _document_root_stem(path)
    if path.parent.name == root:
        return path.parent
    return path.with_name(root)


def _project_root_dir_for_path(path: Path) -> Path:
    work_dir = _artifact_work_dir_for_path(path)
    if path.parent == work_dir:
        return work_dir.parent
    return path.parent


def _mpx_path_for_path(path: Path) -> Path:
    root = _document_root_stem(path)
    return _project_root_dir_for_path(path) / f"{root}.mpx"


def _update_mpx_for_path(path: Path, *, source_md: Optional[Path] = None) -> Path:
    root = _document_root_stem(path)
    work_dir = _artifact_work_dir_for_path(path)
    work_dir.mkdir(parents=True, exist_ok=True)
    mpx_path = _mpx_path_for_path(path)
    project_root = _project_root_dir_for_path(path)

    mp3_files = sorted([p for p in work_dir.glob("*.mp3") if p.is_file()])
    mp4_files = sorted([p for p in work_dir.glob("*.mp4") if p.is_file()])
    words_files = sorted([p for p in work_dir.glob("*.words.json") if p.is_file()])
    vtt_files = sorted([p for p in work_dir.glob("*.vtt") if p.is_file()])
    synth_files = sorted([p for p in work_dir.glob("*.synth.json") if p.is_file()])
    adapted_md = sorted([p for p in work_dir.glob("*.md") if p.is_file()])
    enriched_json = sorted(
        [p for p in work_dir.glob("*.json") if p.is_file() and not p.name.endswith(".words.json") and not p.name.endswith(".synth.json")]
    )

    payload: Dict[str, Any] = {
        "schema": "tts3.project.mpx.v1",
        "project_id": root,
        "project_dir": str(work_dir),
        "project_root": str(project_root),
        "source_md": str(source_md) if source_md else str(project_root / f"{root}.md"),
        "artifacts": {
            "mp3": [str(p) for p in mp3_files],
            "mp4": [str(p) for p in mp4_files],
            "words_json": [str(p) for p in words_files],
            "vtt": [str(p) for p in vtt_files],
            "synth_json": [str(p) for p in synth_files],
            "adapted_md": [str(p) for p in adapted_md],
            "enriched_json": [str(p) for p in enriched_json],
        },
    }
    mpx_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mpx_path


def _artifact_base_for_audio(audio_path: Path) -> Path:
    work_dir = _artifact_work_dir_for_path(audio_path)
    return work_dir / audio_path.with_suffix("").name


def resolve_variant_path(base: Path, fmt: str) -> Path:
    root = _document_root_stem(base)
    work_dir = _artifact_work_dir_for_path(base)
    legacy_stem = base.with_suffix("")
    if fmt == "ted-talk" and base.exists() and base.suffix.lower() == ".md":
        try:
            if _looks_like_ted_talk_markdown(_read_text(base)):
                return base
        except Exception:
            pass
    if fmt == "literal":
        candidates = [
            work_dir / f"{root}-literal.md",
            base,
            Path(str(legacy_stem) + "-literal.md"),
        ]
    elif fmt == "ted-talk":
        candidates = [
            work_dir / f"{root}-ted-talk.md",
            work_dir / f"{root}-tedtalk.md",
            Path(str(legacy_stem) + "-ted-talk.md"),
            Path(str(legacy_stem) + "-tedtalk.md"),
        ]
    else:
        candidates = [
            work_dir / f"{root}-{fmt}.md",
            Path(str(legacy_stem) + f"-{fmt}.md"),
        ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def read_headers_filter(level: str, line: str) -> Optional[str]:
    m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not m:
        return line
    wanted = (level or "none").strip().lower()
    if wanted in {"none", "0", "no"}:
        return None
    try:
        max_level = int(wanted)
    except Exception:
        max_level = 0
    hdr_level = len(m.group(1))
    text = m.group(2).strip()
    if hdr_level <= max_level and text:
        return text + "."
    return None


def parse_markdown_for_synth(
    text: str,
    *,
    read_headers: str,
    url_policy: str,
    url_placeholder: str,
    img_policy: str,
    img_placeholder: str,
) -> List[Dict[str, Any]]:
    state = HeaderState()
    paragraphs: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_kind = "paragraph"

    def flush() -> None:
        nonlocal cur, cur_kind
        if not cur:
            return
        raw = " ".join([x.strip() for x in cur if x.strip()]).strip()
        cur = []
        if not raw:
            return

        s, imgs = _extract_md_images(raw)
        s, urls = _extract_md_links(s)

        if img_policy == "mention":
            s = s + (" " + img_placeholder if imgs else "")
        s = _sanitize_urls(s, "placeholder" if url_policy == "mention" else url_policy, url_placeholder)
        if img_policy == "none":
            s = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", s)

        spoken = " ".join(s.split())
        if not spoken:
            return

        paragraphs.append(
            {
                "id": f"p_{len(paragraphs)+1:04d}",
                "kind": cur_kind,
                "spoken_text": spoken,
                "tokens": _tokenize(spoken),
                "h1": state.h1,
                "h2": state.h2,
                "h3": state.h3,
                "urls": urls,
                "images": imgs,
            }
        )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            cur_kind = "paragraph"
            continue

        hm = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if hm:
            flush()
            lvl = len(hm.group(1))
            title = hm.group(2).strip()
            if lvl == 1:
                state.h1, state.h2, state.h3 = title, None, None
            elif lvl == 2:
                state.h2, state.h3 = title, None
            else:
                state.h3 = title
            hline = read_headers_filter(read_headers, line)
            if hline:
                cur = [hline]
                cur_kind = "header"
                flush()
            continue

        bm = re.match(r"^\s*(?:[-\*\+]|\d+\.)\s+(.+)$", line)
        if bm:
            flush()
            cur_kind = "bullet"
            cur = [bm.group(1).strip()]
            flush()
            cur_kind = "paragraph"
            continue

        cur.append(line.strip())

    flush()
    return paragraphs


def _combine_mp3s(parts: List[Path], output_path: Path) -> None:
    if not parts:
        raise RuntimeError("No audio chunks were generated")
    if len(parts) == 1:
        shutil.copyfile(parts[0], output_path)
        return

    concat_file = output_path.parent / f"{output_path.stem}.concat.txt"
    concat_file.write_text("\n".join([f"file '{p.as_posix()}'" for p in parts]), encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        concat_file.unlink(missing_ok=True)


def _to_wav_16k_mono(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _audio_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    try:
        return float((p.stdout or "").strip())
    except Exception:
        return 0.0


def _timing_coverage(payload: Dict[str, Any], *, audio_duration: float) -> Dict[str, float]:
    words = _payload_to_words(payload)
    if not words or audio_duration <= 0:
        return {"coverage": 0.0, "first_start": 0.0, "last_end": 0.0}

    first_start = float(words[0].get("start", 0.0))
    last_end = float(words[-1].get("end", first_start))
    if last_end < first_start:
        last_end = first_start
    coverage = max(0.0, min(1.0, (last_end - first_start) / max(audio_duration, 0.001)))
    return {"coverage": coverage, "first_start": first_start, "last_end": last_end}


def _extract_pdf_text_pypdf(input_path: Path) -> Tuple[str, Dict[str, Any]]:
    from pypdf import PdfReader  # optional dependency

    reader = PdfReader(str(input_path))
    page_texts: List[str] = []
    empty_pages = 0
    for p in reader.pages:
        t = (p.extract_text() or "").strip()
        if not t:
            empty_pages += 1
        page_texts.append(t)
    text = "\n\n".join(page_texts)
    meta = {
        "pages": len(page_texts),
        "empty_pages": empty_pages,
        "chars": len(text),
    }
    return text, meta


def import_to_md(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    ocr: str = "auto",
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    ext = input_path.suffix.lower()
    output = output_path or input_path.with_suffix(".md")
    ocr_mode = (ocr or "auto").strip().lower()
    if ocr_mode not in {"auto", "force", "never"}:
        raise ValueError("ocr must be one of: auto, force, never")
    import_meta: Dict[str, Any] = {"ocr_mode": ocr_mode}

    if ext in {".md", ".txt"}:
        text = _read_text(input_path)
        import_meta["source_type"] = ext.lstrip(".")
    elif ext == ".pdf":
        text, pdf_meta = _extract_pdf_text_pypdf(input_path)
        import_meta["source_type"] = "pdf"
        import_meta.update(pdf_meta)
        likely_scanned = bool(pdf_meta.get("pages", 0) > 0 and pdf_meta.get("empty_pages", 0) >= max(1, int(pdf_meta.get("pages", 0) * 0.7)))
        import_meta["likely_scanned"] = likely_scanned

        if ocr_mode == "force":
            raise RuntimeError(
                "OCR mode 'force' is not implemented in this build yet. "
                "Use --ocr auto or --ocr never for now."
            )
        if ocr_mode == "auto" and likely_scanned:
            import_meta["ocr_notice"] = "PDF appears scan-heavy; OCR recommended in next build."
    elif ext in {".docx", ".doc"}:
        import docx2txt  # optional dependency

        text = docx2txt.process(str(input_path)) or ""
        import_meta["source_type"] = ext.lstrip(".")
    else:
        raise ValueError(f"Unsupported input type for import: {ext}")

    output.write_text(text, encoding="utf-8")
    import_meta["md"] = str(output)
    return import_meta


def adapt_markdown(
    input_md: Path,
    *,
    formats: List[str],
    summary: bool,
    instructions: bool,
    url_mode: str,
    url_mention: str,
    img_mode: str,
    img_mention: str,
    explained_model: str = "gpt-4o-mini",
    explained_window: int = 2,
    explained_back: Optional[int] = None,
    explained_forward: Optional[int] = None,
    explained_persona: str = "Iles",
    explained_profile: str = DEFAULT_EXPLAINED_PROFILE,
    explained_prompts_file: Optional[Path] = None,
    pronunciation_file: Optional[Path] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    info_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    text = _read_text(input_md)
    out: Dict[str, Any] = {}
    total_steps = 0
    total_steps += 1  # literal (paragraph parsed) is always produced by adapt
    total_steps += 1 if "speakable" in formats else 0
    total_steps += 1 if "explained" in formats else 0
    total_steps += 1 if "ted-talk" in formats else 0
    total_steps += 1 if ("summary" in formats and summary) else 0
    total_steps += 1 if instructions else 0
    step = 0
    root = _document_root_stem(input_md)
    work_dir = _artifact_work_dir_for_path(input_md)
    work_dir.mkdir(parents=True, exist_ok=True)

    def write_variant(suffix: str, content: str) -> None:
        path = work_dir / f"{root}-{suffix}.md"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        out[suffix] = str(path)

    # Always create a literal, paragraph-parsed version to normalize downstream processing.
    step += 1
    if info_cb:
        info_cb("adapt building literal (paragraph parsed)")
    literal_md = _build_literal_markdown_parsed(text)
    write_variant("literal", literal_md)
    if progress_cb:
        progress_cb("adapt", step, max(total_steps, 1), "Literal complete")

    if "speakable" in formats:
        step += 1
        if info_cb:
            info_cb("adapt building speakable")
        speakable = _build_speakable_markdown_v2(
            text,
            url_mode=url_mode,
            url_mention=url_mention,
            img_mode=img_mode,
            img_mention=img_mention,
        )
        write_variant("speakable", speakable)
        if progress_cb:
            progress_cb("adapt", step, max(total_steps, 1), "Speakable complete")

    if "explained" in formats:
        step += 1
        if info_cb:
            info_cb("adapt building explained")
        back = int(explained_back) if explained_back is not None else int(explained_window)
        forward = int(explained_forward) if explained_forward is not None else int(explained_window)
        pron = _load_pronunciation_map(pronunciation_file=pronunciation_file)
        explained_md = _build_explained_markdown_ai(
            source_markdown=literal_md,
            model=explained_model,
            back=max(0, back),
            forward=max(0, forward),
            prompt_profile=explained_profile,
            prompt_file=explained_prompts_file,
            persona_name=str(explained_persona or "Iles"),
            pronunciation_map=pron,
            progress_cb=progress_cb,
            info_cb=info_cb,
        )
        write_variant("explained", explained_md)
        if progress_cb:
            progress_cb("adapt", step, max(total_steps, 1), "Explained complete")

    if "ted-talk" in formats:
        step += 1
        if info_cb:
            info_cb("adapt building ted-talk")
        if _looks_like_ted_talk_markdown(text):
            ted_talk_md = text.strip() + "\n"
        else:
            back = int(explained_back) if explained_back is not None else int(explained_window)
            forward = int(explained_forward) if explained_forward is not None else int(explained_window)
            pron = _load_pronunciation_map(pronunciation_file=pronunciation_file)
            ted_talk_md = _build_explained_markdown_ai(
                source_markdown=literal_md,
                model=explained_model,
                back=max(0, back),
                forward=max(0, forward),
                prompt_profile="ted_talk",
                prompt_file=explained_prompts_file,
                persona_name=str(explained_persona or "Iles"),
                pronunciation_map=pron,
                progress_cb=progress_cb,
                info_cb=info_cb,
            )
            if not re.match(r"^\s*#\s*ted\s*talk\b", ted_talk_md, flags=re.IGNORECASE):
                ted_talk_md = "# Ted Talk\n\n" + ted_talk_md.lstrip()
        write_variant("ted-talk", ted_talk_md)
        if progress_cb:
            progress_cb("adapt", step, max(total_steps, 1), "Ted-talk complete")

    if "summary" in formats and summary:
        step += 1
        if info_cb:
            info_cb("adapt building summary")
        words = _tokenize(text)
        top = " ".join(words[:160])
        write_variant("summary", f"# Summary\n\n{top}")
        if progress_cb:
            progress_cb("adapt", step, max(total_steps, 1), "Summary complete")

    if instructions:
        step += 1
        inst_path = work_dir / f"{root}-instructions.md"
        inst_path.write_text(
            "# Listening Instructions\n\nUse section headers for orientation; links and images are metadata anchors.",
            encoding="utf-8",
        )
        out["instructions"] = str(inst_path)
        if progress_cb:
            progress_cb("adapt", step, max(total_steps, 1), "Instructions complete")

    out["mpx"] = str(_update_mpx_for_path(input_md, source_md=input_md))
    return out


def synth(
    input_path: Path,
    *,
    output_path: Optional[Path],
    format_spec: Optional[str],
    read_headers: str,
    voice: Optional[str],
    voices_file: Optional[str],
    model: str,
    url_policy: str,
    url_placeholder: str,
    img_policy: str,
    img_placeholder: str,
    emit_sidecar: bool,
    request_timeout_seconds: float = 90.0,
    max_retries: int = 3,
    multi_thread: int = 1,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    info_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    default_voice, voice_map = _load_voice_map(voice, voices_file)
    default_voice = _normalize_voice_name(default_voice or "nova")
    if default_voice not in VALID_VOICES:
        raise ValueError(f"Invalid default voice '{default_voice}'. Valid: {', '.join(sorted(VALID_VOICES))}")

    normalized_map: Dict[str, str] = {}
    for k, v in voice_map.items():
        vv = _normalize_voice_name(v)
        if vv not in VALID_VOICES:
            raise ValueError(f"Invalid mapped voice '{v}' for speaker '{k}'. Valid: {', '.join(sorted(VALID_VOICES))}")
        normalized_map[k] = vv
    voice_map = normalized_map

    derived = detect_derived(input_path)
    targets: List[Tuple[str, Path]] = []
    if derived:
        targets = [(derived, input_path)]
    else:
        for fmt in parse_formats(format_spec):
            p = resolve_variant_path(input_path, fmt)
            if p.exists():
                targets.append((fmt, p))

    if not targets:
        raise RuntimeError("No synth targets resolved")

    if output_path and output_path.suffix.lower() == ".mp3" and len(targets) > 1:
        raise ValueError("Single .mp3 output cannot be used for multiple formats")

    workers = max(1, int(multi_thread))
    results: List[Dict[str, Any]] = []
    if info_cb:
        if voice_map:
            vm = ", ".join([f"{k}:{v}" for k, v in sorted(voice_map.items())])
            info_cb(f"synth default_voice={default_voice} voice_map={vm}")
        else:
            info_cb(f"synth default_voice={default_voice}")
        info_cb(f"synth workers={workers}")

    for t_idx, (fmt, src) in enumerate(targets, start=1):
        if progress_cb:
            progress_cb("synth_targets", t_idx - 1, len(targets), f"Preparing {fmt}")
        text = _read_text(src)
        paragraphs = parse_markdown_for_synth(
            text,
            read_headers=read_headers,
            url_policy=url_policy,
            url_placeholder=url_placeholder,
            img_policy=img_policy,
            img_placeholder=img_placeholder,
        )

        # speaker labels [Name]:
        for p in paragraphs:
            m = re.match(r"^\[([^\]]+)\]\s*:\s*(.+)$", p["spoken_text"])
            if m:
                p["speaker"] = m.group(1).strip().lower()
                p["spoken_text"] = m.group(2).strip()
            else:
                p["speaker"] = None

        if output_path:
            if output_path.suffix.lower() == ".mp3":
                out_mp3 = output_path
            else:
                out_mp3 = Path(str(output_path) + ("" if fmt == "literal" else f"-{fmt}") + ".mp3")
        else:
            out_mp3 = src.with_suffix(".mp3")

        out_mp3.parent.mkdir(parents=True, exist_ok=True)

        chunks: List[Dict[str, Any]] = []
        temp_parts: List[Path] = []
        total_chars = 0

        for p in paragraphs:
            pv = voice_map.get((p.get("speaker") or "").lower(), default_voice)
            if (p.get("speaker") or "").lower() == "quote" and "quote" not in voice_map:
                pv = DEFAULT_QUOTE_VOICE
            if pv not in VALID_VOICES:
                pv = default_voice
            for part in _split_text(p["spoken_text"]):
                if not part.strip():
                    continue
                chunk_id = f"c_{len(chunks)+1:04d}"
                total_chars += len(part)
                chunks.append(
                    {
                        "id": chunk_id,
                        "paragraph_id": p["id"],
                        "voice": pv,
                        "text": part,
                        "token_count": len(_tokenize(part)),
                    }
                )
        if info_cb:
            info_cb(f"synth format={fmt} chunks={len(chunks)} chars={total_chars:,} source={src.name}")

        def _synth_one_chunk(c_idx: int, c: Dict[str, Any]) -> Path:
            tf = Path(tempfile.gettempdir()) / f"tts3x_{uuid.uuid4().hex}.mp3"
            last_err: Optional[Exception] = None
            local_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            for attempt in range(1, max(1, int(max_retries)) + 1):
                try:
                    resp = local_client.audio.speech.create(
                        model=model,
                        voice=c["voice"],
                        input=c["text"],
                        response_format="mp3",
                        timeout=float(request_timeout_seconds),
                    )
                    resp.write_to_file(str(tf))
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if info_cb:
                        info_cb(
                            f"synth retry format={fmt} chunk={c_idx}/{len(chunks)} "
                            f"attempt={attempt}/{max_retries} err={type(e).__name__}"
                        )
                    time.sleep(min(2.0 * attempt, 8.0))
            if last_err is not None:
                raise RuntimeError(
                    f"Failed to synthesize chunk {c_idx}/{len(chunks)} for format '{fmt}' "
                    f"after {max_retries} attempts: {last_err}"
                )
            return tf

        total_chunks = len(chunks)
        if workers == 1 or total_chunks <= 1:
            for c_idx, c in enumerate(chunks, start=1):
                if progress_cb:
                    progress_cb("synth_chunks", c_idx, max(total_chunks, 1), f"{fmt} chunk {c_idx}/{total_chunks} voice={c['voice']}")
                tf = _synth_one_chunk(c_idx, c)
                temp_parts.append(tf)
        else:
            temp_parts = [Path()] * total_chunks
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_idx = {
                    ex.submit(_synth_one_chunk, c_idx, c): c_idx
                    for c_idx, c in enumerate(chunks, start=1)
                }
                for fut in as_completed(fut_to_idx):
                    c_idx = fut_to_idx[fut]
                    tf = fut.result()
                    temp_parts[c_idx - 1] = tf
                    completed += 1
                    if progress_cb:
                        progress_cb(
                            "synth_chunks",
                            completed,
                            max(total_chunks, 1),
                            f"{fmt} chunk {c_idx}/{total_chunks} voice={chunks[c_idx-1]['voice']}",
                        )

        _combine_mp3s(temp_parts, out_mp3)
        for t in temp_parts:
            t.unlink(missing_ok=True)

        sidecar_dir = _artifact_work_dir_for_path(out_mp3)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / f"{out_mp3.with_suffix('').name}.synth.json"
        if emit_sidecar:
            sidecar = {
                "schema": "tts3.synth_plan.v1",
                "source_md": src.name,
                "source_sha256": _sha256(src),
                "audio_mp3": out_mp3.name,
                "audio_sha256": _sha256(out_mp3),
                "model": model,
                "default_voice": default_voice,
                "speaker_voice_map": voice_map,
                "policies": {
                    "read_headers": read_headers,
                    "url": {"policy": url_policy, "placeholder": url_placeholder},
                    "img": {"policy": img_policy, "placeholder": img_placeholder},
                },
                "chars_total": total_chars,
                "paragraphs": paragraphs,
                "chunks": chunks,
            }
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        est_cost = (total_chars / 1_000_000.0) * (30.0 if model == "tts-1-hd" else 15.0)
        if info_cb:
            info_cb(f"synth done format={fmt} out={out_mp3.name} est_cost_usd=${est_cost:.4f}")
        mpx_path = _update_mpx_for_path(src, source_md=src if src.suffix.lower() == ".md" else None)
        results.append(
            {
                "format": fmt,
                "source": str(src),
                "mp3": str(out_mp3),
                "synth_json": str(sidecar_path) if emit_sidecar else None,
                "mpx": str(mpx_path),
                "chars": total_chars,
                "est_cost_usd": round(est_cost, 4),
                "voices": sorted(set([c["voice"] for c in chunks])),
            }
        )
        if progress_cb:
            progress_cb("synth_targets", t_idx, len(targets), f"Completed {fmt}")

    return {"outputs": results}


def _payload_to_words(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = payload.get("words")
    if isinstance(words, list) and words:
        return words
    out = []
    for seg in payload.get("segments") or []:
        out.extend(seg.get("words") or [])
    return out


def _payload_to_vtt(payload: Dict[str, Any]) -> str:
    words = _payload_to_words(payload)
    lines = ["WEBVTT", ""]
    idx = 1
    for w in words:
        txt = (w.get("word") or w.get("text") or "").strip()
        if not txt:
            continue
        st = float(w.get("start", 0.0))
        en = float(w.get("end", st))

        def ts(sec: float) -> str:
            ms = int(round(sec * 1000))
            h, r = divmod(ms, 3600000)
            m, r = divmod(r, 60000)
            s, mm = divmod(r, 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{mm:03d}"

        lines.append(str(idx))
        lines.append(f"{ts(st)} --> {ts(en)}")
        lines.append(txt)
        lines.append("")
        idx += 1
    return "\n".join(lines).rstrip() + "\n"


def _transcribe_cloud(audio_path: Path, *, model: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for cloud captions")
    client = OpenAI(api_key=api_key)
    with audio_path.open("rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        except TypeError:
            f.seek(0)
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
            )
    return resp.model_dump() if hasattr(resp, "model_dump") else resp


def _apple_transcribe_words(wav_path: Path, *, locale: str = "en-US", on_device: bool = False) -> Dict[str, Any]:
    if not APPLE_SPEECH_AVAILABLE:
        raise RuntimeError(
            "Apple Speech frameworks are not available. "
            "Install Apple bridge packages in your venv:\n"
            "pip install pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-AVFoundation pyobjc-framework-Speech\n"
            "or use cloud captions with --cloud."
        )

    try:
        import AppKit  # type: ignore

        AppKit.NSApplication.sharedApplication()
    except Exception:
        pass

    status = Speech.SFSpeechRecognizer.authorizationStatus()
    try:
        NOT_DETERMINED = Speech.SFSpeechRecognizerAuthorizationStatusNotDetermined
        AUTHORIZED = Speech.SFSpeechRecognizerAuthorizationStatusAuthorized
        DENIED = Speech.SFSpeechRecognizerAuthorizationStatusDenied
        RESTRICTED = Speech.SFSpeechRecognizerAuthorizationStatusRestricted
    except Exception:
        NOT_DETERMINED = 0
        DENIED = 1
        RESTRICTED = 2
        AUTHORIZED = 3

    if status == NOT_DETERMINED:
        ev = threading.Event()
        holder = {"status": status}

        def _auth_cb(new_status: Any) -> None:
            try:
                holder["status"] = int(new_status)
            except Exception:
                holder["status"] = new_status
            ev.set()

        Speech.SFSpeechRecognizer.requestAuthorization_(_auth_cb)
        deadline = time.time() + 30.0
        while not ev.is_set() and time.time() < deadline:
            try:
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
            except Exception:
                pass
            ev.wait(timeout=0.05)
        status = holder.get("status", status)

    if status != AUTHORIZED:
        hint = (
            "Enable Speech Recognition for the app executing this process in "
            "System Settings -> Privacy & Security -> Speech Recognition."
        )
        if status in (DENIED, RESTRICTED):
            hint += " If previously denied, reset permissions and restart the app."
        raise RuntimeError(f"Apple Speech is not authorized ({status}). {hint}")

    ns_locale = None
    try:
        from Foundation import NSLocale  # type: ignore

        ns_locale = NSLocale.localeWithLocaleIdentifier_(locale)
    except Exception:
        ns_locale = None

    recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(ns_locale) if ns_locale else Speech.SFSpeechRecognizer.alloc().init()
    if recognizer is None:
        raise RuntimeError("Failed to initialize Apple speech recognizer")

    request = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(NSURL.fileURLWithPath_(str(wav_path)))
    request.setShouldReportPartialResults_(False)
    if hasattr(request, "setRequiresOnDeviceRecognition_"):
        request.setRequiresOnDeviceRecognition_(bool(on_device))

    done = threading.Event()
    out: Dict[str, Any] = {"payload": None, "error": None}

    def _handler(result: Any, error: Any) -> None:
        if error is not None:
            out["error"] = str(error)
            done.set()
            return
        if result is None:
            return
        if hasattr(result, "isFinal") and not result.isFinal():
            return
        bt = result.bestTranscription()
        text = str(bt.formattedString()) if bt is not None else ""
        words: List[Dict[str, Any]] = []
        if bt is not None:
            for seg in bt.segments():
                token = str(seg.substring()).strip()
                if not token:
                    continue
                start = float(seg.timestamp())
                dur = float(seg.duration())
                words.append({"word": token, "start": start, "end": start + dur})
        out["payload"] = {"source": "apple_speech", "locale": locale, "text": text, "words": words}
        done.set()

    _task = recognizer.recognitionTaskWithRequest_resultHandler_(request, _handler)
    _ = _task
    deadline = time.time() + 300.0
    while not done.is_set() and time.time() < deadline:
        try:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
        except Exception:
            pass
        done.wait(timeout=0.05)

    if out["error"]:
        raise RuntimeError(f"Apple Speech transcription failed: {out['error']}")
    if not out["payload"]:
        raise RuntimeError("Apple Speech transcription produced no output")
    return out["payload"]


def _extract_wav_chunk(input_wav: Path, output_wav: Path, *, start_sec: float, dur_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{dur_sec:.3f}",
        "-i",
        str(input_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True)


def _apple_transcribe_words_chunked(
    wav_path: Path,
    *,
    locale: str = "en-US",
    on_device: bool = False,
    chunk_seconds: float = 45.0,
    overlap_seconds: float = 0.5,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    info_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Transcribe long WAVs in chunks to avoid partial/final-only Apple Speech output.
    """
    total_dur = _audio_duration_seconds(wav_path)
    if total_dur <= 0:
        return _apple_transcribe_words(wav_path, locale=locale, on_device=on_device)

    if chunk_seconds <= 0:
        chunk_seconds = 120.0
    if overlap_seconds < 0:
        overlap_seconds = 0.0
    step = max(1.0, chunk_seconds - overlap_seconds)
    starts: List[float] = []
    t = 0.0
    while t < total_dur:
        starts.append(t)
        t += step

    merged_words: List[Dict[str, Any]] = []
    texts: List[str] = []

    with tempfile.TemporaryDirectory(prefix="tts3x_apple_chunks_") as td:
        for idx, start in enumerate(starts, start=1):
            dur = min(chunk_seconds, max(0.1, total_dur - start))
            chunk = Path(td) / f"chunk_{idx:04d}.wav"
            _extract_wav_chunk(wav_path, chunk, start_sec=start, dur_sec=dur)
            if progress_cb:
                progress_cb("apple_chunks", idx, len(starts), f"Apple chunk {idx}/{len(starts)}")
            payload = _apple_transcribe_words(chunk, locale=locale, on_device=on_device)
            words = payload.get("words") or []
            text = (payload.get("text") or "").strip()
            if text:
                texts.append(text)
            for w in words:
                token = (w.get("word") or w.get("text") or "").strip()
                if not token:
                    continue
                ws = float(w.get("start", 0.0)) + start
                we = float(w.get("end", ws)) + start
                if we < ws:
                    we = ws
                merged_words.append({"word": token, "start": ws, "end": we})

    # Sort and dedupe overlapping duplicates caused by chunk overlap.
    merged_words.sort(key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))))
    dedup: List[Dict[str, Any]] = []
    for w in merged_words:
        if not dedup:
            dedup.append(w)
            continue
        prev = dedup[-1]
        # Consider same word within ~250ms as overlap duplicate.
        if (
            (w.get("word") or "").strip().lower() == (prev.get("word") or "").strip().lower()
            and abs(float(w.get("start", 0.0)) - float(prev.get("start", 0.0))) < 0.25
        ):
            continue
        dedup.append(w)

    if info_cb:
        info_cb(f"apple chunks={len(starts)} words={len(dedup)} duration={total_dur:.2f}s")
    return {
        "source": "apple_speech",
        "locale": locale,
        "text": " ".join(texts).strip(),
        "words": dedup,
    }


def captions(
    audio_path: Path,
    *,
    output_base: Optional[Path],
    local: bool,
    cloud: bool,
    model: str,
    locale: str,
    on_device: bool,
    keep_wav: bool,
    overwrite_wav: bool,
    md_path: Optional[Path],
    cooldown: float,
    verify: bool = False,
    verify_threshold: float = 85.0,
    verify_max_start_seconds: float = 10.0,
    verify_max_start_ratio: float = 0.02,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    info_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    base = output_base or _artifact_base_for_audio(audio_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    words_json = Path(str(base) + ".words.json")
    vtt = Path(str(base) + ".vtt")

    use_cloud = bool(cloud)
    use_local = bool(local) or not use_cloud

    payload: Dict[str, Any]
    wav_path = Path(str(base) + ".wav")
    audio_dur = _audio_duration_seconds(audio_path)
    stage_total = 4 if md_path else 3
    stage_idx = 0

    if info_cb:
        mode = "local_apple" if use_local else "cloud"
        info_cb(f"captions mode={mode} model={model} locale={locale} on_device={on_device}")
        info_cb(f"captions audio={audio_path} base={base} duration={audio_dur:.2f}s")

    if use_local:
        stage_idx += 1
        if progress_cb:
            progress_cb("captions", stage_idx, stage_total, "Preparing WAV for Apple Speech")
        if overwrite_wav or not wav_path.exists():
            _to_wav_16k_mono(audio_path, wav_path)
        if info_cb and wav_path.exists():
            info_cb(f"captions wav={wav_path} overwrite={overwrite_wav}")
        stage_idx += 1
        if progress_cb:
            progress_cb("captions", stage_idx, stage_total, "Transcribing with Apple Speech")
        payload = _apple_transcribe_words_chunked(
            wav_path,
            locale=locale,
            on_device=on_device,
            chunk_seconds=45.0,
            overlap_seconds=0.5,
            progress_cb=progress_cb,
            info_cb=info_cb,
        )
    else:
        stage_idx += 1
        if progress_cb:
            progress_cb("captions", stage_idx, stage_total, "Uploading/transcribing with cloud STT")
        payload = _transcribe_cloud(audio_path, model=model)

    stage_idx += 1
    if progress_cb:
        progress_cb("captions", stage_idx, stage_total, "Writing words.json and vtt")
    words_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    vtt.write_text(_payload_to_vtt(payload), encoding="utf-8")
    if info_cb:
        info_cb(f"captions words_json={words_json}")
        info_cb(f"captions vtt={vtt}")
        words = _payload_to_words(payload)
        if words:
            fs = float(words[0].get("start", 0.0))
            ls = float(words[-1].get("end", fs))
            info_cb(f"captions words_count={len(words)} span={fs:.2f}s..{ls:.2f}s")

    if verify:
        thr = float(verify_threshold)
        # Support either percent (85) or ratio (0.85).
        if thr > 1.0:
            thr = thr / 100.0
        cov = _timing_coverage(payload, audio_duration=audio_dur)
        c = cov["coverage"]
        if info_cb:
            info_cb(
                "captions verify "
                f"coverage={c*100:.2f}% threshold={thr*100:.2f}% "
                f"first_start={cov['first_start']:.2f}s last_end={cov['last_end']:.2f}s"
            )
        max_start_abs = max(0.0, float(verify_max_start_seconds))
        max_start_ratio = max(0.0, float(verify_max_start_ratio))
        max_start = max(max_start_abs, audio_dur * max_start_ratio)
        if cov["first_start"] > max_start:
            raise RuntimeError(
                "Caption timing start too late: "
                f"{cov['first_start']:.2f}s > {max_start:.2f}s "
                f"(dur={audio_dur:.2f}s, abs_limit={max_start_abs:.2f}s, ratio_limit={max_start_ratio:.4f}). "
                "This usually indicates partial transcription."
            )
        if c < thr:
            raise RuntimeError(
                "Caption timing coverage below threshold: "
                f"{c*100:.2f}% < {thr*100:.2f}% "
                f"(first_start={cov['first_start']:.2f}s last_end={cov['last_end']:.2f}s dur={audio_dur:.2f}s)"
            )

    out = {"words_json": str(words_json), "vtt": str(vtt)}
    if keep_wav and wav_path.exists():
        out["wav"] = str(wav_path)
    elif wav_path.exists() and not keep_wav:
        wav_path.unlink(missing_ok=True)

    if md_path:
        stage_idx += 1
        if progress_cb:
            progress_cb("captions", stage_idx, stage_total, "Enriching timeline with markdown metadata")
        enriched = enrich_timeline(base=base, audio_path=audio_path, md_path=md_path, cooldown=cooldown)
        out["enriched_json"] = str(enriched)
        if info_cb:
            info_cb(f"captions enriched_json={enriched}")

    out["mpx"] = str(_update_mpx_for_path(audio_path, source_md=md_path))
    return out


def enrich_timeline(base: Path, *, audio_path: Path, md_path: Path, cooldown: float) -> Path:
    words_payload = json.loads(Path(str(base) + ".words.json").read_text(encoding="utf-8"))
    words_raw = _payload_to_words(words_payload)
    words = []
    for w in words_raw:
        txt = (w.get("word") or w.get("text") or "").strip()
        if not txt:
            continue
        st = float(w.get("start", 0.0))
        en = float(w.get("end", st))
        if en < st:
            en = st
        words.append({"word": txt, "start": st, "end": en})

    sidecar_path = Path(str(base) + ".synth.json")

    # Exact mode (preferred): use synth sidecar paragraph token counts.
    if sidecar_path.exists():
        synth = json.loads(sidecar_path.read_text(encoding="utf-8"))
        paragraphs = synth.get("paragraphs") or []
    else:
        text = _read_text(md_path)
        paragraphs = parse_markdown_for_synth(
            text,
            read_headers="none",
            url_policy="drop",
            url_placeholder="[link]",
            img_policy="none",
            img_placeholder="[image]",
        )

    w_idx = 0
    out_paras = []
    header_events = []
    prev = HeaderState()

    for p in paragraphs:
        n = len(p.get("tokens") or _tokenize(p.get("spoken_text") or ""))
        if n <= 0:
            continue
        if w_idx >= len(words):
            break
        start_word = w_idx
        end_word = min(len(words), w_idx + n) - 1
        if end_word < start_word:
            continue

        start_t = float(words[start_word]["start"])
        end_t = float(words[end_word]["end"])

        out_paras.append(
            {
                "id": p.get("id"),
                "kind": p.get("kind", "paragraph"),
                "start_word": start_word,
                "end_word": end_word,
                "start": start_t,
                "end": end_t,
                "h1": p.get("h1"),
                "h2": p.get("h2"),
                "h3": p.get("h3"),
                "urls": p.get("urls", []),
                "images": p.get("images", []),
            }
        )

        h1, h2, h3 = p.get("h1"), p.get("h2"), p.get("h3")
        if h1 and h1 != prev.h1:
            header_events.append({"type": "header_change", "level": 1, "text": h1, "start": start_t, "cooldown_sec": cooldown})
        if h2 and h2 != prev.h2:
            header_events.append({"type": "header_change", "level": 2, "text": h2, "start": start_t, "cooldown_sec": cooldown})
        if h3 and h3 != prev.h3:
            header_events.append({"type": "header_change", "level": 3, "text": h3, "start": start_t, "cooldown_sec": cooldown})
        prev = HeaderState(h1=h1, h2=h2, h3=h3)

        w_idx += n

    enriched = {
        "schema": "tts3.timeline.v1",
        "audio_file": audio_path.name,
        "source_md": md_path.name,
        "cooldown_sec": cooldown,
        "words": words,
        "paragraphs": out_paras,
        "header_events": header_events,
    }

    out_path = Path(str(base) + ".json")
    out_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return out_path


def _resolve_audio_words(input_path: Path) -> Tuple[Path, Path]:
    lower_name = input_path.name.lower()
    if lower_name.endswith(".words.json"):
        stem = input_path.name[: -len(".words.json")]
        base = input_path.with_name(stem)
        audio = Path(str(base) + ".mp3")
        if not audio.exists():
            parent_audio = input_path.parent.parent / f"{stem}.mp3"
            if parent_audio.exists():
                audio = parent_audio
        words = input_path
        return audio, words
    if input_path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}:
        audio = input_path
        work_base = _artifact_base_for_audio(input_path)
        words = Path(str(work_base) + ".words.json")
        if not words.exists():
            words = Path(str(input_path.with_suffix("")) + ".words.json")
        if not words.exists():
            words = Path(str(work_base) + ".json")
        if not words.exists():
            words = Path(str(input_path.with_suffix("")) + ".json")
        return audio, words

    base = input_path.with_suffix("")
    audio = Path(str(base) + ".mp3")
    work_base = _artifact_base_for_audio(audio)
    words = Path(str(work_base) + ".words.json")
    if not words.exists():
        words = Path(str(base) + ".words.json")
    return audio, words


def _normalize_words(words_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in _payload_to_words(words_payload):
        token = (w.get("word") or w.get("text") or "").strip()
        if not token:
            continue
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        if end < start:
            end = start
        out.append({"word": token, "start": start, "end": end})
    return out


def _ass_ts(seconds: float) -> str:
    t = max(0.0, float(seconds))
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100.0))
    if cs >= 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", " ")
    )


def _focus_center_markup(token: str) -> str:
    t = (token or "").strip()
    if not t:
        return ""
    n = len(t)
    c = max(0, (n - 1) // 2)
    left = _ass_escape(t[:c])
    mid = _ass_escape(t[c : c + 1])
    right = _ass_escape(t[c + 1 :])
    return f"{left}{{\\c&H0000FF&}}{mid}{{\\c&HFFFFFF&}}{right}"


def make_mp4(
    input_path: Path,
    *,
    format_spec: str = "s",
    output_path: Optional[Path] = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    font_size: int = 72,
) -> Dict[str, Any]:
    p = input_path.expanduser()
    audio_path: Path
    words_path: Path

    if p.suffix.lower() == ".md":
        aliases = {"l": "literal", "s": "speakable", "e": "explained", "t": "ted-talk", "m": "summary", "a": "speakable"}
        fmt = aliases.get((format_spec or "s").strip().lower(), (format_spec or "speakable").strip().lower())
        if fmt not in {"literal", "speakable", "explained", "ted-talk", "summary"}:
            fmt = "speakable"
        md_variant = resolve_variant_path(p, fmt)
        audio_path = md_variant.with_suffix(".mp3")
        words_path = _resolve_audio_words(audio_path)[1]
    else:
        audio_path, words_path = _resolve_audio_words(p)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not words_path.exists():
        raise FileNotFoundError(f"Words JSON not found: {words_path}")

    words_payload = json.loads(words_path.read_text(encoding="utf-8"))
    words = _normalize_words(words_payload)
    if not words:
        raise RuntimeError(f"No words found in {words_path}")

    if output_path is None:
        work_dir = _artifact_work_dir_for_path(audio_path)
        work_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = work_dir / f"{audio_path.with_suffix('').name}.mp4"
    else:
        out_mp4 = output_path.expanduser()
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8") as tf:
        ass_path = Path(tf.name)
        tf.write("[Script Info]\n")
        tf.write("ScriptType: v4.00+\n")
        tf.write(f"PlayResX: {int(width)}\n")
        tf.write(f"PlayResY: {int(height)}\n")
        tf.write("WrapStyle: 2\n")
        tf.write("ScaledBorderAndShadow: yes\n\n")
        tf.write("[V4+ Styles]\n")
        tf.write(
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        )
        tf.write(
            f"Style: Default,Menlo,{int(font_size)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
            "0,0,0,0,100,100,0,0,1,2,0,5,40,40,40,1\n\n"
        )
        tf.write("[Events]\n")
        tf.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for w in words:
            token = (w.get("word") or "").strip()
            if not token:
                continue
            st = float(w.get("start", 0.0))
            en = float(w.get("end", st))
            if en <= st:
                en = st + 0.08
            text = _focus_center_markup(token)
            tf.write(
                f"Dialogue: 0,{_ass_ts(st)},{_ass_ts(en)},Default,,0,0,0,,{text}\n"
            )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        ass_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg not found in PATH.")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={int(width)}x{int(height)}:r={int(fps)}",
        "-i",
        str(audio_path),
        "-vf",
        f"ass={ass_path}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_mp4),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        ass_path.unlink(missing_ok=True)
        err = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"ffmpeg failed: {err}") from e

    ass_path.unlink(missing_ok=True)
    mpx = _update_mpx_for_path(audio_path, source_md=None)
    return {
        "audio": str(audio_path),
        "words_json": str(words_path),
        "mp4": str(out_mp4),
        "mpx": str(mpx),
        "ffmpeg_rc": proc.returncode,
    }


def read(input_path: Path, *, seconds: float = 0.0, window: int = 8, speed_preset: int = 2) -> Dict[str, int]:
    if not APPLE_SPEECH_AVAILABLE:
        raise RuntimeError("Read mode requires macOS AVFoundation/pyobjc.")
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    audio_path, words_path = _resolve_audio_words(input_path.expanduser())
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not words_path.exists():
        raise FileNotFoundError(f"Words JSON not found: {words_path}")

    payload = json.loads(words_path.read_text(encoding="utf-8"))
    words = _normalize_words(payload)
    if not words:
        raise RuntimeError(f"No words in payload: {words_path}")

    url = NSURL.fileURLWithPath_(str(audio_path.resolve()))
    player, err = AVFoundation.AVAudioPlayer.alloc().initWithContentsOfURL_error_(url, None)
    if err is not None or player is None:
        raise RuntimeError(f"Failed to load audio: {err}")

    speed_map = {
        1: 1.00,
        2: 1.25,
        3: 1.50,
        4: 2.00,
        5: 2.25,
        6: 2.50,
        7: 3.00,
        8: 4.00,
    }
    preset = max(1, min(8, int(speed_preset)))
    player.setEnableRate_(True)
    player.setRate_(float(speed_map[preset]))

    starts = [w["start"] for w in words]
    dur = float(player.duration()) if player is not None else 0.0
    if starts and dur > 0 and starts[0] > max(5.0, dur * 0.2):
        raise RuntimeError(
            f"Word timings start too late ({starts[0]:.2f}s of {dur:.2f}s). "
            "Re-run captions; local Apple STT may have produced a partial transcript."
        )

    def find_idx(t: float) -> int:
        lo, hi = 0, len(starts) - 1
        if hi < 0:
            return 0
        if t <= starts[0]:
            return 0
        if t >= starts[-1]:
            return len(starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if starts[mid] <= t:
                lo = mid + 1
            else:
                hi = mid - 1
        return max(0, min(len(starts) - 1, hi))

    desired_start: Optional[float] = None
    if seconds > 0:
        desired_start = float(seconds)
    elif starts and starts[0] > 0.5:
        # If captions start after a long preroll, jump near the first timed word.
        desired_start = max(0.0, float(starts[0]) - 0.05)

    player.prepareToPlay()
    if desired_start is not None:
        player.setCurrentTime_(desired_start)
    player.play()
    # Some AVAudioPlayer setups ignore pre-play seeks; enforce again once playback starts.
    if desired_start is not None and float(player.currentTime()) < desired_start - 0.25:
        player.setCurrentTime_(desired_start)

    state = {"quit": False, "window": max(0, int(window)), "speed_preset": preset}
    kb = KeyBindings()

    def toggle() -> None:
        if player.isPlaying():
            player.pause()
        else:
            player.play()

    def seek(delta: float) -> None:
        t = float(player.currentTime()) + delta
        t = max(0.0, min(t, float(player.duration())))
        was_playing = bool(player.isPlaying())
        player.pause()
        player.setCurrentTime_(t)
        if was_playing:
            player.play()

    @kb.add(" ")
    def _(event: Any) -> None:
        toggle()

    @kb.add("q")
    @kb.add("Q")
    @kb.add("escape")
    def _(event: Any) -> None:
        state["quit"] = True
        event.app.exit()

    @kb.add("left")
    @kb.add("b")
    def _(event: Any) -> None:
        seek(-10.0)

    @kb.add("right")
    @kb.add("f")
    def _(event: Any) -> None:
        seek(10.0)

    @kb.add("c-left")
    @kb.add("home")
    def _(event: Any) -> None:
        was_playing = bool(player.isPlaying())
        player.pause()
        player.setCurrentTime_(0.0)
        if was_playing:
            player.play()

    @kb.add("+")
    @kb.add("=")
    def _(event: Any) -> None:
        state["window"] = min(200, int(state["window"]) + 1)

    @kb.add("-")
    @kb.add("_")
    def _(event: Any) -> None:
        state["window"] = max(0, int(state["window"]) - 1)

    def set_speed(p: int) -> None:
        p = max(1, min(8, int(p)))
        state["speed_preset"] = p
        player.setRate_(float(speed_map[p]))

    @kb.add("1")
    def _(event: Any) -> None:
        set_speed(1)

    @kb.add("2")
    def _(event: Any) -> None:
        set_speed(2)

    @kb.add("3")
    def _(event: Any) -> None:
        set_speed(3)

    @kb.add("4")
    def _(event: Any) -> None:
        set_speed(4)

    @kb.add("5")
    def _(event: Any) -> None:
        set_speed(5)

    @kb.add("6")
    def _(event: Any) -> None:
        set_speed(6)

    @kb.add("7")
    def _(event: Any) -> None:
        set_speed(7)

    @kb.add("8")
    def _(event: Any) -> None:
        set_speed(8)

    def _distance_style(dist: int) -> str:
        if dist <= 2:
            return "class:near"
        if dist <= 5:
            return "class:mid"
        return "class:far"

    def _build_centered_line(idx: int, ww: int) -> List[Tuple[str, str]]:
        cols = max(40, shutil.get_terminal_size((120, 30)).columns)
        target = max(20, int(cols * 0.8))

        selected = {idx}
        left = idx - 1
        right = idx + 1
        text_len = len(words[idx]["word"]) + 2  # [word]

        # Expand around focus word while respecting window and char budget.
        for _ in range(ww * 2):
            added = False
            if left >= 0 and (idx - left) <= ww:
                w = words[left]["word"]
                if text_len + len(w) + 1 <= target:
                    selected.add(left)
                    text_len += len(w) + 1
                    left -= 1
                    added = True
            if right < len(words) and (right - idx) <= ww:
                w = words[right]["word"]
                if text_len + len(w) + 1 <= target:
                    selected.add(right)
                    text_len += len(w) + 1
                    right += 1
                    added = True
            if not added:
                break

        lo = min(selected)
        hi = max(selected)
        segments: List[Tuple[str, str]] = []
        if lo > 0:
            segments.append(("class:edge", "… "))

        focus_segment_index = -1
        for i in range(lo, hi + 1):
            token = words[i]["word"]
            if i == idx:
                focus_segment_index = len(segments)
                segments.append(("class:focus", token))
            else:
                dist = abs(i - idx)
                segments.append((_distance_style(dist), token))
            if i < hi:
                segments.append(("class:space", " "))

        if hi < len(words) - 1:
            segments.append(("class:edge", " …"))

        # Place the center character of the focus word at terminal center.
        center_col = cols // 2
        prefix_len = 0
        focus_center_offset = 0
        for si, (_, txt) in enumerate(segments):
            if si == focus_segment_index:
                n = len(txt)
                # For even lengths, bias to the left-center char.
                focus_center_offset = max(0, (n - 1) // 2)
                break
            prefix_len += len(txt)
        left_pad = max(0, center_col - (prefix_len + focus_center_offset))
        if left_pad:
            segments.insert(0, ("", " " * left_pad))

        # Split focus word so center letter can be blue.
        if focus_segment_index >= 0:
            # account for inserted left pad segment if present
            fs = focus_segment_index + (1 if left_pad else 0)
            _, focus_txt = segments[fs]
            n = len(focus_txt)
            if n > 0:
                c = max(0, (n - 1) // 2)
                left_part = focus_txt[:c]
                center_part = focus_txt[c : c + 1]
                right_part = focus_txt[c + 1 :]
                replacement: List[Tuple[str, str]] = []
                if left_part:
                    replacement.append(("class:focus", left_part))
                replacement.append(("class:focus_center", center_part))
                if right_part:
                    replacement.append(("class:focus", right_part))
                segments = segments[:fs] + replacement + segments[fs + 1 :]
        return segments

    def render() -> List[Tuple[str, str]]:
        t = float(player.currentTime())
        dur = float(player.duration())
        idx = find_idx(t)
        ww = int(state["window"])
        sp = int(state["speed_preset"])
        mode = "PLAY" if player.isPlaying() else "PAUSE"
        info = f"{mode} {t:6.2f}s/{dur:6.2f}s  idx={idx+1}/{len(words)}  window={ww}  speed={sp}({speed_map[sp]:.2f}x)"
        controls = "space pause/resume | q/Q/esc quit | <-/-> seek 10s | ctrl+left/home start | +/- context | 1-8 speed"

        body: List[Tuple[str, str]] = [("class:header", info + "\n\n")]
        body.extend(_build_centered_line(idx, ww))
        body.append(("", "\n\n"))
        body.append(("class:meta", controls))
        return body

    control = FormattedTextControl(render)
    root = HSplit([Window(control, wrap_lines=False)])
    style = Style.from_dict(
        {
            "header": "bold",
            "meta": "fg:#888888",
            "space": "",
            "focus": "bold fg:#ff3b30",
            "focus_center": "bold fg:#3b82f6",
            "near": "fg:#b8b8b8",
            "mid": "fg:#7a7a7a",
            "far": "fg:#4b4b4b",
            "edge": "fg:#2f2f2f",
        }
    )
    app = Application(layout=Layout(root), key_bindings=kb, style=style, full_screen=False)

    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            time.sleep(0.05)
            if state["quit"]:
                stop.set()
                return
            if (not player.isPlaying()) and float(player.currentTime()) >= float(player.duration()) - 0.05:
                state["quit"] = True
                try:
                    app.exit()
                except Exception:
                    pass
                stop.set()
                return
            try:
                app.invalidate()
            except Exception:
                pass

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    try:
        app.run()
    finally:
        stop.set()
        try:
            if player.isPlaying():
                player.stop()
        except Exception:
            pass
    return {"window": int(state["window"]), "speed_preset": int(state["speed_preset"])}
