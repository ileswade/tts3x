from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import mimetypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import base64

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import pipeline

console = Console()

class UserCancelledError(RuntimeError):
    pass

DEFAULT_CONFIG: dict[str, Any] = {
    "global": {
        "display": "normal",
        "verbose": 2,
        "logging": 0,
        "logging_file": None,
        "logging_clear": False,
    },
    "import": {
        "ocr": "auto",
        "output_dir": None,
    },
    "adapt": {
        "format": "speakable,explained,summary",
        "url": "mention",
        "url_mention": "URL provided",
        "img": "mention",
        "img_mention": "Image provided",
        "explained_model": "gpt-4o-mini",
        "explained_back": 2,
        "explained_forward": 2,
        "explained_persona": "Iles",
        "explained_profile": "conversational_companion",
        "no_summary": False,
        "no_instructions": False,
    },
    "synth": {
        "quality": "hd",
        "voice": "nova",
        "request_timeout_seconds": 60,
        "max_retries": 3,
        "multi_thread": 5,
        "read_headers": "none",
        "url": "drop",
        "url_mention": "URL provided",
        "img": "none",
        "img_mention": "Image provided",
        "no_sidecar": False,
    },
    "read": {
        "window": 8,
        "speed_preset": 2,
    },
    "convert": {
        "format": "s",
        "captions": False,
        "jobs": 1,
        "ocr": "auto",
    },
    "mp4": {
        "format": "s",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "font_size": 72,
    },
}

_LAST_CONFIG_META: dict[str, Any] = {}


def _merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            nested = dict(out[k])
            nested.update(v)
            out[k] = nested
        else:
            out[k] = v
    return out


def _config_path() -> Path:
    explicit = os.getenv("TTS3X_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".config" / "tts3x" / "config.json"


def _hermes_export_candidates() -> list[Path]:
    out: list[Path] = []
    explicit = os.getenv("TTS3X_HERMES_EXPORT")
    if explicit:
        out.append(Path(explicit).expanduser())
    out.extend(
        [
            Path.home() / ".config" / "hermes" / "export.toml",
            Path.home() / ".config" / "hermes" / "export.json",
            Path.home() / ".hermes" / "export.toml",
            Path.home() / ".hermes" / "export.json",
        ]
    )
    return out


def _load_blob(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    # Support wrapped export payloads where config sits under "tts3x" or "config".
    if isinstance(data.get("tts3x"), dict):
        return data["tts3x"]
    if isinstance(data.get("config"), dict):
        return data["config"]
    return data


def _load_config_with_meta() -> tuple[dict[str, Any], dict[str, Any]]:
    provider = os.getenv("TTS3X_CONFIG_PROVIDER", "auto").strip().lower() or "auto"
    if provider not in {"auto", "local", "hermes"}:
        provider = "auto"

    local_path = _config_path()
    local_cfg: dict[str, Any] = {}
    local_error: Optional[str] = None
    if local_path.exists():
        try:
            local_cfg = _load_blob(local_path)
        except Exception as e:
            local_error = str(e)

    hermes_path_used: Optional[Path] = None
    hermes_cfg: dict[str, Any] = {}
    hermes_error: Optional[str] = None
    for candidate in _hermes_export_candidates():
        if not candidate.exists():
            continue
        try:
            hermes_cfg = _load_blob(candidate)
            hermes_path_used = candidate
            break
        except Exception as e:
            hermes_error = f"{candidate}: {e}"
            continue

    use_hermes = bool(hermes_cfg) and provider in {"auto", "hermes"}
    cfg = _merge_config(dict(DEFAULT_CONFIG), local_cfg)
    if use_hermes:
        cfg = _merge_config(cfg, hermes_cfg)

    resolved = "hermes" if use_hermes else "local"
    meta = {
        "provider_requested": provider,
        "provider_resolved": resolved,
        "local_config_path": str(local_path),
        "local_config_found": local_path.exists(),
        "local_error": local_error,
        "hermes_export_path": str(hermes_path_used) if hermes_path_used else None,
        "hermes_found": hermes_path_used is not None,
        "hermes_error": hermes_error,
    }
    return cfg, meta


def _load_config() -> dict[str, Any]:
    global _LAST_CONFIG_META
    cfg, meta = _load_config_with_meta()
    _LAST_CONFIG_META = meta
    return cfg


def _config_source_meta() -> dict[str, Any]:
    if not _LAST_CONFIG_META:
        _, meta = _load_config_with_meta()
        return meta
    return _LAST_CONFIG_META


def _save_config(cfg: dict[str, Any]) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def _runtime_state_path() -> Path:
    return Path.home() / ".config" / "tts3x" / "runtime.json"


def _load_runtime_state() -> dict[str, Any]:
    p = _runtime_state_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_runtime_state(state: dict[str, Any]) -> Path:
    p = _runtime_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return p


def _coerce_scalar(text: str) -> Any:
    t = text.strip()
    tl = t.lower()
    if tl in {"true", "false"}:
        return tl == "true"
    if tl in {"null", "none"}:
        return None
    try:
        if "." in t:
            return float(t)
        return int(t)
    except Exception:
        pass
    # Try JSON for arrays/objects.
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(t)
        except Exception:
            pass
    return text


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _cfg_set(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur: dict[str, Any] = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _cfg_flatten_keys(d: dict[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(_cfg_flatten_keys(v, p))
        else:
            keys.append(p)
    return keys


def _resolve(cli_value: Any, cfg: dict[str, Any], path: str, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return _cfg_get(cfg, path, fallback)


class _CliLogger:
    def __init__(self, level: int, log_path: Optional[Path]) -> None:
        self.level = max(0, min(3, int(level)))
        self.log_path = log_path
        self._enabled = self.level > 0 and self.log_path is not None

    def write(self, level: int, message: str) -> None:
        if not self._enabled or int(level) > self.level:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")


def _derive_primary_input_path(args: argparse.Namespace) -> Optional[Path]:
    if getattr(args, "input", None):
        return Path(args.input).expanduser()
    if getattr(args, "audio", None):
        return Path(args.audio).expanduser()
    return None


def _resolve_log_file_path(args: argparse.Namespace, primary_input: Optional[Path], raw: Optional[str]) -> Optional[Path]:
    if raw is None:
        raw = getattr(args, "logging_file", None)
    if int(getattr(args, "logging", 0) or 0) <= 0 and not any(
        [getattr(args, "l1", False), getattr(args, "l2", False), getattr(args, "l3", False)]
    ):
        return None

    ts = datetime.now().strftime("%y%m%d.%H%M")
    base_stem = primary_input.stem if primary_input is not None else "tts3x"
    default_name = f"{base_stem}-{ts}.log"

    if raw:
        p = Path(raw).expanduser()
        # Treat existing directory, trailing slash, or no suffix as "folder target".
        if p.exists() and p.is_dir():
            return p / default_name
        if str(raw).endswith("/") or p.suffix == "":
            return p / default_name
        return p

    if primary_input is not None:
        return primary_input.parent / default_name
    return Path.cwd() / default_name


def _setup_logger(args: argparse.Namespace, cfg: Optional[dict[str, Any]] = None) -> _CliLogger:
    if cfg is None:
        cfg = _load_config()
    cli_logging = getattr(args, "logging", None)
    cfg_logging = _cfg_get(cfg, "global.logging", 0)
    if getattr(args, "l0", False):
        level = 0
    elif getattr(args, "l1", False):
        level = 1
    elif getattr(args, "l2", False):
        level = 2
    elif getattr(args, "l3", False):
        level = 3
    else:
        level = int(cli_logging if cli_logging is not None else cfg_logging or 0)
    args.logging = level
    primary = _derive_primary_input_path(args)
    log_file_raw = getattr(args, "logging_file", None)
    if log_file_raw is None:
        log_file_raw = _cfg_get(cfg, "global.logging_file", None)
    log_path = _resolve_log_file_path(args, primary, log_file_raw)
    clear = bool(getattr(args, "logging_clear", False))
    if not clear:
        clear = bool(_cfg_get(cfg, "global.logging_clear", False))
    if log_path and clear:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    logger = _CliLogger(level=level, log_path=log_path)
    if logger.log_path is not None and logger.level > 0:
        logger.write(1, f"log_file={logger.log_path}")
    return logger


def _display_mode(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    cli = args.display
    if cli is None:
        cli = _cfg_get(cfg, "global.display", "normal")
    if cli in {"r", "rich"}:
        return "rich"
    return "normal"


def _verbosity(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    if getattr(args, "quiet", False):
        return 0
    if getattr(args, "v0", False):
        return 0
    if getattr(args, "v1", False):
        return 1
    if getattr(args, "v2", False):
        return 2
    if getattr(args, "v3", False):
        return 3
    if args.verbose is not None:
        return max(0, min(3, int(args.verbose)))
    return int(_cfg_get(cfg, "global.verbose", 2))


def _print(obj: Any, *, verbosity: int, display: str) -> None:
    if verbosity <= 0:
        return
    if verbosity == 1:
        if isinstance(obj, dict):
            for k in ("md", "words_json", "vtt", "enriched_json"):
                if k in obj:
                    print(obj[k])
            if "outputs" in obj:
                for o in obj["outputs"]:
                    if isinstance(o, dict) and o.get("mp3"):
                        print(o["mp3"])
            else:
                for _, v in obj.items():
                    if isinstance(v, str) and v:
                        print(v)
        return
    if display == "rich":
        console.print_json(json.dumps(obj, indent=2))
    else:
        print(json.dumps(obj, indent=2))


def _is_derived_md(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith(("-literal", "-speakable", "-explained", "-explain", "-ted-talk", "-tedtalk", "-summary", "-instructions"))


def _variant_type_from_stem(stem: str) -> str:
    s = stem.lower()
    if s.endswith("-speakable"):
        return "S"
    if s.endswith("-explained") or s.endswith("-explain"):
        return "E"
    if s.endswith("-ted-talk") or s.endswith("-tedtalk"):
        return "T"
    return "L"


def _interactive_pick_index(title: str, items: list[str], *, display: str) -> int:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
    except Exception as e:
        raise RuntimeError(f"Interactive picker requires prompt_toolkit: {e}")

    state: dict[str, Any] = {"idx": 0, "typed": "", "cancelled": False}
    kb = KeyBindings()

    @kb.add("up")
    def _(event: Any) -> None:
        state["idx"] = max(0, int(state["idx"]) - 1)
        state["typed"] = ""

    @kb.add("down")
    def _(event: Any) -> None:
        state["idx"] = min(len(items) - 1, int(state["idx"]) + 1)
        state["typed"] = ""

    @kb.add("q")
    @kb.add("Q")
    @kb.add("escape")
    @kb.add("backspace")
    @kb.add("c-c")
    def _(event: Any) -> None:
        state["cancelled"] = True
        event.app.exit(result=-1)

    @kb.add("enter")
    @kb.add("c-m")
    def _(event: Any) -> None:
        typed = str(state.get("typed", "")).strip()
        if typed:
            if not typed.isdigit():
                state["cancelled"] = True
                event.app.exit(result=-1)
                return
            idx = int(typed) - 1
            if idx < 0 or idx >= len(items):
                state["cancelled"] = True
                event.app.exit(result=-1)
                return
            event.app.exit(result=idx)
            return
        event.app.exit(result=int(state["idx"]))

    @kb.add("0")
    @kb.add("1")
    @kb.add("2")
    @kb.add("3")
    @kb.add("4")
    @kb.add("5")
    @kb.add("6")
    @kb.add("7")
    @kb.add("8")
    @kb.add("9")
    def _(event: Any) -> None:
        state["typed"] = str(state.get("typed", "")) + event.data

    @kb.add("<any>")
    def _(event: Any) -> None:
        # Any key outside arrows/numbers/enter/cancel counts as invalid and cancels.
        state["cancelled"] = True
        event.app.exit(result=-1)

    def render() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        out.append(("class:title", f"{title}\n\n"))
        out.append(("class:meta", "Use ↑/↓ to choose, number+Enter to pick, q/esc/backspace to cancel.\n\n"))
        typed = str(state.get("typed", ""))
        if typed:
            out.append(("class:meta", f"Typed index: {typed}\n\n"))
        for i, item in enumerate(items, start=1):
            marker = "❯ " if (i - 1) == int(state["idx"]) else "  "
            style = "class:selected" if (i - 1) == int(state["idx"]) else ""
            out.append((style, f"{marker}{i}. {item}\n"))
        return out

    style = Style.from_dict(
        {
            "title": "bold",
            "meta": "fg:#888888",
            "selected": "bold fg:#4ade80",
        }
    )
    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(render), always_hide_cursor=True)])),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )
    result = int(app.run())
    if result < 0 or bool(state.get("cancelled")):
        raise UserCancelledError("Selection cancelled.")
    return result


def _interactive_pick_path_with_lse_filter(
    candidates: list[Path],
    *,
    title: str,
    row_label: Callable[[Path], str],
) -> Path:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
    except Exception as e:
        raise RuntimeError(f"Interactive picker requires prompt_toolkit: {e}")

    if not candidates:
        raise FileNotFoundError("No candidates available for selection.")

    state: dict[str, Any] = {"idx": 0, "typed": "", "cancelled": False, "mode": "A"}
    kb = KeyBindings()

    def filtered() -> list[Path]:
        mode = str(state.get("mode", "A"))
        if mode == "A":
            out = list(candidates)
        else:
            out = [p for p in candidates if _variant_type_from_stem(p.stem) == mode]
        return out

    def clamp_idx() -> None:
        items = filtered()
        if not items:
            state["idx"] = 0
            return
        state["idx"] = max(0, min(int(state["idx"]), len(items) - 1))

    @kb.add("up")
    def _(event: Any) -> None:
        state["idx"] = max(0, int(state["idx"]) - 1)
        state["typed"] = ""

    @kb.add("down")
    def _(event: Any) -> None:
        items = filtered()
        state["idx"] = min(max(0, len(items) - 1), int(state["idx"]) + 1)
        state["typed"] = ""

    @kb.add("a")
    @kb.add("A")
    def _(event: Any) -> None:
        state["mode"] = "A"
        state["typed"] = ""
        clamp_idx()

    @kb.add("l")
    @kb.add("L")
    def _(event: Any) -> None:
        state["mode"] = "L"
        state["typed"] = ""
        clamp_idx()

    @kb.add("s")
    @kb.add("S")
    def _(event: Any) -> None:
        state["mode"] = "S"
        state["typed"] = ""
        clamp_idx()

    @kb.add("e")
    @kb.add("E")
    def _(event: Any) -> None:
        state["mode"] = "E"
        state["typed"] = ""
        clamp_idx()

    @kb.add("t")
    @kb.add("T")
    def _(event: Any) -> None:
        state["mode"] = "T"
        state["typed"] = ""
        clamp_idx()

    @kb.add("q")
    @kb.add("Q")
    @kb.add("escape")
    @kb.add("backspace")
    @kb.add("c-c")
    def _(event: Any) -> None:
        state["cancelled"] = True
        event.app.exit(result=-1)

    @kb.add("enter")
    @kb.add("c-m")
    def _(event: Any) -> None:
        items = filtered()
        if not items:
            state["cancelled"] = True
            event.app.exit(result=-1)
            return
        typed = str(state.get("typed", "")).strip()
        if typed:
            if not typed.isdigit():
                state["cancelled"] = True
                event.app.exit(result=-1)
                return
            idx = int(typed) - 1
            if idx < 0 or idx >= len(items):
                state["cancelled"] = True
                event.app.exit(result=-1)
                return
            event.app.exit(result=idx)
            return
        event.app.exit(result=int(state["idx"]))

    @kb.add("0")
    @kb.add("1")
    @kb.add("2")
    @kb.add("3")
    @kb.add("4")
    @kb.add("5")
    @kb.add("6")
    @kb.add("7")
    @kb.add("8")
    @kb.add("9")
    def _(event: Any) -> None:
        state["typed"] = str(state.get("typed", "")) + event.data

    @kb.add("<any>")
    def _(event: Any) -> None:
        state["cancelled"] = True
        event.app.exit(result=-1)

    def render() -> list[tuple[str, str]]:
        items = filtered()
        clamp_idx()
        out: list[tuple[str, str]] = []
        mode = str(state.get("mode", "A"))
        out.append(("class:title", f"{title}\n\n"))
        out.append(("class:meta", "Use ↑/↓ to choose, number+Enter to pick.\n"))
        out.append(("class:meta", "Press A/L/S/E/T to filter live. q/esc/backspace to cancel.\n"))
        out.append(("class:meta", f"Filter: {mode} ({len(items)} shown / {len(candidates)} total)\n\n"))
        typed = str(state.get("typed", ""))
        if typed:
            out.append(("class:meta", f"Typed index: {typed}\n\n"))
        if not items:
            out.append(("class:edge", "(No files for this filter)\n"))
            return out
        for i, item in enumerate(items, start=1):
            marker = "❯ " if (i - 1) == int(state["idx"]) else "  "
            style = "class:selected" if (i - 1) == int(state["idx"]) else ""
            out.append((style, f"{marker}{i}. {row_label(item)}\n"))
        return out

    style = Style.from_dict(
        {
            "title": "bold",
            "meta": "fg:#888888",
            "selected": "bold fg:#4ade80",
            "edge": "fg:#666666",
        }
    )
    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(render), always_hide_cursor=True)])),
        key_bindings=kb,
        style=style,
        full_screen=False,
    )
    result = int(app.run())
    items = filtered()
    if result < 0 or bool(state.get("cancelled")) or not items:
        raise UserCancelledError("Selection cancelled.")
    if result < 0 or result >= len(items):
        raise UserCancelledError("Selection cancelled.")
    return items[result]


def _pick_from_list(
    candidates: list[Path],
    *,
    title: str,
    display: str,
    labels: Optional[list[str]] = None,
) -> Path:
    if not candidates:
        raise FileNotFoundError(f"No files available for selection: {title}")
    if not sys.stdin.isatty():
        raise UserCancelledError("Interactive selection requires a TTY. Pass input explicitly.")

    rows = labels if labels is not None else [str(p.relative_to(Path.cwd())) for p in candidates]
    idx = _interactive_pick_index(f"{title} ({len(candidates)} found)", rows, display=display)
    return candidates[idx]


def _pick_lse_filter(default_filter: str = "A") -> str:
    if not sys.stdin.isatty():
        return default_filter
    raw = input(f"Filter [A/L/S/E/T] [{default_filter}]: ").strip().upper()
    if raw == "":
        return default_filter
    if raw.lower() in {"q", "quit", "exit"} or raw in {"\x1b", "\x7f", "\b"}:
        raise UserCancelledError("Selection cancelled.")
    if raw not in {"A", "L", "S", "E", "T"}:
        raise UserCancelledError("Selection cancelled (invalid filter).")
    return raw


def _import_target_md_for_source(src: Path, cfg: dict[str, Any]) -> Path:
    import_dir = _cfg_get(cfg, "import.output_dir", None)
    if import_dir:
        candidate = Path(str(import_dir)).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate / f"{src.stem}.md"
    return src.with_suffix(".md")


def _strip_variant_suffix(stem: str) -> str:
    s = stem
    for suf in ("-literal", "-speakable", "-explained", "-explain", "-ted-talk", "-tedtalk", "-summary", "-instructions"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def _artifact_work_dir_for(path: Path) -> Path:
    root = _strip_variant_suffix(path.stem)
    if path.parent.name == root:
        return path.parent
    return path.with_name(root)


def _adapt_status_label(base_md: Path) -> str:
    stem = _strip_variant_suffix(base_md.stem)
    work = _artifact_work_dir_for(base_md)
    literal = (work / f"{stem}-literal.md").exists()
    speakable = (work / f"{stem}-speakable.md").exists()
    explained = (work / f"{stem}-explained.md").exists() or (work / f"{stem}-explain.md").exists()
    ted = (work / f"{stem}-ted-talk.md").exists() or (work / f"{stem}-tedtalk.md").exists()
    return f"L:{'Y' if literal else '-'} S:{'Y' if speakable else '-'} E:{'Y' if explained else '-'} T:{'Y' if ted else '-'}"


def _synth_status_label(derived_md: Path) -> str:
    return f"MP3:{'Y' if derived_md.with_suffix('.mp3').exists() else '-'}"


def _captions_status_label(audio_path: Path) -> str:
    work = _artifact_work_dir_for(audio_path)
    base = work / audio_path.with_suffix("").name
    has_words = Path(str(base) + ".words.json").exists() or Path(str(base) + ".json").exists()
    has_vtt = Path(str(base) + ".vtt").exists()
    return f"W:{'Y' if has_words else '-'} V:{'Y' if has_vtt else '-'}"


def _expand_convert_inputs(raw_inputs: list[str]) -> list[str]:
    out: list[str] = []
    for token in raw_inputs:
        t = str(token).strip()
        if not t:
            continue
        if t.startswith("[") and t.endswith("]"):
            inner = t[1:-1].strip()
            if not inner:
                continue
            parts = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
            out.extend(parts)
            continue
        # Support loose comma-separated input token.
        if "," in t and not Path(t).expanduser().exists():
            parts = [x.strip().strip("'\"") for x in t.split(",") if x.strip()]
            out.extend(parts)
            continue
        out.append(t.strip("'\""))
    return out


def _caption_targets_from_base_token(token: str) -> list[Path]:
    p = Path(token).expanduser()
    base_dir = p.parent if str(p.parent) not in {"", "."} else Path.cwd()
    stem = p.stem if p.suffix.lower() == ".md" else p.name
    # If user gave an already suffixed stem, normalize to its root.
    for suf in ("-literal", "-speakable", "-explained", "-explain", "-summary"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    names = [
        f"{stem}.mp3",
        f"{stem}-literal.mp3",
        f"{stem}-speakable.mp3",
        f"{stem}-explained.mp3",
        f"{stem}-explain.mp3",
        f"{stem}-ted-talk.mp3",
        f"{stem}-tedtalk.mp3",
        f"{stem}-summary.mp3",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for n in names:
        candidate = (base_dir / n).resolve()
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _md_for_audio(audio_path: Path) -> Optional[Path]:
    stem = audio_path.stem
    root = _strip_variant_suffix(stem)
    work = _artifact_work_dir_for(audio_path)
    in_work = work / f"{stem}.md"
    if in_work.exists():
        return in_work
    if stem.lower().endswith("-explain"):
        alt = work / f"{stem[: -len('-explain')]}-explained.md"
        if alt.exists():
            return alt
    if stem.lower().endswith("-explained"):
        alt = work / f"{stem[: -len('-explained')]}-explain.md"
        if alt.exists():
            return alt
    if stem.lower().endswith("-tedtalk"):
        alt = work / f"{stem[: -len('-tedtalk')]}-ted-talk.md"
        if alt.exists():
            return alt
    if stem.lower().endswith("-ted-talk"):
        alt = work / f"{stem[: -len('-ted-talk')]}-tedtalk.md"
        if alt.exists():
            return alt
    if stem.lower().endswith("-literal"):
        base_md = audio_path.with_name(f"{root}.md")
        if base_md.exists():
            return base_md
    candidate = audio_path.with_suffix(".md")
    if candidate.exists():
        return candidate
    return None


def _load_mpx(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid MPX payload: {path}")
    return data


def _cmd_import(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)
    import_input = args.input
    if not import_input:
        exts = {".pdf", ".txt", ".docx"}
        candidates = sorted([p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() in exts])
        if not candidates:
            raise FileNotFoundError(f"No import candidates (*.pdf|*.txt|*.docx) in {Path.cwd()}")
        labels = [f"{p.relative_to(Path.cwd())}  | imported.md:{'Y' if _import_target_md_for_source(p, cfg).exists() else '-'}" for p in candidates]
        import_input = str(
            _pick_from_list(
                candidates,
                title="Select source file to import",
                display=display,
                labels=labels,
            )
        )
    logger.write(1, f"command=import input={import_input}")
    ocr = _resolve(args.ocr, cfg, "import.ocr", "auto")
    output_path: Optional[Path] = Path(args.output).expanduser() if args.output else None
    if output_path is None:
        import_dir = _cfg_get(cfg, "import.output_dir", None)
        if import_dir:
            candidate = Path(str(import_dir)).expanduser()
            if candidate.exists() and candidate.is_dir():
                output_path = candidate / f"{Path(import_input).stem}.md"
            elif verbosity >= 1:
                print(
                    f"Warning: config import.output_dir='{candidate}' not found. "
                    f"Falling back to source folder."
                )
    out = pipeline.import_to_md(
        Path(import_input),
        output_path,
        ocr=ocr,
    )
    logger.write(2, f"import ocr={ocr} output={out.get('md') if isinstance(out, dict) else None}")
    _print(out, verbosity=verbosity, display=display)


def _cmd_adapt(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)
    if args.list_explained_profiles:
        out = {
            "profiles": pipeline.list_explained_prompt_profiles(
                Path(args.explained_prompts_file).expanduser() if args.explained_prompts_file else None
            )
        }
        logger.write(1, "command=adapt list_explained_profiles=true")
        _print(out, verbosity=verbosity, display=display)
        return

    adapt_input = args.input
    if not adapt_input:
        candidates = sorted(
            [
                p
                for p in Path.cwd().iterdir()
                if p.is_file() and p.suffix.lower() == ".md" and not _is_derived_md(p)
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No base .md files found in {Path.cwd()}")
        labels = [f"{p.relative_to(Path.cwd())}  | {_adapt_status_label(p)}" for p in candidates]
        adapt_input = str(
            _pick_from_list(
                candidates,
                title="Select base markdown to adapt",
                display=display,
                labels=labels,
            )
        )

    formats_str = _resolve(args.format, cfg, "adapt.format", "speakable,explained,summary")
    formats = [x.strip().lower() for x in (formats_str or "").split(",") if x.strip()]
    url_mode = _resolve(args.url, cfg, "adapt.url", "mention")
    url_mention = _resolve(args.url_mention, cfg, "adapt.url_mention", "URL provided")
    img_mode = _resolve(args.img, cfg, "adapt.img", "mention")
    img_mention = _resolve(args.img_mention, cfg, "adapt.img_mention", "Image provided")
    explained_model = _resolve(args.explained_model, cfg, "adapt.explained_model", "gpt-4o-mini")
    explained_profile = _resolve(args.explained_profile, cfg, "adapt.explained_profile", "conversational_companion")
    explained_persona = _resolve(args.explained_persona, cfg, "adapt.explained_persona", "Iles")
    explained_back = _resolve(args.explained_back, cfg, "adapt.explained_back", 2)
    explained_forward = _resolve(args.explained_forward, cfg, "adapt.explained_forward", 2)
    no_summary = _resolve(args.no_summary, cfg, "adapt.no_summary", False)
    no_instructions = _resolve(args.no_instructions, cfg, "adapt.no_instructions", False)
    logger.write(
        1,
        f"command=adapt input={adapt_input} format={','.join(formats)} profile={explained_profile}",
    )

    use_rich_progress = display == "rich" and verbosity >= 2
    if not use_rich_progress:
            out = pipeline.adapt_markdown(
                Path(adapt_input).expanduser(),
                formats=formats,
                summary=not bool(no_summary),
                instructions=not bool(no_instructions),
                url_mode=url_mode,
                url_mention=url_mention,
                img_mode=img_mode,
                img_mention=img_mention,
                explained_model=explained_model,
                explained_window=int(_resolve(args.explained_window, cfg, "adapt.explained_window", 2)),
                explained_back=int(explained_back) if explained_back is not None else None,
                explained_forward=int(explained_forward) if explained_forward is not None else None,
                explained_persona=explained_persona,
                explained_profile=explained_profile,
                explained_prompts_file=Path(args.explained_prompts_file).expanduser() if args.explained_prompts_file else None,
                pronunciation_file=Path(args.pronunciation_file).expanduser() if args.pronunciation_file else None,
            )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            tasks: dict[str, int] = {}

            def progress_cb(stage: str, current: int, total: int, message: str) -> None:
                key = "adapt_main" if stage == "adapt" else stage
                if key not in tasks:
                    tasks[key] = progress.add_task(message, total=max(total, 1))
                progress.update(tasks[key], completed=max(0, min(current, max(total, 1))), total=max(total, 1), description=message)
                logger.write(3, f"progress stage={stage} {current}/{total} msg={message}")

            def info_cb(message: str) -> None:
                console.log(message)
                logger.write(2, message)

            out = pipeline.adapt_markdown(
                Path(adapt_input).expanduser(),
                formats=formats,
                summary=not bool(no_summary),
                instructions=not bool(no_instructions),
                url_mode=url_mode,
                url_mention=url_mention,
                img_mode=img_mode,
                img_mention=img_mention,
                explained_model=explained_model,
                explained_window=int(_resolve(args.explained_window, cfg, "adapt.explained_window", 2)),
                explained_back=int(explained_back) if explained_back is not None else None,
                explained_forward=int(explained_forward) if explained_forward is not None else None,
                explained_persona=explained_persona,
                explained_profile=explained_profile,
                explained_prompts_file=Path(args.explained_prompts_file).expanduser() if args.explained_prompts_file else None,
                pronunciation_file=Path(args.pronunciation_file).expanduser() if args.pronunciation_file else None,
                progress_cb=progress_cb,
                info_cb=info_cb,
            )
    _print(out, verbosity=verbosity, display=display)


def _cmd_synth(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)

    quality = _resolve(args.quality, cfg, "synth.quality", "sd")
    synth_model = _resolve(args.model, cfg, "synth.model", None) or ("tts-1-hd" if quality == "hd" else "tts-1")
    format_spec = _resolve(args.format, cfg, "synth.format", None)
    read_headers = _resolve(args.read_headers, cfg, "synth.read_headers", "none")
    voice = _resolve(args.voice, cfg, "synth.voice", None)
    voices_file = _resolve(args.voices_file, cfg, "synth.voices_file", None)
    url_policy = _resolve(args.url, cfg, "synth.url", "drop")
    url_mention = _resolve(args.url_mention, cfg, "synth.url_mention", "URL provided")
    img_policy = _resolve(args.img, cfg, "synth.img", "none")
    img_mention = _resolve(args.img_mention, cfg, "synth.img_mention", "Image provided")
    no_sidecar = _resolve(args.no_sidecar, cfg, "synth.no_sidecar", False)
    request_timeout_seconds = float(_resolve(args.request_timeout_seconds, cfg, "synth.request_timeout_seconds", 90))
    max_retries = int(_resolve(args.max_retries, cfg, "synth.max_retries", 3))
    multi_thread = int(_resolve(args.multi_thread, cfg, "synth.multi_thread", 1))
    synth_input = args.input
    if not synth_input:
        candidates = sorted(
            [
                p
                for p in Path.cwd().rglob("*.md")
                if p.is_file() and p.suffix.lower() == ".md" and _is_derived_md(p)
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No adapted .md files found in {Path.cwd()}")
        chosen = _interactive_pick_path_with_lse_filter(
            candidates,
            title="Select adapted markdown to synth",
            row_label=lambda p: f"{p.relative_to(Path.cwd())}  | {_synth_status_label(p)}",
        )
        synth_input = str(
            chosen
        )
    logger.write(
        1,
        f"command=synth input={synth_input} format={format_spec or 'auto'} model={synth_model} mt={multi_thread}",
    )

    use_rich_progress = display == "rich" and verbosity >= 2
    if not use_rich_progress:
        out = pipeline.synth(
            Path(synth_input).expanduser(),
            output_path=Path(args.output).expanduser() if args.output else None,
            format_spec=format_spec,
            read_headers=read_headers,
            voice=voice,
            voices_file=voices_file,
            model=synth_model,
            url_policy=url_policy,
            url_placeholder=url_mention,
            img_policy=img_policy,
            img_placeholder=img_mention,
            emit_sidecar=not bool(no_sidecar),
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            multi_thread=multi_thread,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            tasks: dict[str, int] = {}

            def progress_cb(stage: str, current: int, total: int, message: str) -> None:
                key = stage
                if key not in tasks:
                    tasks[key] = progress.add_task(message, total=max(total, 1))
                progress.update(tasks[key], completed=max(0, min(current, max(total, 1))), total=max(total, 1), description=message)
                logger.write(3, f"progress stage={stage} {current}/{total} msg={message}")

            def info_cb(message: str) -> None:
                console.log(message)
                logger.write(2, message)

            out = pipeline.synth(
                Path(synth_input).expanduser(),
                output_path=Path(args.output).expanduser() if args.output else None,
                format_spec=format_spec,
                read_headers=read_headers,
                voice=voice,
                voices_file=voices_file,
                model=synth_model,
                url_policy=url_policy,
                url_placeholder=url_mention,
                img_policy=img_policy,
                img_placeholder=img_mention,
                emit_sidecar=not bool(no_sidecar),
                request_timeout_seconds=request_timeout_seconds,
                max_retries=max_retries,
                multi_thread=multi_thread,
                progress_cb=progress_cb,
                info_cb=info_cb,
            )
    _print(out, verbosity=verbosity, display=display)


def _cmd_captions(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)
    caption_targets: list[Path] = []
    if args.audio:
        audio_input = Path(args.audio).expanduser()
        if audio_input.exists() and audio_input.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}:
            caption_targets = [audio_input]
        else:
            caption_targets = _caption_targets_from_base_token(args.audio)
            if not caption_targets:
                raise FileNotFoundError(
                    f"No caption audio targets found for '{args.audio}'. "
                    "Provide an existing audio file or base name with matching MP3 variants."
                )
    else:
        candidates = sorted(
            [
                p
                for p in Path.cwd().iterdir()
                if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No caption audio files found in {Path.cwd()}")
        chosen = _interactive_pick_path_with_lse_filter(
            candidates,
            title="Select audio for captions",
            row_label=lambda p: f"{p.relative_to(Path.cwd())}  | {_captions_status_label(p)}",
        )
        caption_targets = [
            chosen
        ]

    logger.write(
        1,
        f"command=captions targets={len(caption_targets)} local={args.local} cloud={args.cloud}",
    )

    use_rich_progress = display == "rich" and verbosity >= 2
    outputs: list[dict[str, Any]] = []
    if use_rich_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id: int | None = None

            def progress_cb(stage: str, current: int, total: int, message: str) -> None:
                nonlocal task_id
                if task_id is None:
                    task_id = progress.add_task(message, total=max(total, 1))
                progress.update(task_id, completed=max(0, min(current, max(total, 1))), total=max(total, 1), description=message)
                logger.write(3, f"progress stage={stage} {current}/{total} msg={message}")

            def info_cb(message: str) -> None:
                console.log(message)
                logger.write(2, message)

            for audio_path in caption_targets:
                md_for_target = Path(args.md).expanduser() if args.md else _md_for_audio(audio_path)
                out = pipeline.captions(
                    audio_path,
                    output_base=Path(args.output).expanduser() if args.output and len(caption_targets) == 1 else None,
                    local=args.local,
                    cloud=args.cloud,
                    model=args.model,
                    locale=args.locale,
                    on_device=args.on_device,
                    keep_wav=args.keep_wav,
                    overwrite_wav=args.overwrite_wav,
                    md_path=md_for_target,
                    cooldown=float(args.cooldown),
                    verify=args.verify,
                    verify_threshold=float(args.verify_threshold),
                    verify_max_start_seconds=float(args.verify_max_start_seconds),
                    verify_max_start_ratio=float(args.verify_max_start_ratio),
                    progress_cb=progress_cb,
                    info_cb=info_cb,
                )
                outputs.append({"audio": str(audio_path), **(out if isinstance(out, dict) else {"result": out})})
    else:
        for audio_path in caption_targets:
            md_for_target = Path(args.md).expanduser() if args.md else _md_for_audio(audio_path)
            out = pipeline.captions(
                audio_path,
                output_base=Path(args.output).expanduser() if args.output and len(caption_targets) == 1 else None,
                local=args.local,
                cloud=args.cloud,
                model=args.model,
                locale=args.locale,
                on_device=args.on_device,
                keep_wav=args.keep_wav,
                overwrite_wav=args.overwrite_wav,
                md_path=md_for_target,
                cooldown=float(args.cooldown),
                verify=args.verify,
                verify_threshold=float(args.verify_threshold),
                verify_max_start_seconds=float(args.verify_max_start_seconds),
                verify_max_start_ratio=float(args.verify_max_start_ratio),
            )
            outputs.append({"audio": str(audio_path), **(out if isinstance(out, dict) else {"result": out})})

    final_out: Any = outputs[0] if len(outputs) == 1 else {"outputs": outputs}
    _print(final_out, verbosity=verbosity, display=display)


def _cmd_convert(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)

    inputs = _expand_convert_inputs(list(args.inputs or []))
    if not inputs:
        raise ValueError("convert requires at least one input file")

    format_spec = _resolve(args.format, cfg, "convert.format", "s")
    if args.captions:
        captions_enabled = True
    elif args.no_caption:
        captions_enabled = False
    else:
        captions_enabled = bool(_cfg_get(cfg, "convert.captions", False))
    jobs = max(1, int(_resolve(args.jobs, cfg, "convert.jobs", 1)))
    ocr_mode = _resolve(args.ocr, cfg, "convert.ocr", "auto")

    # Reuse existing adapt/synth defaults.
    formats = pipeline.parse_formats(format_spec)
    adapt_formats = [f for f in formats if f in {"speakable", "explained", "summary"}]
    summary_enabled = "summary" in formats

    url_mode = _cfg_get(cfg, "adapt.url", "mention")
    url_mention = _cfg_get(cfg, "adapt.url_mention", "URL provided")
    img_mode = _cfg_get(cfg, "adapt.img", "mention")
    img_mention = _cfg_get(cfg, "adapt.img_mention", "Image provided")
    explained_model = _cfg_get(cfg, "adapt.explained_model", "gpt-4o-mini")
    explained_profile = _cfg_get(cfg, "adapt.explained_profile", "conversational_companion")
    explained_persona = _cfg_get(cfg, "adapt.explained_persona", "Iles")
    explained_back = int(_cfg_get(cfg, "adapt.explained_back", 2))
    explained_forward = int(_cfg_get(cfg, "adapt.explained_forward", 2))

    quality = _cfg_get(cfg, "synth.quality", "hd")
    synth_model = _cfg_get(cfg, "synth.model", None) or ("tts-1-hd" if quality == "hd" else "tts-1")
    read_headers = _cfg_get(cfg, "synth.read_headers", "none")
    voice = _cfg_get(cfg, "synth.voice", "nova")
    voices_file = _cfg_get(cfg, "synth.voices_file", None)
    synth_url = _cfg_get(cfg, "synth.url", "drop")
    synth_url_mention = _cfg_get(cfg, "synth.url_mention", "URL provided")
    synth_img = _cfg_get(cfg, "synth.img", "none")
    synth_img_mention = _cfg_get(cfg, "synth.img_mention", "Image provided")
    no_sidecar = bool(_cfg_get(cfg, "synth.no_sidecar", False))
    request_timeout_seconds = float(_cfg_get(cfg, "synth.request_timeout_seconds", 90))
    max_retries = int(_cfg_get(cfg, "synth.max_retries", 3))
    multi_thread = int(_cfg_get(cfg, "synth.multi_thread", 1))

    logger.write(
        1,
        f"command=convert files={len(inputs)} format={format_spec} captions={captions_enabled} jobs={jobs}",
    )

    def run_one(input_token: str) -> dict[str, Any]:
        try:
            p = Path(input_token).expanduser()
            if not p.exists():
                raise FileNotFoundError(p)
            ext = p.suffix.lower()
            if ext not in {".md", ".pdf", ".txt", ".doc", ".docx", ".rtf"}:
                raise ValueError(f"Unsupported convert input type: {ext}")

            imported: Optional[dict[str, Any]] = None
            md_path = p
            if ext != ".md":
                imported = pipeline.import_to_md(p, None, ocr=ocr_mode)
                md_path = Path(str(imported["md"])).expanduser()

            adapted = pipeline.adapt_markdown(
                md_path,
                formats=adapt_formats,
                summary=summary_enabled,
                instructions=False,
                url_mode=str(url_mode),
                url_mention=str(url_mention),
                img_mode=str(img_mode),
                img_mention=str(img_mention),
                explained_model=str(explained_model),
                explained_window=2,
                explained_back=explained_back,
                explained_forward=explained_forward,
                explained_persona=str(explained_persona),
                explained_profile=str(explained_profile),
                explained_prompts_file=None,
                pronunciation_file=None,
            )

            synth_out = pipeline.synth(
                md_path,
                output_path=None,
                format_spec=format_spec,
                read_headers=str(read_headers),
                voice=str(voice) if voice is not None else None,
                voices_file=str(voices_file) if voices_file else None,
                model=str(synth_model),
                url_policy=str(synth_url),
                url_placeholder=str(synth_url_mention),
                img_policy=str(synth_img),
                img_placeholder=str(synth_img_mention),
                emit_sidecar=not bool(no_sidecar),
                request_timeout_seconds=request_timeout_seconds,
                max_retries=max_retries,
                multi_thread=multi_thread,
            )

            captions_out: list[dict[str, Any]] = []
            if captions_enabled:
                for s in synth_out.get("outputs", []):
                    mp3 = Path(str(s.get("mp3"))).expanduser()
                    cap = pipeline.captions(
                        mp3,
                        output_base=None,
                        local=True,
                        cloud=False,
                        model="whisper-1",
                        locale="en-US",
                        on_device=True,
                        keep_wav=False,
                        overwrite_wav=False,
                        md_path=_md_for_audio(mp3),
                        cooldown=15.0,
                        verify=False,
                    )
                    captions_out.append({"audio": str(mp3), **cap})

            return {
                "input": str(p),
                "import": imported,
                "adapt": adapted,
                "synth": synth_out,
                "captions": captions_out,
            }
        except Exception as e:
            return {"input": str(input_token), "error": f"{type(e).__name__}: {e}"}

    results: list[dict[str, Any]] = []
    if jobs <= 1 or len(inputs) <= 1:
        for x in inputs:
            results.append(run_one(x))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, x) for x in inputs]
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [r for r in results if r.get("error")]
    logger.write(1, f"convert complete ok={len(results)-len(failed)} failed={len(failed)}")
    _print({"outputs": results, "failed": len(failed)}, verbosity=verbosity, display=display)


def _cmd_mp4(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)
    fmt = _resolve(args.format, cfg, "mp4.format", "s")
    width = int(_resolve(args.width, cfg, "mp4.width", 1280))
    height = int(_resolve(args.height, cfg, "mp4.height", 720))
    fps = int(_resolve(args.fps, cfg, "mp4.fps", 30))
    font_size = int(_resolve(args.font_size, cfg, "mp4.font_size", 72))
    in_path = Path(args.input).expanduser()
    if in_path.suffix.lower() == ".mpx":
        payload = _load_mpx(in_path)
        words_list = [Path(x) for x in ((payload.get("artifacts") or {}).get("words_json") or [])]
        words_list = [x for x in words_list if x.exists()]
        if not words_list:
            raise FileNotFoundError(f"No .words.json entries found in {in_path}")
        if len(words_list) == 1:
            in_path = words_list[0]
        else:
            selected = _pick_from_list(words_list, title="Select words timeline for MP4", display=display)
            in_path = selected
    logger.write(1, f"command=mp4 input={in_path} format={fmt} size={width}x{height} fps={fps}")
    out = pipeline.make_mp4(
        in_path,
        format_spec=str(fmt),
        output_path=Path(args.output).expanduser() if args.output else None,
        width=width,
        height=height,
        fps=fps,
        font_size=font_size,
    )
    _print(out, verbosity=verbosity, display=display)


def _cmd_play(args: argparse.Namespace) -> None:
    cfg = _load_config()
    logger = _setup_logger(args, cfg)
    fmt = _resolve(args.format, cfg, "mp4.format", "s")
    p = Path(args.input).expanduser()

    target: Optional[Path] = None
    if p.suffix.lower() == ".mp4":
        target = p
    elif p.suffix.lower() == ".md":
        aliases = {"l": "literal", "s": "speakable", "e": "explained", "t": "ted-talk", "m": "summary", "a": "speakable"}
        f = aliases.get(str(fmt).strip().lower(), str(fmt).strip().lower())
        if f not in {"literal", "speakable", "explained", "ted-talk", "summary"}:
            f = "speakable"
        md_variant = pipeline.resolve_variant_path(p, f)
        target = md_variant.with_suffix(".mp4")
        if not target.exists():
            root = _strip_variant_suffix(p.stem)
            work = _artifact_work_dir_for(p)
            if f == "literal":
                fallback = work / f"{root}-literal.mp4"
            else:
                fallback = work / f"{root}-{f}.mp4"
            if fallback.exists():
                target = fallback
    elif p.suffix.lower() == ".mpx":
        payload = _load_mpx(p)
        mp4s = [Path(x) for x in ((payload.get("artifacts") or {}).get("mp4") or [])]
        mp4s = [x for x in mp4s if x.exists()]
        if mp4s:
            target = mp4s[0]
    else:
        if p.with_suffix(".mp4").exists():
            target = p.with_suffix(".mp4")

    if not target or not target.exists():
        raise FileNotFoundError(f"Playable MP4 not found for input: {args.input}")

    logger.write(1, f"command=play target={target}")
    opener = shutil.which("open") or shutil.which("xdg-open")
    if not opener:
        raise RuntimeError("No system opener found (expected 'open' on macOS or 'xdg-open' on Linux).")
    subprocess.run([opener, str(target)], check=True)


def _cmd_read(args: argparse.Namespace) -> None:
    cfg = _load_config()
    display = _display_mode(args, cfg)
    verbosity = _verbosity(args, cfg)
    logger = _setup_logger(args, cfg)
    read_input = args.input
    if not read_input:
        mpx_candidates = sorted(Path.cwd().glob("*.mpx"))
        if mpx_candidates:
            selected_mpx = _pick_from_list(mpx_candidates, title="Select .mpx project", display=display)
            payload = _load_mpx(selected_mpx)
            _print(payload, verbosity=verbosity, display=display)
            words_list = [Path(p) for p in ((payload.get("artifacts") or {}).get("words_json") or [])]
            words_list = [p for p in words_list if p.exists()]
            if not words_list:
                raise FileNotFoundError(f"No .words.json entries found in {selected_mpx}")
            selected = _pick_from_list(words_list, title="Select words timeline from MPX", display=display)
            read_input = str(selected)
        else:
            candidates = sorted(Path.cwd().rglob("*.words.json"))
            if not candidates:
                raise FileNotFoundError(f"No .words.json files found under {Path.cwd()}")
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "No read input provided in non-interactive mode. "
                    "Pass an input path (audio/base/.words.json/.mpx)."
                )
            selected = _pick_from_list(candidates, title="Select .words.json file", display=display)
            read_input = str(selected)
        logger.write(1, f"command=read selected={read_input}")
    else:
        p = Path(read_input).expanduser()
        if p.suffix.lower() == ".mpx":
            payload = _load_mpx(p)
            _print(payload, verbosity=verbosity, display=display)
            words_list = [Path(x) for x in ((payload.get("artifacts") or {}).get("words_json") or [])]
            words_list = [x for x in words_list if x.exists()]
            if not words_list:
                raise FileNotFoundError(f"No .words.json entries found in {p}")
            selected = _pick_from_list(words_list, title="Select words timeline from MPX", display=display)
            read_input = str(selected)
        logger.write(1, f"command=read input={read_input}")

    runtime = _load_runtime_state()
    runtime_read = runtime.get("read", {}) if isinstance(runtime.get("read"), dict) else {}

    cfg_window = int(_cfg_get(cfg, "read.window", 8))
    cfg_speed = int(_cfg_get(cfg, "read.speed_preset", 2))
    rt_window = int(runtime_read.get("window", cfg_window))
    rt_speed = int(runtime_read.get("speed_preset", cfg_speed))

    result = pipeline.read(
        Path(read_input).expanduser(),
        seconds=float(args.seconds) if args.seconds is not None else 0.0,
        window=int(args.window) if args.window is not None else rt_window,
        speed_preset=int(args.speed) if args.speed is not None else rt_speed,
    )
    runtime["read"] = {
        "window": int(result.get("window", rt_window)) if isinstance(result, dict) else rt_window,
        "speed_preset": int(result.get("speed_preset", rt_speed)) if isinstance(result, dict) else rt_speed,
    }
    saved = _save_runtime_state(runtime)
    logger.write(2, f"read runtime state saved: {saved}")


def _cmd_config(args: argparse.Namespace) -> None:
    cfg = _load_config()
    logger = _setup_logger(args, cfg)
    logger.write(1, f"command=config action={args.config_action}")
    if args.config_action == "path":
        print(_config_path())
        return
    if args.config_action == "source":
        print(json.dumps(_config_source_meta(), indent=2))
        return
    if args.config_action == "show":
        print(json.dumps(cfg, indent=2))
        print("\nHow to change settings:")
        print("  tts3x config set <dotted.key> <value>")
        print("  tts3x config get <dotted.key>")
        print("  tts3x config source")
        print("\nExamples:")
        print("  tts3x config set synth.quality hd")
        print("  tts3x config set synth.multi_thread 5")
        print("  tts3x config set import.ocr auto")
        print("  tts3x config set global.display rich")
        print("  tts3x config set global.verbose 2")
        print("  tts3x config set global.logging 2")
        print("\nEditable keys:")
        for k in sorted(_cfg_flatten_keys(cfg)):
            print(f"  - {k}")
        return
    if args.config_action == "get":
        val = _cfg_get(cfg, args.key, None)
        print(json.dumps(val, indent=2))
        return
    if args.config_action == "set":
        val = _coerce_scalar(args.value)
        _cfg_set(cfg, args.key, val)
        path = _save_config(cfg)
        print(f"Saved {args.key} in {path}")
        return
    raise ValueError(f"Unknown config action: {args.config_action}")


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("-d", "--display", choices=["rich", "normal", "r", "n"], default=None, help="Display style")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=None, help="Verbosity level")
    parser.add_argument("-v0", action="store_true", help="Verbosity 0 (silent)")
    parser.add_argument("-v1", action="store_true", help="Verbosity 1 (minimal)")
    parser.add_argument("-v2", action="store_true", help="Verbosity 2 (default info)")
    parser.add_argument("-v3", action="store_true", help="Verbosity 3 (debug)")
    parser.add_argument("--logging", type=int, choices=[0, 1, 2, 3], default=None, help="File logging level")
    parser.add_argument("-l0", action="store_true", help="Logging level 0 (off)")
    parser.add_argument("-l1", action="store_true", help="Logging level 1")
    parser.add_argument("-l2", action="store_true", help="Logging level 2")
    parser.add_argument("-l3", action="store_true", help="Logging level 3")
    parser.add_argument("--logging-file", default=None, help="Log file path or folder")
    parser.add_argument("--logging-clear", action="store_true", help="Clear log file before writing")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tts3x", description="TTS3 Extended (tts3x)")
    _add_common_options(p)

    sub = p.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="Import PDF/DOCX/TXT/MD into markdown")
    _add_common_options(imp)
    imp.add_argument("input", nargs="?")
    imp.add_argument("--output", "-o")
    imp.add_argument("--ocr", choices=["auto", "force", "never"], default=None)
    imp.set_defaults(func=_cmd_import)

    ad = sub.add_parser("adapt", help="Create speakable/explained/summary markdown variants")
    _add_common_options(ad)
    ad.add_argument("input", nargs="?")
    ad.add_argument("--format", default=None, help="Comma list: speakable,explained,ted-talk,summary")
    ad.add_argument("--url", choices=["none", "mention", "full"], default=None)
    ad.add_argument("--url-mention", default=None)
    ad.add_argument("--img", choices=["none", "mention"], default=None)
    ad.add_argument("--img-mention", default=None)
    ad.add_argument("--explained-model", default=None)
    ad.add_argument("--explained-window", default=None, help="Paragraph context on each side for explained generation")
    ad.add_argument("--explained-back", help="Previous paragraph count for explained context (overrides --explained-window)")
    ad.add_argument("--explained-forward", help="Next paragraph count for explained context (overrides --explained-window)")
    ad.add_argument("--explained-persona", default=None, help="Listener/persona name used in explained prompt")
    ad.add_argument("--explained-profile", default=None, help="Prompt profile for explained generation")
    ad.add_argument("--explained-prompts-file", help="Path to custom explained prompt profiles JSON")
    ad.add_argument("--pronunciation-file", help="Path to pronunciation mapping JSON (e.g., Iles -> eye-ulls)")
    ad.add_argument("--list-explained-profiles", action="store_true", help="List available explained prompt profiles and exit")
    ad.add_argument("--no-summary", action="store_true", default=None)
    ad.add_argument("--no-instructions", action="store_true", default=None)
    ad.set_defaults(func=_cmd_adapt)

    sy = sub.add_parser("synth", help="Synthesize markdown variants to MP3")
    _add_common_options(sy)
    sy.add_argument("input", nargs="?")
    sy.add_argument("--output", help="Output mp3 or output prefix")
    sy.add_argument("--format", help="literal|speakable|explained|ted-talk|summary|all or l|s|e|t|m|a")
    sy.add_argument("--read-headers", default=None, help="none|1|2|3")
    sy.add_argument("--voice", default=None, help="Default voice, voice mapping, or voices json path")
    sy.add_argument("--voices-file", help="Path to voice map json")
    sy.add_argument("--quality", choices=["sd", "hd"], default=None, help="Synth quality preset: sd=tts-1, hd=tts-1-hd")
    sy.add_argument("--model", help="Explicit model override (e.g., tts-1, tts-1-hd)")
    sy.add_argument("--url", choices=["drop", "placeholder", "keep"], default=None)
    sy.add_argument("--url-mention", default=None)
    sy.add_argument("--img", choices=["none", "mention"], default=None)
    sy.add_argument("--img-mention", default=None)
    sy.add_argument("--no-sidecar", action="store_true", default=None)
    sy.add_argument("--request-timeout-seconds", default=None, help="Per synth chunk request timeout in seconds")
    sy.add_argument("--max-retries", default=None, help="Retries per synth chunk")
    sy.add_argument("-mt", "--multi-thread", default=None, help="Number of parallel chunk synth workers (default: 1)")
    sy.set_defaults(func=_cmd_synth)

    ca = sub.add_parser("captions", help="Generate words.json + vtt (+ enriched timeline json)")
    _add_common_options(ca)
    ca.add_argument("audio", nargs="?")
    ca.add_argument("--output", help="Output base path")
    ca.add_argument("--local", action="store_true", help="Local mode (default)")
    ca.add_argument("--cloud", action="store_true", help="Cloud STT")
    ca.add_argument("--model", default="whisper-1")
    ca.add_argument("--locale", default="en-US")
    ca.add_argument("--on-device", action="store_true", help="Apple Speech: prefer on-device recognition when available")
    ca.add_argument("--keep-wav", action="store_true")
    ca.add_argument("--overwrite-wav", action="store_true")
    ca.add_argument("--md", help="Markdown source for enrichment")
    ca.add_argument("--cooldown", default="15")
    ca.add_argument("--verify", action="store_true", help="Fail if timing coverage is below threshold")
    ca.add_argument("--verify-threshold", default="85", help="Coverage threshold (percent or ratio). Default: 85")
    ca.add_argument("--verify-max-start-seconds", default="10", help="Fail if first word starts after this many seconds (default: 10)")
    ca.add_argument("--verify-max-start-ratio", default="0.02", help="Fail if first word starts after this fraction of duration (default: 0.02)")
    ca.set_defaults(func=_cmd_captions)

    cv = sub.add_parser("convert", help="End-to-end convert: import/adapt/synth/(optional captions)")
    _add_common_options(cv)
    cv.add_argument("inputs", nargs="+", help="One or more input files (.md/.pdf/.txt/.docx)")
    cv.add_argument("--format", default=None, help="Output format(s): l|s|e|t|m|a or literal,speakable,explained,ted-talk,summary,all")
    cv.add_argument("--captions", action="store_true", help="Also generate captions (words.json + vtt)")
    cv.add_argument("--no-caption", action="store_true", help="Do not generate captions")
    cv.add_argument("--jobs", "-j", default=None, help="Parallel file workers for multi-file convert")
    cv.add_argument("--ocr", choices=["auto", "force", "never"], default=None, help="OCR mode for non-md inputs")
    cv.set_defaults(func=_cmd_convert)

    v4 = sub.add_parser("mp4", help="Generate MP4 from audio + words timeline (centered word highlight)")
    _add_common_options(v4)
    v4.add_argument("input", help="Input .md/.mp3/.words.json/.mpx/base path")
    v4.add_argument("--format", default=None, help="When input is .md: l|s|e|t|m (literal/speakable/explained/ted-talk/summary)")
    v4.add_argument("--output", help="Output .mp4 path")
    v4.add_argument("--width", default=None, help="Video width (default 1280)")
    v4.add_argument("--height", default=None, help="Video height (default 720)")
    v4.add_argument("--fps", default=None, help="Video fps (default 30)")
    v4.add_argument("--font-size", default=None, help="Caption font size (default 72)")
    v4.set_defaults(func=_cmd_mp4)

    pl = sub.add_parser("play", help="Play generated MP4 for .md/.mp4/.mpx inputs")
    _add_common_options(pl)
    pl.add_argument("input", help="Input .md/.mp4/.mpx/base path")
    pl.add_argument("--format", default=None, help="When input is .md: l|s|e|t|m (default from config)")
    pl.set_defaults(func=_cmd_play)

    rd = sub.add_parser("read", help="Play audio and show focused word timeline in terminal")
    _add_common_options(rd)
    rd.add_argument("input", nargs="?", help="Path to audio (.mp3/.wav), .words.json, .mpx, or base source path")
    rd.add_argument("--seconds", help="Start time in seconds", default=None)
    rd.add_argument("--window", help="Words of context around focus word", default=None)
    rd.add_argument("--speed", choices=[str(i) for i in range(1, 9)], default=None, help="Speed preset: 1=1x, 2=1.25x, 3=1.5x, 4=2x, 5=2.25x, 6=2.5x, 7=3x, 8=4x")
    rd.set_defaults(func=_cmd_read)

    cfg = sub.add_parser("config", help="Show or update tts3x defaults config")
    _add_common_options(cfg)
    cfg_sub = cfg.add_subparsers(dest="config_action", required=True)
    cfg_path = cfg_sub.add_parser("path", help="Show config file path")
    _add_common_options(cfg_path)
    cfg_source = cfg_sub.add_parser("source", help="Show active config source/provider resolution")
    _add_common_options(cfg_source)
    cfg_show = cfg_sub.add_parser("show", help="Show effective config")
    _add_common_options(cfg_show)
    cfg_get = cfg_sub.add_parser("get", help="Get config value by dotted path")
    _add_common_options(cfg_get)
    cfg_get.add_argument("key")
    cfg_set = cfg_sub.add_parser("set", help="Set config value by dotted path")
    _add_common_options(cfg_set)
    cfg_set.add_argument("key")
    cfg_set.add_argument("value")
    cfg.set_defaults(func=_cmd_config)

    return p


def _infer_command_from_file_token(token: str) -> Optional[str]:
    lower = token.lower()
    if lower.endswith(".mp4"):
        return "play"
    if lower.endswith(".mpx"):
        return "read"
    if lower.endswith(".words.json"):
        return "read"
    if lower.endswith((".pdf", ".doc", ".docx", ".txt", ".rtf")):
        return "import"
    if lower.endswith(".md"):
        return "synth"
    if lower.endswith((".mp3", ".wav", ".m4a", ".aac")):
        return "captions"
    return None


def _argv_with_inferred_command(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    known = {"import", "adapt", "synth", "captions", "convert", "mp4", "play", "read", "config"}
    for token in argv:
        if token.startswith("-"):
            continue
        if token in known:
            return argv
        inferred = _infer_command_from_file_token(token)
        if inferred:
            return [inferred, *argv]
        return argv
    return argv


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args(_argv_with_inferred_command(sys.argv[1:]))
        args.func(args)
    except UserCancelledError as e:
        print(str(e))
        raise SystemExit(1)
    except RuntimeError as e:
        msg = str(e)
        if msg.lower().startswith("selection cancelled"):
            print(msg)
            raise SystemExit(1)
        raise
    except KeyboardInterrupt:
        print("Cancelled.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
