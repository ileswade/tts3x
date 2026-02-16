from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from . import pipeline


class TTS3X:
    """Programmatic API over pipeline functions for manager/MCP integration."""

    def import_file(self, input_path: str, output_path: Optional[str] = None, ocr: str = "auto") -> Dict[str, Any]:
        return pipeline.import_to_md(Path(input_path).expanduser(), Path(output_path).expanduser() if output_path else None, ocr=ocr)

    def adapt(
        self,
        input_md: str,
        *,
        format: str = "speakable,explained,summary",
        summary: bool = True,
        instructions: bool = True,
        url: str = "mention",
        url_mention: str = "URL provided",
        img: str = "mention",
        img_mention: str = "Image provided",
        explained_model: str = "gpt-4o-mini",
        explained_window: int = 2,
        explained_back: Optional[int] = None,
        explained_forward: Optional[int] = None,
        explained_persona: str = "Iles",
        explained_profile: str = "conversational_companion",
        explained_prompts_file: Optional[str] = None,
        pronunciation_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        formats = [x.strip().lower() for x in (format or "").split(",") if x.strip()]
        return pipeline.adapt_markdown(
            Path(input_md).expanduser(),
            formats=formats,
            summary=summary,
            instructions=instructions,
            url_mode=url,
            url_mention=url_mention,
            img_mode=img,
            img_mention=img_mention,
            explained_model=explained_model,
            explained_window=int(explained_window),
            explained_back=int(explained_back) if explained_back is not None else None,
            explained_forward=int(explained_forward) if explained_forward is not None else None,
            explained_persona=explained_persona,
            explained_profile=explained_profile,
            explained_prompts_file=Path(explained_prompts_file).expanduser() if explained_prompts_file else None,
            pronunciation_file=Path(pronunciation_file).expanduser() if pronunciation_file else None,
        )

    def list_explained_profiles(self, *, explained_prompts_file: Optional[str] = None) -> Dict[str, str]:
        return pipeline.list_explained_prompt_profiles(
            Path(explained_prompts_file).expanduser() if explained_prompts_file else None
        )

    def synth(
        self,
        input_path: str,
        *,
        output_path: Optional[str] = None,
        format: Optional[str] = None,
        read_headers: str = "none",
        voice: Optional[str] = None,
        voices_file: Optional[str] = None,
        model: str = "tts-1",
        url: str = "drop",
        url_mention: str = "URL provided",
        img: str = "none",
        img_mention: str = "Image provided",
        sidecar: bool = True,
    ) -> Dict[str, Any]:
        return pipeline.synth(
            Path(input_path).expanduser(),
            output_path=Path(output_path).expanduser() if output_path else None,
            format_spec=format,
            read_headers=read_headers,
            voice=voice,
            voices_file=voices_file,
            model=model,
            url_policy=url,
            url_placeholder=url_mention,
            img_policy=img,
            img_placeholder=img_mention,
            emit_sidecar=sidecar,
        )

    def captions(
        self,
        audio_path: str,
        *,
        output: Optional[str] = None,
        local: bool = False,
        cloud: bool = False,
        model: str = "whisper-1",
        locale: str = "en-US",
        on_device: bool = False,
        keep_wav: bool = False,
        overwrite_wav: bool = False,
        md_path: Optional[str] = None,
        cooldown: float = 15.0,
        verify: bool = False,
        verify_threshold: float = 85.0,
        verify_max_start_seconds: float = 10.0,
        verify_max_start_ratio: float = 0.02,
        progress_cb=None,
        info_cb=None,
    ) -> Dict[str, Any]:
        return pipeline.captions(
            Path(audio_path).expanduser(),
            output_base=Path(output).expanduser() if output else None,
            local=local,
            cloud=cloud,
            model=model,
            locale=locale,
            on_device=on_device,
            keep_wav=keep_wav,
            overwrite_wav=overwrite_wav,
            md_path=Path(md_path).expanduser() if md_path else None,
            cooldown=float(cooldown),
            verify=verify,
            verify_threshold=float(verify_threshold),
            verify_max_start_seconds=float(verify_max_start_seconds),
            verify_max_start_ratio=float(verify_max_start_ratio),
            progress_cb=progress_cb,
            info_cb=info_cb,
        )

    def read(self, input_path: str, *, seconds: float = 0.0, window: int = 8) -> None:
        pipeline.read(Path(input_path).expanduser(), seconds=float(seconds), window=int(window))
