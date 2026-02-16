from pathlib import Path

from tts3x.pipeline import detect_derived, parse_formats, parse_markdown_for_synth


def test_detect_derived():
    assert detect_derived(Path("chapter-speakable.md")) == "speakable"
    assert detect_derived(Path("chapter-explained.md")) == "explained"
    assert detect_derived(Path("chapter-summary.md")) == "summary"
    assert detect_derived(Path("chapter.md")) is None


def test_parse_formats_aliases():
    assert parse_formats(None) == ["literal", "speakable", "explained", "summary"]
    assert parse_formats("s") == ["speakable"]
    assert parse_formats("l,e") == ["literal", "explained"]


def test_markdown_parsing_keeps_header_meta_and_suppresses_header_speech():
    md = "# H1\n\n## H2\n\nParagraph with [ref](https://example.com) and image ![x](img.png)."
    paras = parse_markdown_for_synth(
        md,
        read_headers="none",
        url_policy="drop",
        url_placeholder="URL provided",
        img_policy="none",
        img_placeholder="Image provided",
    )
    assert len(paras) == 1
    p = paras[0]
    assert p["h1"] == "H1"
    assert p["h2"] == "H2"
    assert p["urls"] == ["https://example.com"]
    assert p["images"] == ["img.png"]


def test_voice_alias_orca_maps_to_onyx():
    from tts3x.pipeline import _normalize_voice_name

    assert _normalize_voice_name("orca") == "onyx"
