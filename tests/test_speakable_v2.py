from tts3x.pipeline import _build_speakable_markdown_v2, parse_markdown_for_synth


def test_bullets_narrated_with_ordinals():
    md = "# H1\n\n- one item\n- two item\n- three item"
    out = _build_speakable_markdown_v2(md, url_mode="mention", url_mention="URL provided", img_mode="mention", img_mention="Image provided")
    assert "First, one item." in out
    assert "Second, two item." in out
    assert "Finally, three item." in out


def test_quote_block_tagged_for_quote_voice():
    md = "> This is a quoted line."
    out = _build_speakable_markdown_v2(md, url_mode="mention", url_mention="URL provided", img_mode="none", img_mention="Image provided")
    assert "[Quote]: This is a quoted line." in out


def test_url_and_image_metadata_tags_survive_for_synth_parse():
    md = "Paragraph with [ref](https://example.com) and ![graph](img.png)."
    out = _build_speakable_markdown_v2(md, url_mode="none", url_mention="URL provided", img_mode="mention", img_mention="Image provided")
    paras = parse_markdown_for_synth(
        out,
        read_headers="none",
        url_policy="drop",
        url_placeholder="URL provided",
        img_policy="mention",
        img_placeholder="Image provided",
    )
    assert paras
    p = paras[0]
    assert "https://example.com" in p.get("urls", [])
    assert "img.png" in p.get("images", [])
