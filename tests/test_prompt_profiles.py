import json

from tts3x.pipeline import list_explained_prompt_profiles


def test_default_profiles_present():
    profiles = list_explained_prompt_profiles()
    assert "academic_default" in profiles
    assert "whitepaper_education" in profiles
    assert "novel_literary" in profiles


def test_custom_profile_file_overrides(tmp_path):
    p = tmp_path / "prompts.json"
    p.write_text(
        json.dumps(
            {
                "academic_default": {"description": "custom academic"},
                "policy_brief": {
                    "description": "policy mode",
                    "title": "Policy Brief Walkthrough",
                    "system_prompt": "Explain policy text.",
                    "task": "Explain paragraph.",
                    "constraints": ["One paragraph."],
                },
            }
        ),
        encoding="utf-8",
    )

    profiles = list_explained_prompt_profiles(p)
    assert profiles["academic_default"] == "custom academic"
    assert "policy_brief" in profiles
