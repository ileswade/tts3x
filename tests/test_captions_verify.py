from tts3x.pipeline import _timing_coverage


def test_timing_coverage_basic():
    payload = {
        "words": [
            {"word": "a", "start": 10.0, "end": 10.2},
            {"word": "b", "start": 89.8, "end": 90.0},
        ]
    }
    cov = _timing_coverage(payload, audio_duration=100.0)
    assert round(cov["coverage"], 2) == 0.8
    assert cov["first_start"] == 10.0
    assert cov["last_end"] == 90.0
