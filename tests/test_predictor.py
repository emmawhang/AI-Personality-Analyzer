import sys
import os
import pytest
# ensure repo root is on sys.path so `app` package can be imported during tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import predictor


def test_analyze_returns_expected_keys():
    sample = "I enjoy quiet mornings reading and planning my week. I prefer small gatherings over big parties."
    out = predictor.analyze(sample)
    assert isinstance(out, dict)
    for k in ["mbti", "axis_probs", "emotion_probs", "vibe", "vibe_ranked"]:
        assert k in out


def test_axis_probs_sane():
    sample = "I like parties and meeting friends at cafes."
    out = predictor.analyze(sample)
    # axis_probs values should be between 0 and 1
    for v in out["axis_probs"].values():
        assert 0.0 <= v <= 1.0
