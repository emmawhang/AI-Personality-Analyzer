"""Predictor and analyzer for the AI Personality & Vibe Analyzer.

This module tries to load optional trained MBTI per-axis models from
`models/mbti_EI.pkl`, `models/mbti_SN.pkl`, `models/mbti_TF.pkl`, and
`models/mbti_JP.pkl`. If they aren't present, it falls back to a simple
heuristic. It uses a Hugging Face emotion model (if available) and
SentenceTransformers for vibe matching.
"""
from typing import Dict, List, Tuple
import re
import os
import csv
import math
from pathlib import Path

try:
    # joblib is typically used to save sklearn pipelines
    import joblib
except Exception:
    joblib = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# -----------------------------
# Emotion pipeline
# -----------------------------
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
emotion_pipe = None
if pipeline is not None:
    try:
        emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )
    except Exception:
        try:
            emotion_pipe = pipeline("sentiment-analysis", return_all_scores=True)
            EMOTION_LABELS = ["NEGATIVE", "POSITIVE"]
        except Exception:
            emotion_pipe = None
else:
    emotion_pipe = None


def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    return t.strip()


def predict_emotions(text: str) -> Dict[str, float]:
    text = (text or "").strip()
    if not text:
        return {lbl: 0.0 for lbl in EMOTION_LABELS}
    if emotion_pipe is None:
        # Fallback: return uniform distribution so downstream logic has values
        n = len(EMOTION_LABELS) or 1
        return {lbl: 1.0 / n for lbl in EMOTION_LABELS}

    scores = emotion_pipe(text)[0]
    out: Dict[str, float] = {}
    for item in scores:
        label = item.get("label", "")
        score = float(item.get("score", 0.0))
        out[label.lower()] = score
    # Ensure consistent labels
    for lbl in EMOTION_LABELS:
        if lbl not in out:
            out[lbl] = 0.0
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


# -----------------------------
# Vibe matching (SentenceTransformers)
# -----------------------------
DEFAULT_VIBES = {
    "dark academia": "scholarly, moody, introspective, classical literature, libraries",
    "cottagecore": "pastoral, nature, simple living, gentle, floral, countryside",
    "city cool": "urban, chic, confident, fast-paced, modern street style",
    "techcore": "futuristic, efficient, sleek, innovation, coding, productivity",
    "soft minimalism": "calm, neutral, tidy, intentional, subtle, airy",
    "y2k playful": "nostalgic, bubbly, colorful, fun, early 2000s internet",
    "cozy cafe": "warm, inviting, coffee, soft music, journaling, rainy day",
    "bold maximalism": "vibrant, expressive, layered, eclectic, loud, energetic",
}

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embed_model = None


def _load_vibe_anchors() -> Tuple[List[str], List[str]]:
    labels = []
    anchors = []
    csv_path = _DATA_DIR / "vibes.csv"
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    name = r.get("vibe_name") or r.get("vibe")
                    txt = r.get("example_text") or r.get("description")
                    if name and txt:
                        labels.append(name)
                        anchors.append(txt)
        except Exception:
            pass
    if not labels:
        labels = list(DEFAULT_VIBES.keys())
        anchors = list(DEFAULT_VIBES.values())
    return labels, anchors


VIBE_LABELS, VIBE_ANCHORS = _load_vibe_anchors()
if embed_model is not None:
    try:
        VIBE_ANCHOR_EMB = embed_model.encode(VIBE_ANCHORS, normalize_embeddings=True)
    except Exception:
        VIBE_ANCHOR_EMB = None
else:
    VIBE_ANCHOR_EMB = None


def build_and_save_vibe_index(save_dir: str = None):
    """Compute vibe anchor embeddings and save them to `models/` so the app can
    load them offline. This writes two files: `vibe_anchor_emb.npy` and
    `vibe_labels.txt` under `models/` by default.
    """
    if embed_model is None:
        raise RuntimeError("SentenceTransformer embedding model not available")
    save_dir = Path(save_dir) if save_dir else _MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    import numpy as _np

    emb = embed_model.encode(VIBE_ANCHORS, normalize_embeddings=True)
    emb_path = save_dir / "vibe_anchor_emb.npy"
    _np.save(emb_path, emb)
    labels_path = save_dir / "vibe_labels.txt"
    with open(labels_path, "w", encoding="utf8") as fh:
        for lbl in VIBE_LABELS:
            fh.write(lbl + "\n")
    return str(emb_path), str(labels_path)


def load_vibe_index_from_models(save_dir: str = None):
    """Attempt to load precomputed vibe anchor embeddings from `models/`.
    If found, set VIBE_ANCHOR_EMB and VIBE_LABELS accordingly.
    """
    global VIBE_LABELS, VIBE_ANCHOR_EMB
    save_dir = Path(save_dir) if save_dir else _MODEL_DIR
    emb_path = save_dir / "vibe_anchor_emb.npy"
    labels_path = save_dir / "vibe_labels.txt"
    if emb_path.exists() and labels_path.exists():
        import numpy as _np

        emb = _np.load(emb_path)
        with open(labels_path, "r", encoding="utf8") as fh:
            labels = [l.strip() for l in fh if l.strip()]
        VIBE_LABELS = labels
        VIBE_ANCHOR_EMB = emb
        return True
    return False


def predict_vibe(text: str) -> Tuple[str, List[Tuple[str, float]]]:
    txt = clean_text(text)
    if embed_model is None or VIBE_ANCHOR_EMB is None:
        # fallback: keyword-based simple mapping
        txt_low = txt.lower()
        for lbl in VIBE_LABELS:
            if lbl in txt_low:
                return lbl, [(lbl, 1.0)]
        return VIBE_LABELS[0], [(VIBE_LABELS[0], 0.5)]

    txt_emb = embed_model.encode([txt], normalize_embeddings=True)
    sims = util.cos_sim(txt_emb, VIBE_ANCHOR_EMB).cpu().numpy()[0]
    pairs = list(zip(VIBE_LABELS, sims.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    # convert to floats
    return pairs[0][0], [(lbl, float(s)) for lbl, s in pairs]


# -----------------------------
# MBTI models: optional trained per-axis loaders
# -----------------------------
MBTI_MODEL_NAMES = {
    "EI": _MODEL_DIR / "mbti_EI.pkl",
    "SN": _MODEL_DIR / "mbti_SN.pkl",
    "TF": _MODEL_DIR / "mbti_TF.pkl",
    "JP": _MODEL_DIR / "mbti_JP.pkl",
}

_mbti_models = {}


def _load_mbti_models():
    global _mbti_models
    if joblib is None:
        return
    for axis, path in MBTI_MODEL_NAMES.items():
        try:
            if path.exists():
                _mbti_models[axis] = joblib.load(path)
        except Exception:
            # ignore load errors and leave heuristic fallback
            _mbti_models.pop(axis, None)


_load_mbti_models()


# Simple lexical heuristic as fallback (same hints as earlier plan)
HINTS = {
    "E": ["party", "friends", "outgoing", "network", "energized by people", "talk", "meetup"],
    "I": ["introspective", "reflect", "alone time", "quiet", "journaling", "read", "contemplate"],
    "S": ["practical", "details", "present", "facts", "step-by-step", "concrete"],
    "N": ["intuitive", "abstract", "imagine", "possibility", "pattern", "vision", "concept"],
    "T": ["logic", "analyze", "rational", "objective", "systems", "efficient"],
    "F": ["values", "empathy", "care", "harmony", "feelings", "kindness", "support"],
    "J": ["plan", "schedule", "organized", "decide", "structure", "deadline"],
    "P": ["spontaneous", "explore", "adapt", "flexible", "open-ended", "improvise"],
}


def mbti_heuristic(text: str) -> Tuple[str, Dict[str, float]]:
    t = (text or "").lower()
    axis_scores = {k: 0 for k in HINTS.keys()}
    for k, words in HINTS.items():
        for w in words:
            if w in t:
                axis_scores[k] += 1

    def axis_prob(pos, neg):
        s_pos = axis_scores[pos]
        s_neg = axis_scores[neg]
        total = s_pos + s_neg
        if total == 0:
            return 0.5, 0.5
        return s_pos / total, s_neg / total

    e, i = axis_prob("E", "I")
    s, n = axis_prob("S", "N")
    t_, f = axis_prob("T", "F")
    j, p = axis_prob("J", "P")

    letters = "".join([
        "E" if e >= i else "I",
        "S" if s >= n else "N",
        "T" if t_ >= f else "F",
        "J" if j >= p else "P",
    ])
    probs = {"E": e, "I": i, "S": s, "N": n, "T": t_, "F": f, "J": j, "P": p}
    return letters, probs


def _predict_mbti_with_models(text: str) -> Tuple[str, Dict[str, float]]:
    """If trained models are available, use them to compute axis probs.

    Each model is expected to be a scikit-learn Pipeline that supports
    `predict_proba(X)` returning [[prob_neg, prob_pos], ...]
    where 'pos' corresponds to the first letter of the axis (E,S,T,J).
    """
    if not _mbti_models:
        return None
    probs = {}
    letters = []
    for axis, model in _mbti_models.items():
        try:
            proba = model.predict_proba([text])[0]
            # scikit's predict_proba convention: classes order in model.classes_
            # try to find index of positive class (1) or the axis letter
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    idx = classes.index(1)
                    pos_prob = float(proba[idx])
                else:
                    # fallback: take the max
                    pos_prob = float(max(proba))
            else:
                pos_prob = float(max(proba))
        except Exception:
            # if anything fails, fall back to heuristic entirely
            return None
        # map axis name to letters
        if axis == "EI":
            probs["E"] = pos_prob
            probs["I"] = 1.0 - pos_prob
            letters.append("E" if pos_prob >= 0.5 else "I")
        elif axis == "SN":
            probs["S"] = pos_prob
            probs["N"] = 1.0 - pos_prob
            letters.append("S" if pos_prob >= 0.5 else "N")
        elif axis == "TF":
            probs["T"] = pos_prob
            probs["F"] = 1.0 - pos_prob
            letters.append("T" if pos_prob >= 0.5 else "F")
        elif axis == "JP":
            probs["J"] = pos_prob
            probs["P"] = 1.0 - pos_prob
            letters.append("J" if pos_prob >= 0.5 else "P")

    mbti_label = "".join(letters)
    return mbti_label, probs


def analyze(text: str) -> Dict:
    """Top-level analyzer used by the Gradio app.

    Returns a dict with keys: mbti, axis_probs, emotion_probs, vibe, vibe_ranked
    """
    text = (text or "").strip()
    if len(text) < 1:
        raise ValueError("Please provide text input")

    emotions = predict_emotions(text)
    vibe_label, vibe_ranked = predict_vibe(text)

    # MBTI: prefer trained models if available
    mbti_res = None
    try:
        mbti_res = _predict_mbti_with_models(text)
    except Exception:
        mbti_res = None

    if mbti_res is None:
        mbti_label, axis_probs = mbti_heuristic(text)
    else:
        mbti_label, axis_probs = mbti_res

    return {
        "mbti": mbti_label,
        "axis_probs": axis_probs,
        "emotion_probs": emotions,
        "vibe": vibe_label,
        "vibe_ranked": vibe_ranked,
    }


if __name__ == "__main__":
    # quick smoke test
    s = "I love slow mornings with a good book and a tidy desk. Planning the week makes me calm and productive."
    print(analyze(s))
