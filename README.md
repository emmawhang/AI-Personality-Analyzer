# AI Personality & Vibe Analyzer

Short description
-----------------
AI Personality & Vibe Analyzer predicts personality (MBTI + trait scores), emotional tone, and the visual/ aesthetic "vibe" from short text (tweets, bios, journal excerpts). Outputs include: personality label + trait radar, emotion probabilities + mood cloud, and a vibe badge (e.g., "dark academia").

Quick demo 
----------------------------



Table of contents
-----------------
- Features
- Demo
- Tech stack
- Folder structure
- Installation
- Quickstart (run locally)
- Models & datasets
- Design & pipeline
- Evaluation
- Ethics & privacy
- Stretch features
- License

Features
--------
- MBTI inference + per-dimension trait scores (E/I, S/N, T/F, J/P)
- Emotion classification (multi-label; top-k emotions)
- Aesthetic vibe detection from a curated list (CLIP or text embedding matching)
- Visualizations: radar chart, mood cloud, vibe badge
- Gradio/Streamlit demo for interactive use

Tech stack
----------
- Python 3.10+
- Hugging Face Transformers / SentenceTransformers
- scikit-learn
- Gradio or Streamlit
- Plotly / matplotlib / wordcloud

Folder structure
----------------
```
ai-vibe-analyzer/
├── data/                   # raw & processed datasets (not checked-in)
├── models/                 # trained model files (.pt / .pkl)
├── app/
│   ├── main.py             # Gradio/Streamlit demo
│   ├── predictor.py        # load models & inference logic
│   ├── visuals.py          # plotting helpers
├── notebooks/
│   ├── mbti_training.ipynb
│   ├── emotion_training.ipynb
├── requirements.txt
├── PROPOSAL.md
└── README.md
```

Installation
------------
1. Create a virtualenv and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set up environment variables (optional; only needed for OpenAI embeddings or hosted model keys):
```bash
# .env or export

Quickstart (local demo)
----------------------
1. Run the demo (Gradio example):
```bash
python app/main.py
```
2. Open the provided localhost URL in your browser (e.g., http://127.0.0.1:7860).

Models & Data
-------------
- MBTI dataset: Kaggle "MBTI 5000" — use for initial labeling (link in `data/README.md`).
- Emotions: GoEmotions dataset — multi-label emotion taxonomy.
- Vibe anchors: small curated CSV mapping vibe name → short descriptive sentences for embeddings.

Design & pipeline (diagram)
---------------------------
Text input -> preprocessing -> embedding (SentenceTransformers/CLIP/OpenAI) -> 
- branch A: MBTI classifier -> trait scores & label
- branch B: emotion classifier -> probabilities, top emotions
- branch C: vibe matching -> cosine similarity -> top vibes
-> aggregator -> UI visuals (radar + mood cloud + vibe badge)

Evaluation
----------
- MBTI: accuracy, macro-F1, per-dimension MAE for trait scores
- Emotions: micro/macro-F1, precision@k
- Vibes: precision@1/3 on a small human-labeled set
- Logging experimental runs with `wandb` or simple CSV

Ethics & privacy
----------------
- Do NOT store user inputs unless explicitly opted-in.
- Provide clear labeling that MBTI is a heuristic and not clinical advice.
- Add content moderation to filter hate/toxic speech.

Stretch features
---------------
- Generate a Spotify playlist or Pinterest moodboard matching the vibe.
- Journaling tracker to show personality/emotion trends over time.
- Chat mode: a safe "AI psychologist" with guardrails.

Example outputs
---------------
Input:
> "I love slow mornings with a good book and a tidy desk. Planning the week makes me calm and productive."

Output:
- Personality: likely INJ? (e.g., INFJ) — scores: E:10% / I:90%, S:35% / N:65%, T:40% / F:60%, J:75% / P:25%
- Emotions: calm (0.78), contentment (0.55), nostalgia (0.32)
- Vibe: soft minimalism (score 0.82) — keywords: "cozy, tidy, bookish"

How to contribute
-----------------
- Fork, branch, submit PRs.
- Add dataset links or new vibe anchors to `data/vibes.csv`.
- Add tests for `predictor.py` to ensure reproducible outputs.

License
-------
MIT (or choose your preferred license). Include attribution for datasets and model licenses.
