AI Personality & Vibe Analyzer — One-Page Proposal
=================================================

Abstract
--------
This project builds an AI web app that analyzes a user's text (tweets, bios, journal entries) and returns a personality estimate (MBTI / Big Five), emotional tone (joy, sadness, anxiety, calm), and a visual “aesthetic vibe” (e.g., dark academia, cottagecore, minimalism). The product blends NLP modeling, psycholinguistic mapping, and a lightweight UI so users can quickly see "what their text feels like" and explore personality + visual style matches.

Goals
-----
- Produce interpretable personality predictions from short-form text (MBTI + Big Five score proxies).
- Detect emotional tone and top emotions (using GoEmotions or a finetuned emotion classifier).
- Map text embeddings to predefined aesthetic/vibe anchors (CLIP or text-embedding matching).
- Provide an attractive demo UI (Gradio or Streamlit) with visualizations: personality radar, mood cloud, and vibe badge.
- Ensure privacy-first design (option to not store inputs, local model option).

Datasets & Data Sources
-----------------------
- MBTI 5000 (Kaggle): text data mapped to MBTI labels — used for initial model training/prototyping.
- GoEmotions (Google): multi-label emotion annotations for emotion classifier training.
- Optional: Reddit/Twitter scraped corpora (for vocabulary coverage) — obey TOS; prefer public datasets.
- For vibe anchors: curated label list (e.g., cottagecore, dark academia, techcore) and small curated text/sentences describing each vibe for anchor embeddings.

Modeling Approach
-----------------
- Text encoding: SentenceTransformers (all-MiniLM or DistilBERT) or OpenAI embeddings.
- Personality classifier: logistic regression or light fine-tuned transformer (DistilBERT/RoBERTa) with cross-validation; return MBTI label + trait scores (E/I, S/N, T/F, J/P) or Big Five continuous scores.
- Emotion classifier: multi-label classifier trained on GoEmotions; return probabilities and top-k emotions.
- Vibe detection: compare text embedding to vibe-anchor embeddings (cosine similarity) to produce top-1/top-3 vibes and a “vibe score.”
- Ensemble & calibration: combine model confidences with heuristic mapping rules (e.g., explicit “planning” words boost J-score) for interpretability.

Deliverables
------------
- `PROPOSAL.md` (this document)
- `README.md` with quickstart and deployment notes
- Minimal demo app (`app/main.py`) using Gradio/Streamlit
- Training notebooks (`notebooks/`) for MBTI & emotion models
- Preprocessing + predictor module (`app/predictor.py`)
- Visuals (`app/visuals.py`) producing radar charts and mood clouds
- Requirements file and instructions for local run

Evaluation & Metrics
--------------------
- Personality: accuracy / macro-F1 (MBTI labels) + per-dimension ROC-AUC or mean absolute error for trait scores.
- Emotions: micro/macro-F1, precision@k for top emotions.
- Vibe mapping: human-labeled small test set for precision@1/3; qualitative inspection.
- Calibration: reliability diagrams for confidence estimates.

Project Timeline (suggested)
----------------------------
- Week 1 — Data prep, baseline models, simple Gradio demo with random/synthetic outputs
- Week 2 — Train MBTI/emotion models, add vibe anchors, visuals
- Week 3 — Polishing UI, demo script, README + export models
- Optional Week 4 — Extra features: playlist/moodboard, journaling tracker, chat mode

Ethics & Privacy
----------------
- Clearly state limitations: MBTI is a heuristic and not a clinical diagnosis.
- Add opt-out and ephemeral input handling: by default, do NOT store user text.
- Transparency: show confidence and top contributing tokens/phrases.
- Content moderation: detect and block toxic inputs; avoid amplifying harmful language.
- User consent: on first-run show a short notice about how data is used.

Resources & Tech Stack
----------------------
- Python, Hugging Face Transformers / SentenceTransformers
- scikit-learn for quick baselines and calibration
- Streamlit or Gradio for demo UI
- Plotly / Matplotlib / WordCloud / seaborn for visuals
- Optional OpenAI embeddings if available (API key via `.env`)

Success Criteria
----------------
- A working web demo that can analyze arbitrary short text, display personality + emotion + vibe, and produce attractive visuals.
- Clear README so a reviewer can run demo locally within 10 minutes.
- Code that is modular and easy to extend (e.g., swap from DistilBERT to a finetuned model).
