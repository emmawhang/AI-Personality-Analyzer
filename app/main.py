"""Gradio UI for the AI Personality & Vibe Analyzer.

This UI uses `analyze(text)` from `app.predictor` which returns the
structure described in the README and predictor module.
"""
import gradio as gr
import pandas as pd
from datetime import datetime
from app.predictor import analyze
from pathlib import Path
import sqlite3

DB_PATH = Path("data") / "analysis.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# initialize DB
conn = sqlite3.connect(DB_PATH)
conn.execute(
    "CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, text_len INTEGER, mbti TEXT, vibe TEXT, snippet TEXT)"
)
conn.commit()
conn.close()


def run_analysis(user_text):
    if not user_text or len(user_text.strip()) < 10:
        return "Please paste at least a few sentences.", None, None, None

    res = analyze(user_text)

    # Pretty strings / tables
    mbti_str = f"Predicted MBTI: **{res['mbti']}**"
    vibe_str = f"Top Aesthetic Vibe: **{res['vibe']}**"

    # Emotions table
    emo_items = sorted(res["emotion_probs"].items(), key=lambda x: x[1], reverse=True)
    emo_df = pd.DataFrame(emo_items, columns=["emotion", "probability"]) if emo_items else None

    # Vibes ranked
    vibe_df = pd.DataFrame(res["vibe_ranked"], columns=["vibe", "similarity"]) if res.get("vibe_ranked") else None

    # Axis (E/I etc.)
    axis_pairs = [
        ("E/I", res["axis_probs"].get("E", 0.5), res["axis_probs"].get("I", 0.5)),
        ("S/N", res["axis_probs"].get("S", 0.5), res["axis_probs"].get("N", 0.5)),
        ("T/F", res["axis_probs"].get("T", 0.5), res["axis_probs"].get("F", 0.5)),
        ("J/P", res["axis_probs"].get("J", 0.5), res["axis_probs"].get("P", 0.5)),
    ]
    axis_df = pd.DataFrame(axis_pairs, columns=["axis", "first", "second"] )

    # Append to SQLite log
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO analyses (ts, text_len, mbti, vibe, snippet) VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), len(user_text), res['mbti'], res['vibe'], user_text[:200]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    summary = f"{mbti_str}\n\n{vibe_str}"
    return summary, emo_df, vibe_df, axis_df


with gr.Blocks(title="AI Personality & Vibe Analyzer") as demo:
    gr.Markdown("# 🪞 AI Personality & Vibe Analyzer\nPaste text to see MBTI, emotions, and vibe.")
    txt = gr.Textbox(lines=7, label="Your text", value="I love slow mornings with a good book and a tidy desk. Planning the week makes me calm and productive.")
    btn = gr.Button("Analyze")
    out_summary = gr.Markdown()
    out_emotions = gr.Dataframe(label="Emotion probabilities")
    out_vibes = gr.Dataframe(label="Vibe ranking (cosine similarity)")
    out_axes = gr.Dataframe(label="MBTI axis probabilities (heuristic)")

    btn.click(run_analysis, inputs=txt, outputs=[out_summary, out_emotions, out_vibes, out_axes])
    with gr.Tab("Logs"):
        gr.Markdown("### Recent analyses")
        logs_df = gr.Dataframe(label="Recent logs")
        refresh = gr.Button("Refresh logs")
        clear = gr.Button("Clear logs")

        def _get_logs():
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.execute("SELECT ts, text_len, mbti, vibe, snippet FROM analyses ORDER BY id DESC LIMIT 200")
                rows = cur.fetchall()
                conn.close()
                return rows
            except Exception:
                return []

        def _clear_logs():
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM analyses")
                conn.commit()
                conn.close()
                return []
            except Exception:
                return []

        refresh.click(lambda: _get_logs(), inputs=None, outputs=[logs_df])
        clear.click(lambda: _clear_logs(), inputs=None, outputs=[logs_df])

if __name__ == "__main__":
    demo.launch()
