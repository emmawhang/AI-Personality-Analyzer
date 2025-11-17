"""Helpers to generate visuals for the demo.

Functions here are minimal and intended for the demo app. Replace or extend
with Plotly versions or better styling as desired.
"""
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import os


def radar_chart(trait_scores: Dict[str, float], out_path: str):
    # Expect trait_scores keys: E,I,S,N,T,F,J,P (we'll map to four axes)
    labels = ["E/I", "S/N", "T/F", "J/P"]
    values = [trait_scores.get("E", 0.5) - trait_scores.get("I", 0.5) + 0.5,
              trait_scores.get("S", 0.5) - trait_scores.get("N", 0.5) + 0.5,
              trait_scores.get("T", 0.5) - trait_scores.get("F", 0.5) + 0.5,
              trait_scores.get("J", 0.5) - trait_scores.get("P", 0.5) + 0.5]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="tab:blue", linewidth=2)
    ax.fill(angles, values, color="tab:blue", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def mood_wordcloud(text: str, out_path: str):
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def render_all(text: str, trait_scores: Dict[str, float], out_dir: str) -> Dict[str, str]:
    """Generate radar chart and mood cloud images. Returns paths."""
    ensure_dir(out_dir + "/placeholder")
    radar_path = os.path.join(out_dir, "radar.png")
    cloud_path = os.path.join(out_dir, "mood_cloud.png")
    radar_chart(trait_scores, radar_path)
    mood_wordcloud(text, cloud_path)
    return {"radar": radar_path, "mood_cloud": cloud_path}
