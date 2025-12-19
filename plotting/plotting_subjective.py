import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(BASE_DIR, "results", "subjective_metrics.json")
OUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


SCORE_FIELDS = [
    "task_fulfillment",
    "groundedness",
    "clarity",
    "pleasantness",
    "overall_quality",
]


# -------------------------
# Loading
# -------------------------

def load_subjective_results(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def is_valid_score_block(scores: Dict) -> bool:
    return all(
        k in scores and isinstance(scores[k], (int, float))
        for k in SCORE_FIELDS
    )


# -------------------------
# Aggregation
# -------------------------

def aggregate_scores(conversations: List[Dict]):
    """
    Aggregate mean subjective scores per:
    assistant_variant × persona × memory
    """
    buckets = defaultdict(lambda: defaultdict(list))

    for conv in conversations:
        scores = conv.get("scores", {})
        if not is_valid_score_block(scores):
            continue

        key = (
            conv["assistant_variant"],
            conv["persona"],
            conv.get("long_term_memory_profile", True),
        )

        for field in SCORE_FIELDS:
            buckets[key][field].append(scores[field])

    # reduce to means
    summary = {}
    for key, field_dict in buckets.items():
        summary[key] = {
            field: float(np.mean(values))
            for field, values in field_dict.items()
            if values
        }

    return summary


# -------------------------
# Plotting
# -------------------------

def plot_subjective_means(summary: Dict):
    """
    Bar plot: mean subjective scores per condition.
    """
    conditions = list(summary.keys())
    labels = [
        f"{a}\n{p}\nmemory={m}"
        for (a, p, m) in conditions
    ]

    x = np.arange(len(conditions))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, field in enumerate(SCORE_FIELDS):
        values = [
            summary[c].get(field, 0.0)
            for c in conditions
        ]
        ax.bar(
            x + i * width,
            values,
            width,
            label=field.replace("_", " "),
        )

    ax.set_ylabel("Mean score (1–5)")
    ax.set_title("Subjective Evaluation by LLM Judge")
    ax.set_xticks(x + width * (len(SCORE_FIELDS) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(1, 5)
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "subjective_scores_by_condition.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved {out_path}")


def plot_overall_quality(summary: Dict):
    """
    Compact comparison plot for overall quality only.
    """
    conditions = list(summary.keys())
    labels = [
        f"{a} | {p} | mem={m}"
        for (a, p, m) in conditions
    ]

    values = [
        summary[c]["overall_quality"]
        for c in conditions
        if "overall_quality" in summary[c]
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Mean overall quality (1–5)")
    ax.set_title("Overall Quality by Experimental Condition")
    ax.set_ylim(1, 5)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "subjective_overall_quality.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved {out_path}")


# -------------------------
# Main
# -------------------------

def main():
    data = load_subjective_results(IN_PATH)
    summary = aggregate_scores(data)

    if not summary:
        print("[WARN] No valid subjective scores found.")
        return

    plot_subjective_means(summary)
    plot_overall_quality(summary)


if __name__ == "__main__":
    main()
