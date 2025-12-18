import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECTIVE_PATH = os.path.join(BASE_DIR, "results", "subjective_metrics.json")
OUT_DIR = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid")


def load_subjective():
    with open(SUBJECTIVE_PATH, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        scores = entry.get("scores", {})
        binary = scores.get("binary_success", "").lower()

        rows.append({
            "variant": entry["assistant_variant"],
            "persona": entry["persona"],
            "memory": entry.get("long_term_memory_profile", True),
            "success": 1 if binary == "yes" else 0,
        })

    return pd.DataFrame(rows)


def plot_satisfaction_by_variant(df):
    agg = (
        df.groupby(["variant", "memory"])["success"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=agg,
        x="variant",
        y="success",
        hue="memory",
        ax=ax,
    )

    ax.set_title("Satisfaction Judge – Task Completion Rate")
    ax.set_ylabel("Fraction successful")
    ax.set_xlabel("Assistant variant")
    ax.set_ylim(0, 1)
    ax.legend(title="Memory")

    out_path = os.path.join(OUT_DIR, "satisfaction_by_variant.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {out_path}")


def plot_satisfaction_by_persona(df):
    agg = (
        df.groupby(["persona", "memory"])["success"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=agg,
        x="persona",
        y="success",
        hue="memory",
        ax=ax,
    )

    ax.set_title("Satisfaction Judge – Persona Comparison")
    ax.set_ylabel("Fraction successful")
    ax.set_xlabel("Persona")
    ax.set_ylim(0, 1)
    ax.legend(title="Memory")

    out_path = os.path.join(OUT_DIR, "satisfaction_by_persona.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {out_path}")


def plot_overall_satisfaction(df):
    rate = df["success"].mean()

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(
        x=["All conversations"],
        y=[rate],
        ax=ax,
    )

    ax.set_title("Overall Satisfaction Judge Success Rate")
    ax.set_ylabel("Fraction successful")
    ax.set_ylim(0, 1)

    out_path = os.path.join(OUT_DIR, "satisfaction_overall.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {out_path}")


def main():
    df = load_subjective()
    print(df.head())

    plot_satisfaction_by_variant(df)
    plot_satisfaction_by_persona(df)
    plot_overall_satisfaction(df)


if __name__ == "__main__":
    main()
