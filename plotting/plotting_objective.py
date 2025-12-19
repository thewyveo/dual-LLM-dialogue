import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "results", "objective_metrics.json")
OUT_DIR = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style="whitegrid")


# -------------------------
# Load + normalize
# -------------------------
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        raw = json.load(f)

    rows = []
    for key, vals in raw.items():
        variant, persona, memory = key.split("__")
        memory = memory.replace("memory_", "") == "True"

        rows.append({
            "variant": variant,
            "persona": persona,
            "memory": memory,
            **vals,
        })

    return pd.DataFrame(rows)


def save_plot(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {path}")


# -------------------------
# 1. PERSONA COMPARISON
# -------------------------
def plot_persona_comparison(df):
    metrics = [
        "avg_turns",
        "assistant_tokens_per_turn",
        "assistant_user_token_ratio",
        "assistant_lexical_diversity",
    ]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=df,
            x="persona",
            y=metric,
            hue="variant",
            ax=ax,
        )
        ax.set_title(f"Persona comparison – {metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Persona")
        ax.legend(title="Assistant variant")

        save_plot(fig, f"persona_comparison_{metric}.png")


# -------------------------
# 2. MODEL COMPARISON
# -------------------------
def plot_model_comparison(df):
    metrics = [
        "avg_turns",
        "avg_tokens_total",
        "assistant_tokens_per_turn",
        "assistant_user_token_ratio",
        "assistant_lexical_diversity",
    ]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=df,
            x="variant",
            y=metric,
            hue="memory",
            ax=ax,
        )
        ax.set_title(f"Model comparison – {metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Assistant variant")
        ax.legend(title="Long-term memory")

        save_plot(fig, f"model_comparison_{metric}.png")


# -------------------------
# 3. MEMORY EFFECT (AGGREGATED)
# -------------------------
def plot_memory_comparison(df):
    metrics = [
        "avg_turns",
        "avg_tokens_total",
        "assistant_tokens_per_turn",
        "assistant_user_token_ratio",
        "assistant_lexical_diversity",
    ]

    agg = (
        df.groupby("memory")[metrics]
        .mean()
        .reset_index()
    )

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(
            data=agg,
            x="memory",
            y=metric,
            ax=ax,
        )
        ax.set_title(f"Memory impact – {metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Long-term memory enabled")

        save_plot(fig, f"memory_comparison_{metric}.png")


# -------------------------
# 4. EFFICIENCY VS QUALITY TRADE-OFF
# -------------------------
def plot_tradeoff(df):
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(
        data=df,
        x="avg_turns",
        y="assistant_lexical_diversity",
        hue="variant",          # prompt vs peft
        style="persona",        # minimalist vs explorer
        markers={"minimalist": "o", "explorer": "s"},
        size="memory",          # False / True
        sizes={False: 80, True: 250},
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_title("Efficiency vs Language Richness")
    ax.set_xlabel("Average number of turns")
    ax.set_ylabel("Assistant lexical diversity")

    ax.legend(
        title="Variant / Persona / Memory",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )

    save_plot(fig, "tradeoff_efficiency_vs_diversity.png")



# -------------------------
# MAIN
# -------------------------
def main():
    df = load_metrics()
    print(df.head())

    plot_persona_comparison(df)
    plot_model_comparison(df)
    plot_memory_comparison(df)
    plot_tradeoff(df)


if __name__ == "__main__":
    main()
