import json
import os
from collections import defaultdict
from typing import Dict, List

from transformers import AutoTokenizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, "data", "merged_conversations.json")
OUT_PATH = os.path.join(BASE_DIR, "results", "objective_metrics.json")

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# -------------------------
# Loading
# -------------------------

def load_conversations(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


# -------------------------
# Token metrics
# -------------------------

def count_tokens(history: List[Dict[str, str]], tokenizer):
    user_tokens = 0
    assistant_tokens = 0

    for msg in history:
        n = len(tokenizer.encode(msg["content"]))
        if msg["role"] == "user":
            user_tokens += n
        elif msg["role"] == "assistant":
            assistant_tokens += n

    return user_tokens, assistant_tokens


# -------------------------
# Structural metrics
# -------------------------

def count_roles(history: List[Dict[str, str]]):
    user_msgs = sum(1 for m in history if m["role"] == "user")
    assistant_msgs = sum(1 for m in history if m["role"] == "assistant")
    return user_msgs, assistant_msgs


def assistant_lexical_diversity(history: List[Dict[str, str]], tokenizer) -> float:
    tokens = []
    for msg in history:
        if msg["role"] == "assistant":
            tokens.extend(tokenizer.encode(msg["content"]))

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


# -------------------------
# Main evaluation
# -------------------------

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    conversations = load_conversations(LOGS_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    # Aggregation structure
    stats = defaultdict(lambda: {
        "count": 0,
        "success": 0,

        # core
        "turns": [],
        "user_turns": [],
        "assistant_turns": [],

        # tokens
        "tokens_user": [],
        "tokens_assistant": [],
        "tokens_total": [],
        "assistant_tokens_per_turn": [],
        "assistant_user_token_ratio": [],

        # language quality proxy
        "lexical_diversity": [],
    })

    # -------------------------
    # Collect stats
    # -------------------------
    for conv in conversations:
        variant = conv["assistant_variant"]
        persona = conv["persona"]
        memory = conv.get("long_term_memory_profile", True)

        key = f"{variant}__{persona}__memory_{memory}"

        history = conv["history"]
        num_turns = conv["num_turns"]
        success = conv["stop_reason"] == "user_satisfied"

        user_toks, assistant_toks = count_tokens(history, tokenizer)
        total_toks = user_toks + assistant_toks

        user_turns, assistant_turns = count_roles(history)

        assistant_tpt = (
            assistant_toks / assistant_turns if assistant_turns > 0 else 0.0
        )
        token_ratio = (
            assistant_toks / user_toks if user_toks > 0 else 0.0
        )

        lex_div = assistant_lexical_diversity(history, tokenizer)

        entry = stats[key]
        entry["count"] += 1
        entry["success"] += int(success)

        entry["turns"].append(num_turns)
        entry["user_turns"].append(user_turns)
        entry["assistant_turns"].append(assistant_turns)

        entry["tokens_user"].append(user_toks)
        entry["tokens_assistant"].append(assistant_toks)
        entry["tokens_total"].append(total_toks)
        entry["assistant_tokens_per_turn"].append(assistant_tpt)
        entry["assistant_user_token_ratio"].append(token_ratio)

        entry["lexical_diversity"].append(lex_div)

    # -------------------------
    # Reduce to summary
    # -------------------------
    summary = {}

    for key, v in stats.items():
        n = v["count"]

        summary[key] = {
            "num_conversations": n,

            # success (kept for completeness, not interpretation)
            "success_rate": v["success"] / n if n > 0 else 0.0,

            # efficiency
            "avg_turns": sum(v["turns"]) / n if n > 0 else 0.0,
            "avg_user_turns": sum(v["user_turns"]) / n if n > 0 else 0.0,
            "avg_assistant_turns": sum(v["assistant_turns"]) / n if n > 0 else 0.0,

            # token efficiency
            "avg_tokens_user": sum(v["tokens_user"]) / n if n > 0 else 0.0,
            "avg_tokens_assistant": sum(v["tokens_assistant"]) / n if n > 0 else 0.0,
            "avg_tokens_total": sum(v["tokens_total"]) / n if n > 0 else 0.0,
            "assistant_tokens_per_turn": sum(v["assistant_tokens_per_turn"]) / n if n > 0 else 0.0,
            "assistant_user_token_ratio": sum(v["assistant_user_token_ratio"]) / n if n > 0 else 0.0,

            # language signal
            "assistant_lexical_diversity": sum(v["lexical_diversity"]) / n if n > 0 else 0.0,
        }

    # -------------------------
    # Write output
    # -------------------------
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Objective evaluation written to {OUT_PATH}")

    # Pretty print
    for k, v in summary.items():
        print("\n", k)
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f"  {kk}: {vv:.3f}")
            else:
                print(f"  {kk}: {vv}")


if __name__ == "__main__":
    main()
