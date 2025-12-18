import json
import os
from collections import defaultdict
from typing import Dict, List

from transformers import AutoTokenizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, "logs", "conversations.json")
OUT_PATH = os.path.join(BASE_DIR, "results", "objective_metrics.json")

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_conversations(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


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


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    conversations = load_conversations(LOGS_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    # aggregation structure
    stats = defaultdict(lambda: {
        "count": 0,
        "success": 0,
        "turns": [],
        "tokens_user": [],
        "tokens_assistant": [],
        "tokens_total": [],
    })

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

        entry = stats[key]
        entry["count"] += 1
        entry["success"] += int(success)
        entry["turns"].append(num_turns)
        entry["tokens_user"].append(user_toks)
        entry["tokens_assistant"].append(assistant_toks)
        entry["tokens_total"].append(total_toks)

    # reduce to summary statistics
    summary = {}
    for key, v in stats.items():
        n = v["count"]
        summary[key] = {
            "num_conversations": n,
            "success_rate": v["success"] / n if n > 0 else 0.0,
            "avg_turns": sum(v["turns"]) / n if n > 0 else 0.0,
            "avg_tokens_user": sum(v["tokens_user"]) / n if n > 0 else 0.0,
            "avg_tokens_assistant": sum(v["tokens_assistant"]) / n if n > 0 else 0.0,
            "avg_tokens_total": sum(v["tokens_total"]) / n if n > 0 else 0.0,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Objective evaluation written to {OUT_PATH}")

    # print
    for k, v in summary.items():
        print("\n", k)
        for kk, vv in v.items():
            print(f"  {kk}: {vv:.3f}" if isinstance(vv, float) else f"  {kk}: {vv}")


if __name__ == "__main__":
    main()
