import json
import os
from typing import List, Dict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, "logs", "conversations.json")
OUT_PATH = os.path.join(BASE_DIR, "data", "assistant_ft_train.jsonl")


def load_conversations(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def history_to_examples(
    history: List[Dict[str, str]],
    persona: str,
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    for i, msg in enumerate(history):
        if msg["role"] != "assistant":
            continue

        context_msgs = history[:i]
        assistant_reply = msg["content"]

        transcript_lines = []
        transcript_lines.append(
            f"SYSTEM: You are a hotel recommendation assistant talking "
            f"to a traveler with persona '{persona}'. "
            f"Recommend and compare hotels based on constraints."
        )

        for m in context_msgs:
            if m["role"] == "user":
                transcript_lines.append(f"USER: {m['content']}")
            else:
                transcript_lines.append(f"ASSISTANT: {m['content']}")

        input_text = "\n".join(transcript_lines) + "\nASSISTANT:"

        examples.append(
            {
                "input": input_text,
                "output": assistant_reply.strip(),
                "persona": persona,         # <<< add this
            }
        )

    return examples


def main():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    if not os.path.exists(LOGS_PATH):
        print(f"Logs not found at {LOGS_PATH}. Run main.py first to generate conversations.")
        return

    conversations = load_conversations(LOGS_PATH)
    all_examples: List[Dict[str, str]] = []

    for conv in conversations:
        history = conv.get("history", [])
        persona = conv.get("persona", "unknown")
        variant = conv.get("assistant_variant", "")

        # Only use prompt-based assistant as "teacher"
        if variant != "prompt":
            continue

        exs = history_to_examples(history, persona=persona)
        all_examples.extend(exs)

    if not all_examples:
        print("No examples extracted. Check that logs/conversations.json contains prompt-based runs.")
        return

    # ---- BALANCE BY PERSONA ----
    by_persona: Dict[str, List[Dict[str, str]]] = {}
    for ex in all_examples:
        p = ex.get("persona", "unknown")
        by_persona.setdefault(p, []).append(ex)

    # Only balance the personas you care about
    target_personas = ["minimalist", "explorer"]
    counts = [
        len(by_persona[p]) for p in target_personas if p in by_persona
    ]
    if not counts:
        print("No known personas found in examples.")
        return

    max_per_persona = min(counts)
    print("Examples per persona before balancing:")
    for p in target_personas:
        print(f"  {p}: {len(by_persona.get(p, []))}")
    print(f"Using up to {max_per_persona} examples per persona.")

    balanced_examples: List[Dict[str, str]] = []
    import random

    for p in target_personas:
        exs = by_persona.get(p, [])
        if not exs:
            continue
        # downsample if needed
        if len(exs) > max_per_persona:
            exs = random.sample(exs, max_per_persona)
        balanced_examples.extend(exs)

    random.shuffle(balanced_examples)

    # ---- WRITE OUT WITHOUT persona FIELD ----
    with open(OUT_PATH, "w") as f:
        for ex in balanced_examples:
            f.write(
                json.dumps(
                    {
                        "input": ex["input"],
                        "output": ex["output"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(balanced_examples)} balanced training examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
