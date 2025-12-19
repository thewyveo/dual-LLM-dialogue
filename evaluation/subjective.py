import json
import os
import sys
from typing import Dict, List
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm_client import call_llm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, "data", "merged_conversations.json")
OUT_PATH = os.path.join(BASE_DIR, "results", "subjective_metrics.json")


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator of conversational AI systems.

You will be given a FULL dialogue between a user and a hotel recommendation assistant.

Your task:
- Judge the assistant's performance objectively.
- Do NOT assume access to real-world facts.
- Judge ONLY based on what is said in the dialogue.

IMPORTANT:
- Do NOT reward hallucinated details.
- Penalize invented amenities, locations, restaurants, or facts.
- Penalize answers that speculate instead of saying "I don't know".
- Do NOT reward verbosity for its own sake.

Score each dimension from 1 (very poor) to 5 (excellent).
Be strict but fair.
OUTPUT A JSON object ONLY.
"""


def format_dialogue(history: List[Dict[str, str]]) -> str:
    lines = []
    for m in history:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def build_judge_prompt(dialogue_text: str) -> str:
    return f"""
Evaluate the following conversation.

CONVERSATION
------------
{dialogue_text}

SCORING INSTRUCTIONS
--------------------
Return a JSON object with the following fields:

- task_fulfillment (1-5)
- groundedness (1-5)
- clarity (1-5)
- pleasantness (1-5)
- overall_quality (1-5)
- binary_success ("yes" or "no")
- justification (1-2 sentences explaining the scores)

ONLY return valid JSON.
"""


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(LOGS_PATH, "r") as f:
        conversations = json.load(f)

    results = []

    for idx, conv in enumerate(conversations):
        print(f"Judging conversation {idx+1}/{len(conversations)}")

        dialogue_text = format_dialogue(conv["history"])
        judge_prompt = build_judge_prompt(dialogue_text)

        response = call_llm(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            model="Qwen/Qwen2.5-3B-Instruct",
            temperature=0.0,
            max_tokens=256,
        )

        try:
            parsed = json.loads(response)
        except Exception:
            print("Judge output could not be parsed, storing raw output.")
            parsed = {"raw_output": response}

        results.append({
            "session_id": conv["session_id"],
            "assistant_variant": conv["assistant_variant"],
            "persona": conv["persona"],
            "initial_seed_id": conv.get("initial_seed_id"),
            "long_term_memory_profile": conv.get("long_term_memory_profile", True),
            "scores": parsed,
        })

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Subjective LLM evaluation saved to {OUT_PATH}")


if __name__ == "__main__":
    main()