from typing import List, Dict, Any
import json
import os

from runner.conversation_loop import run_conversation


def run_batch(
    n_histories: int = 10,
    personas: tuple = ("minimalist", "explorer"),
    assistant_variants: tuple = ("prompt", "ft"),
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # Base path stuff
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for persona in personas:
        for variant in assistant_variants:
            for i in range(n_histories):
                print(f"Running convo {i+1}/{n_histories} | persona={persona} | variant={variant}")
                res = run_conversation(
                    persona=persona,
                    assistant_variant=variant,
                    max_turns=15,
                    location="Amsterdam",
                )
                results.append(res)

                # 🔹 NEW: save each conversation immediately
                single_out_path = os.path.join(
                    logs_dir,
                    f"conv_{persona}_{variant}_{i}.json"
                )
                try:
                    with open(single_out_path, "w") as f_single:
                        json.dump(res, f_single, indent=2)
                    print(f"  Saved conversation to {single_out_path}")
                except Exception as e:
                    print(f"  WARNING: Failed to save per-convo log {single_out_path}: {e}")

    # 🔹 OLD behavior: combined log of all conversations
    combined_out_path = os.path.join(logs_dir, "conversations.json")
    try:
        with open(combined_out_path, "w") as f_all:
            json.dump(results, f_all, indent=2)
        print(f"All conversations saved to {combined_out_path}")
    except Exception as e:
        print(f"WARNING: Failed to save combined conversations file: {e}")

    return results
