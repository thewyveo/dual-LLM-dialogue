from typing import List, Dict, Any
import json
import os
import random

from runner.conversation_loop import run_conversation
from data.initial_histories import INITIAL_HISTORIES


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

    if type(personas) is str:
        personas = (personas,)

    if type(assistant_variants) is str:
        assistant_variants = (assistant_variants,)

    for persona in personas:
        # Filter initial histories that match this persona
        persona_seeds = [h for h in INITIAL_HISTORIES if h["persona"] == persona]
        if not persona_seeds:
            print(f"WARNING: no initial histories found for persona='{persona}'")
            continue

        for variant in assistant_variants:
            for i in range(n_histories):
                # Pick a seed for this run (you could also cycle deterministically)
                seed = random.choice(persona_seeds)
                initial_history = seed["messages"]
                location = seed.get("location", "Amsterdam")
                seed_id = seed["id"]

                print(
                    f"Running convo {i+1}/{n_histories} | "
                    f"persona={persona} | variant={variant} | seed={seed_id}"
                )

                res = run_conversation(
                    persona=persona,
                    assistant_variant=variant,
                    max_turns=15,
                    location=location,
                    initial_history=initial_history,
                    seed_id=seed_id,   # NEW: feed through for profiling key
                )

                # Attach seed id for later analysis (also returned inside res now)
                res["initial_seed_id"] = seed_id
                results.append(res)

                # Save each conversation immediately
                single_out_path = os.path.join(
                    logs_dir,
                    f"individual_logs/conv_{persona}_{variant}_{i}.json"
                )
                try:
                    os.makedirs(os.path.dirname(single_out_path), exist_ok=True)
                    with open(single_out_path, "w") as f_single:
                        json.dump(res, f_single, indent=2)
                    print(f"  Saved conversation to {single_out_path}")
                except Exception as e:
                    print(f"  WARNING: Failed to save per-convo log {single_out_path}: {e}")

    # Combined log of all conversations
    combined_out_path = os.path.join(logs_dir, "conversations.json")
    try:
        with open(combined_out_path, "w") as f_all:
            json.dump(results, f_all, indent=2)
        print(f"All conversations saved to {combined_out_path}")
    except Exception as e:
        print(f"WARNING: Failed to save combined conversations file: {e}")

    return results
