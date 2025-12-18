from typing import List, Dict, Any
import json
import os
import random

from runner.conversation_loop import run_conversation
from data.initial_histories import INITIAL_HISTORIES
from utils.profile_cleaner import cleaner
from memory.memory import set_profile_store
import shutil


def run_batch(
    n_histories: int = 10,
    personas: tuple = ("minimalist", "explorer"),
    assistant_variants: tuple = ("prompt", "ft"),
    use_memory: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # base path stuff
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if type(personas) is str:
        personas = (personas,)

    if type(assistant_variants) is str:
        assistant_variants = (assistant_variants,)

    for persona in personas:
        # filter initial histories that match this persona
        persona_seeds = [h for h in INITIAL_HISTORIES if h["persona"] == persona]
        if not persona_seeds:
            print(f"WARNING: no initial histories found for persona='{persona}'")
            continue

        for variant in assistant_variants:
            profile_path = f"profiles_{variant}.json"

            if not os.path.exists(profile_path):
                shutil.copyfile("profiles_beginning.json", profile_path)

            set_profile_store(profile_path)
            for i in range(n_histories):
                # pick a seed for this run (you could also cycle deterministically)
                seed = random.choice(persona_seeds)
                initial_history = seed["messages"]
                location = seed.get("location", "Amsterdam")
                seed_id = seed["id"]

                print(
                    f"Running convo {i+1}/{n_histories} | "
                    f"persona={persona} | variant={variant} | seed={seed_id} | long_term_memory={use_memory}"
                )

                res = run_conversation(
                    persona=persona,
                    assistant_variant=variant,
                    max_turns=15,
                    location=location,
                    initial_history=initial_history,
                    seed_id=seed_id,   
                    long_term_memory_profile=use_memory, 
                )

                # attach seed id for later analysis (also returned inside res now)
                res["initial_seed_id"] = seed_id
                results.append(res)

                # save each conversation immediately
                single_out_path = os.path.join(
                    logs_dir,
                    f"individual_logs/conv_{persona}_{variant}_{i}.json"
                )
                try:
                    os.makedirs(os.path.dirname(single_out_path), exist_ok=True)
                    with open(single_out_path, "w") as f_single:
                        json.dump(res, f_single, indent=2)
                    print(f"  Saved conversation to {single_out_path}")
                    if "bootstrap" in profile_path or "beginning" in profile_path:
                        print(f"[cleaner] Refusing to clean protected profile file: {profile_path}")
                    else:
                        cleaner(profile_path)
                except Exception as e:
                    print(f"  WARNING: Failed to save per-convo log {single_out_path}: {e}")

    # combined log of all conversations
    combined_out_path = os.path.join(logs_dir, "conversations.json")
    try:
        with open(combined_out_path, "w") as f_all:
            json.dump(results, f_all, indent=2)
        print(f"All conversations saved to {combined_out_path}")
    except Exception as e:
        print(f"WARNING: Failed to save combined conversations file: {e}")

    return results
