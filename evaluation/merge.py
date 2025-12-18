import os
import json
from typing import List, Dict, Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_DIR = os.path.join(BASE_DIR, "results", "conversations")
OUT_PATH = os.path.join(CONV_DIR, "merged_conversations.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if not os.path.exists(CONV_DIR):
        raise FileNotFoundError(f"Directory not found: {CONV_DIR}")

    merged: List[Dict[str, Any]] = []

    files = sorted(
        f for f in os.listdir(CONV_DIR)
        if f.startswith("conversations") and f.endswith(".json")
    )

    if not files:
        print("[merge] No conversation files found.")
        return

    for fname in files:
        path = os.path.join(CONV_DIR, fname)
        print(f"[merge] Loading {fname}")

        data = load_json(path)

        if isinstance(data, list):
            merged.extend(data)
        elif isinstance(data, dict):
            merged.append(data)
        else:
            print(f"[merge] Skipping {fname}: unsupported format")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"[merge] Merged {len(merged)} conversations → {OUT_PATH}")


if __name__ == "__main__":
    main()
