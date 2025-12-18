import json
import os
from typing import Dict, Any


def cleaner(path: str = "profiles.json") -> None:
    """
    Load profiles.json, and set free_form_notes = null (None) for any profile
    whose free_form_notes contains both:
    - some form of "thank" ("Thank"/"thank"/"thanks"/etc.)
    - some form of "recommendation" (including common misspelling "reccomendation")

    The file is overwritten in-place.
    """
    if not os.path.exists(path):
        print(f"[cleaner] No file found at {path}, nothing to do.")
        return

    with open(path, "r", encoding="utf-8") as f:
        try:
            data: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[cleaner] Failed to parse JSON from {path}: {e}")
            return

    modified_count = 0

    for user_id, profile in data.items():
        notes = profile.get("free_form_notes")
        if not isinstance(notes, str):
            continue

        text = notes.lower()

        has_thank = "thank" in text  # matches thank, thanks, thankful, etc.
        has_compliment = ("great" in text) or ("good" in text) or ("excellent" in text) or ("awesome" in text) or ("amazing" in text)
        has_reco = ("recommendation" in text) or ("reccomendation" in text)

        if has_thank or (has_reco and has_compliment):
            profile["free_form_notes"] = None
            modified_count += 1

    # write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[cleaner] Cleaned {modified_count} profile(s) in {path}.")
