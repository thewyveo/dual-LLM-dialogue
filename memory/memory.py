from typing import List, Dict, Optional
import os

# Import the profiler + store
from agents.user_profiler import (
    ProfileStore,
    UserProfile,
    infer_profile_from_session,
)


class Memory:
    """
    Simple memory module.

    - session_memories: full dialogue history per session
    - long_term: list of textual facts or summaries about the user
      (you can still use this for other experiments if you like)
    """

    def __init__(self):
        self.long_term: List[str] = []
        self.session_memories: Dict[str, List[Dict[str, str]]] = {}

    def init_session(self, session_id: str):
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []

    def add_turn(self, session_id: str, role: str, content: str):
        if session_id not in self.session_memories:
            self.init_session(session_id)
        self.session_memories[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.session_memories.get(session_id, [])

    def summarize_and_store_long_term(self, summary: str):
        """
        Store a long-term summary (e.g., about user preferences across sessions).
        """
        self.long_term.append(summary)

    def get_long_term_context(self) -> str:
        if not self.long_term:
            return ""
        return "\n".join(f"- {fact}" for fact in self.long_term)


# =====================================================
#  Profile store + helpers (used by runner & assistant)
# =====================================================

# Put profiles.json at the project root (one level above this file)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROFILES_PATH = os.path.join(_BASE_DIR, "profiles.json")

PROFILE_STORE = ProfileStore(path=_PROFILES_PATH)


def update_memory_with_session_profile(
    user_id: str,
    history: List[Dict[str, str]],
    persona_name: str,
    persona_description: str,
    profiler_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> UserProfile:
    """
    After a finished session, call the LLM profiler and upsert into the JSON store.

    - user_id: stable identifier for this "traveler type"
      (we'll use persona__seed_id, e.g. 'minimalist__min_business_trip').
    - history: full dialogue history for the session.
    - persona_name / persona_description: passed through to the profiler to give context.
    """
    # 1) Infer one session-level profile from this conversation
    session_profile = infer_profile_from_session(
        history=history,
        persona_name=persona_name,
        persona_description=persona_description,
        model=profiler_model,
        temperature=0.1,
    )

    # 2) Upsert into the long-term profile store
    updated_profile = PROFILE_STORE.upsert(user_id, session_profile)
    return updated_profile


def get_profile_prompt_for_user(
    user_id: str,
    default_text: str = "",
) -> str:
    """
    Fetch a compact, bullet-point summary string for the assistant's system prompt.

    - If no profile exists, return default_text.
    - If a profile exists but has no meaningful bullets, also fall back to default_text.
    """
    profile = PROFILE_STORE.get(user_id)
    if not profile:
        return default_text

    summary = profile.to_prompt_summary()
    if not summary or summary.strip() == "":
        return default_text

    return summary
