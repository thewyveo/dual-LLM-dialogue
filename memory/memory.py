from typing import List, Dict


class Memory:
    """
    Simple memory module.

    - session_memories: full dialogue history per session
    - long_term: list of textual facts or summaries about the user
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
