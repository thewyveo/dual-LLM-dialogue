from typing import List, Dict
from llm_client import call_llm
import re


class AssistantFineTunedAgent:
    """
    Fine-tuned assistant variant.

    Uses a locally fine-tuned Qwen model ("assistant-ft-qwen") that
    has learned from previous prompt-based assistant dialogues.
    """

    def __init__(self, model: str = "assistant-ft-qwen"):
        self.model = model

    def _clean_assistant_output(self, text: str) -> str:
        """
        Post-process the FT model output so it behaves like a single assistant turn,
        not a whole USER/ASSISTANT script.
        """
        cleaned = text.strip()

        # 1) Cut at any point where it starts writing user lines / transcripts
        stop_markers = [
            "\nUser:", "\nUSER:", "\nuser:",
            "\nTraveler:", "\nTRAVELER:",
            "\nAssistant:", "\nassistant:",
            "\nASSISTANT:"
        ]
        cut_idx = len(cleaned)
        for m in stop_markers:
            idx = cleaned.find(m)
            if idx != -1:
                cut_idx = min(cut_idx, idx)
        cleaned = cleaned[:cut_idx].strip()

        # 2) Strip leading "Assistant:" if present
        lower = cleaned.lower()
        if lower.startswith("assistant:"):
            cleaned = cleaned.split(":", 1)[1].strip()

        # 3) Optionally trim to first 3 sentences
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if sentences:
            cleaned = " ".join(sentences[:3]).strip()

        # 4) Remove any accidental INTERNAL tag leakage
        cleaned = cleaned.replace("INTERNAL_REVIEWS", "").strip()
        cleaned = cleaned.replace("internal_reviews", "").strip()

        return cleaned

    def respond(
        self,
        dialogue_history: List[Dict[str, str]],
        candidate_hotels: List[Dict],
    ) -> str:
        hotels_text = ""
        for h in candidate_hotels:
            reviews_str = " | ".join(h.get("review_snippets", []))
            hotels_text += (
                f"[HOTEL]\n"
                f"name: {h['name']}\n"
                f"rating: {h['rating']}\n"
                f"price: {h['price']}\n"
                f"location: {h['location']}\n"
                f"internal_reviews (for assistant only, DO NOT expose directly): {reviews_str}\n\n"
            )

        system_msg = (
            "You are a hotel recommendation assistant.\n"
            "You talk to the traveler in plain text.\n"
            "IMPORTANT RULES:\n"
            "- Only write the ASSISTANT's next reply as natural text.\n"
            "- NEVER write 'USER:' or 'ASSISTANT:' in your answer.\n"
            "- NEVER write a dialogue transcript.\n"
            "- NEVER continue the conversation by inventing new user questions.\n"
            "- Do not invent hotels that are not in the provided list.\n"
            "- You will see internal_reviews for each hotel. "
            "These are INTERNAL_INFORMATION for you only.\n"
            "  * You MUST NOT quote, copy, or closely paraphrase them.\n"
            "  * You may only express high-level properties in your own words "
            "(e.g., 'quiet area', 'close to the center', 'clean', 'good value').\n"
            "  * You MUST NOT say things like 'reviews say' or 'guests said'.\n"
            "- Be concise but helpful, and base your answer on the given candidates.\n"
        )

        kb_msg = (
            "HOTEL CANDIDATES (ONLY choose from these):\n"
            "Remember: internal_reviews are for your reasoning only; "
            "never show them or quote them to the user.\n\n"
            f"{hotels_text}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "system", "content": kb_msg},
        ] + dialogue_history

        raw = call_llm(messages, model=self.model, temperature=0.3, max_tokens=256)
        return self._clean_assistant_output(raw)
