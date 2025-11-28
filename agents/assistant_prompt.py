from typing import List, Dict
from llm_client import call_llm
import re


class AssistantPromptAgent:
    """
    Prompt-based assistant (baseline).
    Uses few-shot style prompting and the hotel candidates provided for each turn.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model = model

    def build_system_prompt(self) -> str:
        return (
            "You are a hotel recommendation assistant.\n"
            "You MUST base your suggestions ONLY on the provided hotel candidates and their reviews.\n"
            "Your job is to:\n"
            "1) Ask clarifying questions if the user's constraints are unclear.\n"
            "2) Compare 2-3 hotels, explaining concise pros and cons.\n"
            "3) Suggest ONE best option at the end of each message.\n"
            "4) Never invent hotels that are not in the provided list.\n"
            "5) You are ONLY the assistant. You must write ONLY your own next reply.\n"
            "   Never write what the user says. Never write a dialogue transcript.\n"
            "6) Do NOT prefix with 'User:' or 'Assistant:'. Just answer naturally as the assistant."
        )

    def format_hotels_for_prompt(self, hotels: List[Dict]) -> str:
        if not hotels:
            return "No hotel candidates were found for this query.\n"
        lines = []
        for h in hotels:
            snippet_texts = h.get("review_snippets", [])
            snippet = " | ".join(snippet_texts[:2]) if snippet_texts else "No review snippets."
            lines.append(
                f"- {h['name']} (rating: {h['rating']}, "
                f"price: {h['price']}, location: {h['location']})\n"
                f"  Reviews: {snippet}"
            )
        return "\n".join(lines)

    def _clean_assistant_output(self, text: str) -> str:
        """
        Ensure we only keep the assistant's single turn, not a whole script.
        """
        cleaned = text.strip()

        # 1) Cut at any point where the model starts writing user lines
        stop_markers = ["\nUser:", "\nTraveler:", "\nTRAVELER:",
                        "\nUSER:", "\nassistant:", "\nAssistant:"]
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

        # 3) Optionally trim to first 3 sentences to keep it concise
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if sentences:
            cleaned = " ".join(sentences[:3]).strip()

        return cleaned

    def respond(
        self,
        dialogue_history: List[Dict[str, str]],
        candidate_hotels: List[Dict],
    ) -> str:
        system_prompt = self.build_system_prompt()
        hotels_text = self.format_hotels_for_prompt(candidate_hotels)
        assistant_context = (
            "Here are the hotel candidates you MUST choose from:\n"
            f"{hotels_text}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": assistant_context},
        ] + dialogue_history

        raw = call_llm(messages, model=self.model, temperature=0.3, max_tokens=256)
        return self._clean_assistant_output(raw)
