from typing import List, Dict
from llm_client import call_llm


class AssistantFineTunedAgent:
    """
    Fine-tuned assistant variant.

    For now this just uses a different model name and a slightly lighter prompt,
    assuming the behavior is learned during fine-tuning.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model = model

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
                f"reviews: {reviews_str}\n\n"
            )

        system_msg = (
            "You are a fine-tuned hotel recommendation assistant.\n"
            "You know how to interpret hotel metadata and user preferences.\n"
            "You must choose only among the candidate hotels provided.\n"
        )
        kb_msg = (
            "HOTEL CANDIDATES (ONLY choose from these):\n"
            f"{hotels_text}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "system", "content": kb_msg},
        ] + dialogue_history

        return call_llm(messages, model=self.model, temperature=0.3, max_tokens=256)
