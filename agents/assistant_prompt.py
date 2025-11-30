from typing import List, Dict, Optional
from llm_client import call_llm
import re
from utils.repetition_filter import is_semantic_repeat
from memory.memory import get_profile_prompt_for_user


class AssistantPromptAgent:
    """
    Prompt-based assistant (baseline).
    Uses few-shot style prompting and the hotel candidates provided for each turn.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        user_id: Optional[str] = None,
    ):
        self.model = model
        # user_id is used to fetch long-term profile (e.g., persona name)
        self.user_id = user_id

    def build_system_prompt(self, profile_snippet: str = "") -> str:
        base = """
        You are a hotel recommendation assistant.

        You MUST follow these rules:

        GENERAL BEHAVIOR
        ----------------
        • You MUST base your suggestions ONLY on the provided hotel candidates and their INTERNAL reviews.
        • The reviews you see are INTERNAL_INFORMATION for you only.
        • You MUST NOT quote, copy, or closely paraphrase these reviews.
        • You may only express high-level properties in your own words
        (e.g., “quiet area”, “close to the center”, “clean”, “good value”).
        • You MUST NOT say things like “reviews say”, “guests said”, or reference review text.
        • NEVER invent hotels that are not in the provided list.
        • NEVER infer or invent real-world facts about Amsterdam, neighborhoods, geography, or attractions.
        • Base ALL hotel descriptions ONLY on the synthetic seed/hotel memory provided.

        NO HALLUCINATIONS
        -----------------
        • If you do NOT know whether a hotel has a feature (parking, shuttle, breakfast, rooftop, gym, etc.),
        you MUST say: “I don't have information about that specific detail.”
        • Do NOT speculate, fill in missing details, or invent amenities, services, or features.
        • You must NOT invent named locations such as restaurants, cafés, bars, shops, landmarks, or attractions
        unless they are explicitly provided in the synthetic seed/memory.
        • If needed, use generic wording like “a nearby café” or “a restaurant in the area.”

        DIALOGUE CONDUCT
        ----------------
        • You MUST ask clarifying questions if the user’s constraints are unclear.
        • Compare 2-3 hotels concisely when appropriate.
        • End each message by suggesting ONE best option.
        • You are ONLY the assistant: write ONLY your own next reply.
        • Do NOT write what the user says. Do NOT produce transcripts.
        • Do NOT prefix messages with “User:” or “Assistant:”.
        • Respond naturally, concisely, and helpfully.

        Your objective:
        Provide accurate, grounded, high-level hotel recommendations strictly within the synthetic information given,
        without quoting reviews, inventing details, or leaking internal data.
        """

        base = base.strip()

        if profile_snippet:
            base += (
                "\n\nTRAVELER LONG-TERM PROFILE\n"
                "---------------------------\n"
                f"{profile_snippet}\n"
            )

        return base

    def format_hotels_for_prompt(self, hotels: List[Dict]) -> str:
        if not hotels:
            return "No hotel candidates were found for this query.\n"

        lines = []
        for h in hotels:
            snippet_texts = h.get("review_snippets", [])
            # We still pass snippets in, but mark them clearly as INTERNAL so the model
            # knows they are not to be exposed verbatim to the user.
            snippet = " | ".join(snippet_texts[:2]) if snippet_texts else "No review snippets."
            lines.append(
                f"- {h['name']} (rating: {h['rating']}, "
                f"price: {h['price']}, location: {h['location']})\n"
                f"  INTERNAL_REVIEWS (for assistant only, DO NOT expose to user): {snippet}"
            )
        return "\n".join(lines)

    def _clean_assistant_output(self, text: str) -> str:
        """
        Ensure we only keep the assistant's single turn, not a whole script.
        Also trims weird roleplay markers if they appear.
        """
        cleaned = text.strip()

        # 1) Cut at any point where the model starts writing user lines / transcripts
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

        # 3) Optionally trim to first 3 sentences to keep it concise
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if sentences:
            cleaned = " ".join(sentences[:3]).strip()

        # 4) Just in case the model ever prints the literal INTERNAL_REVIEWS tag
        cleaned = cleaned.replace("INTERNAL_REVIEWS", "").strip()

        return cleaned

    def respond(
        self,
        dialogue_history: List[Dict[str, str]],
        candidate_hotels: List[Dict],
    ) -> str:
        # Fetch long-term profile snippet if we have a user_id
        if self.user_id:
            profile_snippet = get_profile_prompt_for_user(
                self.user_id,
                default_text=""
            )
        else:
            profile_snippet = ""

        system_prompt = self.build_system_prompt(profile_snippet)
        hotels_text = self.format_hotels_for_prompt(candidate_hotels)
        assistant_context = (
            "Here are the hotel candidates you MUST choose from:\n"
            f"{hotels_text}\n"
        )

        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": assistant_context},
        ] + dialogue_history

        # Collect previous assistant utterances to avoid repeating ourselves
        previous_assistant_utts = [
            m["content"]
            for m in dialogue_history
            if m["role"] == "assistant"
        ]

        temperature = 0.3
        max_attempts = 3
        last_cleaned = ""

        for attempt in range(max_attempts):
            raw = call_llm(
                base_messages,
                model=self.model,
                temperature=temperature,
                max_tokens=256,
            )
            cleaned = self._clean_assistant_output(raw)
            last_cleaned = cleaned

            if not previous_assistant_utts:
                return cleaned

            # If not too similar, accept
            if not is_semantic_repeat(cleaned, previous_assistant_utts, threshold=0.92):
                return cleaned

            # Otherwise, try again with a slightly higher temperature
            temperature = min(temperature + 0.2, 0.9)

        # If all attempts look similar, just return the last one
        # (we don't force closure for the assistant)
        return last_cleaned
