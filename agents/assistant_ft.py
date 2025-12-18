from typing import List, Dict, Optional
from llm_client import call_llm
import re
from utils.repetition_filter import is_semantic_repeat
from memory.memory import get_profile_prompt_for_user


class AssistantFineTunedAgent:
    """
    Fine-tuned assistant variant.

    Behaves like the prompt-based assistant, but uses the locally fine-tuned
    Qwen model ("assistant-ft-qwen") instead of a general instruction model.

    This agent is used both for the FT and PEFT variants, depending on the called model name.
    """

    def __init__(
        self,
        model: str = "assistant-ft-qwen",
        user_id: Optional[str] = None,
    ):
        self.model = model
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
        """
        Format hotel candidates for the LLM in a controlled, safe way.
        Gracefully supports optional fields such as:
        - neighborhood
        - distance_to_center_km
        - amenities
        - review_snippets (INTERNAL ONLY)
        """

        if not hotels:
            return "No hotel candidates were found for this query.\n"

        lines = []
        for h in hotels:
            name = h.get("name", "Unknown Hotel")
            rating = h.get("rating", "N/A")
            price = h.get("price", "?")
            location = h.get("location", "Unknown")

            # optional fields (shows only if they exist)
            neighborhood = h.get("neighborhood")
            distance_km = h.get("distance_to_center_km")
            amenities = h.get("amenities", [])

            # compact description
            desc_parts = [
                f"rating: {rating}",
                f"price: {price}",
                f"location: {location}",
            ]

            if neighborhood:
                desc_parts.append(f"neighborhood: {neighborhood}")

            if distance_km is not None:
                desc_parts.append(f"distance_to_center_km: {distance_km:.1f}")

            if amenities:
                am_str = ", ".join(amenities)
                desc_parts.append(f"amenities: {am_str}")

            desc = ", ".join(desc_parts)

            # review snippets remain INTERNAL ONLY
            snippet_texts = h.get("review_snippets", [])
            if snippet_texts:
                internal_snippet = " | ".join(snippet_texts[:3])
            else:
                internal_snippet = "No review snippets."

            lines.append(
                f"- {name} ({desc})\n"
                f"  INTERNAL_REVIEWS (assistant only): {internal_snippet}"
            )

        return "\n".join(lines)

    def _clean_assistant_output(self, text: str) -> str:
        """
        Ensure we only keep the assistant's single turn, not a whole script.
        Also trims weird roleplay markers if they appear.
        """
        cleaned = text.strip()

        # 1) cut at any point where the model starts writing user lines / transcripts
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

        # 2) strip "Assistant:" if present
        lower = cleaned.lower()
        if lower.startswith("assistant:"):
            cleaned = cleaned.split(":", 1)[1].strip()

        # 3) optionally trim to first 3 sentences to keep it concise
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if sentences:
            cleaned = " ".join(sentences[:3]).strip()

        # 4) just in case the model ever prints the literal INTERNAL_REVIEWS tag
        cleaned = cleaned.replace("INTERNAL_REVIEWS", "").strip()
        cleaned = cleaned.replace("internal_reviews", "").strip()

        return cleaned

    def respond(
        self,
        dialogue_history: List[Dict[str, str]],
        candidate_hotels: List[Dict],
    ) -> str:
        # fetch long-term profile snippet if there is a user_id
        if self.user_id:
            profile_snippet = get_profile_prompt_for_user(
                self.user_id,
                default_text=""
            )
        else: # (fallback)
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

        # collect previous assistant utterances to avoid repeating
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
                model=self.model,          # <-- fine-tuned model
                temperature=temperature,
                max_tokens=256,
            )
            cleaned = self._clean_assistant_output(raw)
            last_cleaned = cleaned

            if not previous_assistant_utts:
                return cleaned

            # if not too similar, accept
            if not is_semantic_repeat(cleaned, previous_assistant_utts, threshold=0.92):
                return cleaned

            # otherwise, try again with a slightly higher temperature
            temperature = min(temperature + 0.2, 0.9)

        # if all attempts look similar, just return the last one
        return last_cleaned
