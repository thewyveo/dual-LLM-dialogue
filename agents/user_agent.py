from typing import Dict, Any, List
from llm_client import call_llm
import re
from utils.repetition_filter import is_semantic_repeat


class UserAgent:
    def __init__(self, persona_name: str, persona_description: str,
                 model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.persona_name = persona_name
        self.persona_description = persona_description
        self.model = model

    def _build_persona_system_msg(self) -> str:
        persona_system_msg = f"""
        You are role-playing the TRAVELER (user), NOT the assistant.

        You are planning a trip and talking to a hotel recommendation assistant.

        STRICT RULES:
        - You ONLY speak as the traveler about your own preferences, questions, feelings, and decisions.
        - NEVER recommend hotels to yourself.
        - NEVER say things like "Based on your criteria, I recommend..." or "Here are some options".
        - YOU are the one making the criteria, requests and decisions.
        - NEVER say "I suggest", "I recommend", or "Based on your requirements".
        - NEVER talk about what "we" recommend, or what "the assistant" will do.
        - You do NOT know anything about the hotel's internal reviews or backend data.
        - You only know what has been said earlier in the conversation.

        When you mention hotels:
        - You may reuse hotel names that have ALREADY appeared in the conversation.
        - Do NOT invent completely new hotel names (like "The Riverside Inn") that haven't been mentioned before.

        Style:
        - 1-2 sentences.
        - Stay in character as the traveler.
        - No "User:" or "Assistant:" prefixes.
        """
        persona_system_msg += f"\n\nPERSONA DETAILS:\n{self.persona_description}\n"
        return persona_system_msg

    def initial_prompt(self, location: str = "Amsterdam") -> str:
        messages = [
            {"role": "system", "content": self._build_persona_system_msg()},
            {
                "role": "user",
                "content": (
                    f"Start the conversation by asking about booking a hotel or restaurant in {location} "
                    "that fits your preferences. Write ONLY the traveler's first message. "
                    "One or two sentences, no dialogue, no 'Assistant:' prefix."
                ),
            },
        ]
        text = call_llm(messages, model=self.model, temperature=0.2, max_tokens=60)
        return self._clean_user_text(text)

    def next_utterance(self, history) -> str:
        """
        history: list of dicts with {'role': 'user'|'assistant', 'content': str}

        NEW:
        - We generate up to 3 candidate utterances.
        - If a candidate is:
            * a semantic/exact repeat of previous traveler utterances, OR
            * it "looks like the assistant" (assistant-style phrasing),
          we discard it (not added to history) and try again with higher temperature.
        - If all 3 attempts are rejected, we return a closure utterance.
        """

        # Build a transcript (Traveler / Assistant) for context
        transcript_lines = []
        user_utterances = []
        for msg in history:
            if msg["role"] == "user":
                transcript_lines.append(f"Traveler: {msg['content']}")
                user_utterances.append(msg["content"])
            else:
                transcript_lines.append(f"Assistant: {msg['content']}")

        transcript = "\n".join(transcript_lines)

        # Explicitly show the traveler what they've already asked
        if user_utterances:
            previous_questions_block = (
                "Here are the things you (the traveler) have ALREADY asked about in this conversation:\n"
                + "\n".join(f"- {u}" for u in user_utterances)
            )
        else:
            previous_questions_block = (
                "You haven't asked anything yet in this conversation."
            )

        # --- early vs. late conversation behavior ---
        num_messages = len(history)
        early_convo = num_messages < 6  # around first 3 utterances total
        late_convo = num_messages > 12

        if early_convo:
            behavior_block = (
                "You are still EARLY in the conversation.\n"
                "- You are NOT allowed to end the conversation yet.\n"
                "- Do NOT say things like 'that's perfect', 'that's all', "
                "'goodbye', or 'I'll book it now'.\n"
                "- You must NOT repeat questions you already asked above.\n"
                "- Instead, ask ONE new, concise follow-up question or request "
                "ONE new piece of information about the hotels/restaurants, "
                "their location, price, amenities, or something about your own preferences."
            )
        elif late_convo:
            behavior_block = (
                "You are in the LATE stage of the conversation.\n"
                "- You should now clearly express that you are satisfied and that the conversation can end.\n"
                "- End in 1–2 sentences, for example by thanking "
                "the assistant and saying you are done (e.g. 'Thanks, that's all, goodbye.').\n"
            )
        else:
            behavior_block = (
                "You are now allowed to either:\n"
                "- Ask ONE more follow-up question or clarification, OR\n"
                "- Clearly express that you are satisfied and that the conversation can end.\n"
                "If you decide to end, do it in 1–2 sentences, for example by thanking "
                "the assistant and saying you are done (e.g. 'Thanks, that's all, goodbye.').\n"
                "You must NOT repeat questions you already asked above."
            )

        user_msg = (
            "Here is the conversation so far:\n\n"
            f"{transcript}\n\n"
            f"{previous_questions_block}\n\n"
            f"{behavior_block}\n\n"
            "Now respond with the NEXT thing the traveler would say.\n"
            "Remember: 1–2 sentences, stay in character, no 'Assistant:' prefix."
        )

        base_messages = [
            {"role": "system", "content": self._build_persona_system_msg()},
            {"role": "user", "content": user_msg},
        ]

        # We only compare against previous traveler utterances
        previous_user_utts = user_utterances

        max_attempts = 3
        duplicate_semantic_threshold = 0.7
        assistant_semantic_threshold = 0.9
        temperature = 0.3
        last_candidate = ""

        for attempt in range(max_attempts):
            text = call_llm(
                base_messages,
                model=self.model,
                temperature=temperature,
                max_tokens=80,
            )
            cleaned = self._clean_user_text(text)
            last_candidate = cleaned

            # 1) Reject if it looks like the assistant talking
            if self._looks_like_assistant(
                cleaned,
                semantic_threshold=assistant_semantic_threshold,
            ):
                temperature = min(temperature + 0.2, 0.9)
                continue

            # 2) Reject if it's a repetition of previous traveler utterances
            if self._is_repetition(
                cleaned,
                previous_user_utts,
                semantic_threshold=duplicate_semantic_threshold,
            ):
                temperature = min(temperature + 0.2, 0.9)
                continue

            # 3) Accept this utterance – caller will add it to history
            return cleaned

        # If all attempts were rejected as repeats / assistant-like,
        # return a closure utterance (force actual end).
        return "Thank you, that was very helpful. That's all I needed, goodbye."

    def _clean_user_text(self, text: str) -> str:
        """
        Make sure the output is a single, natural traveler utterance.
        """
        cleaned = text.strip()

        # 1) Only keep text up to the first time the model starts re-narrating roles
        stop_markers = ["\nUser:", "\nAssistant:", "\nTraveler:",
                        "\nUSER:", "\nASSISTANT:", "\nTRAVELER:"]
        cut_idx = len(cleaned)
        for m in stop_markers:
            idx = cleaned.find(m)
            if idx != -1:
                cut_idx = min(cut_idx, idx)
        cleaned = cleaned[:cut_idx].strip()

        # 2) Only keep the first line (prevents it from writing multi-turn scripts)
        first_line = cleaned.split("\n", 1)[0].strip()

        # 3) Strip common speaker prefixes like "User:", "Assistant:", "Traveler:"
        prefixes = ["user:", "assistant:", "traveler:", "system:"]
        lower = first_line.lower()
        for p in prefixes:
            if lower.startswith(p):
                first_line = first_line[len(p):].strip()
                break

        # 4) Reduce to the first 1–2 sentences
        sentences = re.split(r"(?<=[.!?])\s+", first_line)
        if sentences:
            first_line = " ".join(sentences[:2]).strip()

        return first_line

    def _looks_like_assistant(
        self,
        utterance: str,
        semantic_threshold: float = 0.90,
    ) -> bool:
        """
        Detect if the traveler utterance is actually written in an assistant-like style.

        Uses:
        - Cheap lexical patterns (fast).
        - Semantic similarity (BERTScore via is_semantic_repeat) to typical assistant phrases.
        """
        if not utterance:
            return False

        t = utterance.strip().lower()

        # 1) Hard lexical patterns that almost surely mean "assistant voice"
        hard_patterns = [
            "based on your criteria",
            "based on your requirements",
            "based on your preferences",
            "i would recommend",
            "i recommend",
            "i suggest",
            "here are some options",
            "here are a few options",
            "would you like me to recommend",
            "would you like me to suggest",
            "as the assistant",
            "we recommend",
        ]
        if any(p in t for p in hard_patterns):
            return True

        # 2) Semantic similarity to canonical assistant-ish prototypes
        assistant_prototypes = [
            "Based on your criteria, I would recommend this hotel.",
            "Based on your requirements, here are some options.",
            "Here are some options I suggest for your stay.",
            "I would recommend the following hotels.",
            "I suggest you consider these options.",
            "Based on your preferences, I recommend this place.",
            "Would you like me to recommend some hotels?",
            "Do you have any specific amenities or services you prioritize?",
        ]

        # Reuse is_semantic_repeat: compare utterance to prototypes
        try:
            if is_semantic_repeat(utterance, assistant_prototypes, threshold=semantic_threshold):
                return True
        except Exception:
            # Fail-safe: if BERTScore or repetition_filter breaks, don't crash the loop
            return False

        return False

    def _is_repetition(
        self,
        utterance: str,
        previous_utts: List[str],
        semantic_threshold: float = 0.92,
    ) -> bool:
        """
        Check if `utterance` is a repeat of any previous traveler utterance.

        - Exact match => repeat
        - High BERTScore similarity => semantic repeat
        """
        if not previous_utts:
            return False

        normalized_new = utterance.strip().lower()
        if not normalized_new:
            return False

        prev_lower = [u.strip().lower() for u in previous_utts]

        # Exact repeat against any previous user utterance
        if normalized_new in prev_lower:
            return True

        # Semantic repeat (BERTScore)
        try:
            if is_semantic_repeat(utterance, previous_utts, threshold=semantic_threshold):
                return True
        except Exception:
            # Fail-safe: if BERTScore fails, just don't treat it as semantic repeat
            return False

        return False

    def _avoid_repetition(self, utterance: str, history, min_repeats_before_stop: int = 2) -> str:
        """
        (Currently unused with the new 3-attempt loop, but kept for backwards compatibility.)

        Use the conversation memory to avoid pathological loops:

        - If the model produces an utterance that is EXACTLY the same as a previous
          traveler message, we count how many times it has already appeared.
        - Only if it has appeared at least `min_repeats_before_stop` times
          (i.e. this would be the 3rd+ time) do we convert it into a polite
          'I'm satisfied' message.
        """
        normalized_new = utterance.strip().lower()
        if not normalized_new:
            return utterance

        previous_user_utts = [
            m["content"].strip().lower()
            for m in history
            if m["role"] == "user"
        ]

        repeat_count = previous_user_utts.count(normalized_new)

        if repeat_count >= min_repeats_before_stop:
            return "Thank you, that was very helpful. That's all I needed, goodbye."

        return utterance

    def check_satisfaction(self, user_utterance: str) -> bool:
        text = user_utterance.lower()

        # 1) Explicit closure patterns: clearly done
        if (
            ("thank you" in text or "thanks" in text)
            and ("that's all" in text or "that is all" in text or "all i needed" in text)
        ):
            return True

        if ("thank you" in text or "thanks" in text) and "goodbye" in text:
            return True

        # 2) Very explicit booking confirmation
        if re.search(r"\b(book it|i'll take it|i will take it)\b", text):
            return True

        # 3) Explicit give-up phrases
        give_up_patterns = [
            r"\bi'll look elsewhere\b",
            r"\bconversation over\b",
            r"\bi'm done here\b",
            r"\bno thanks,? (i'm )?done\b",
            r"\bsearch elsewhere\b",
        ]
        for pat in give_up_patterns:
            if re.search(pat, text):
                return True

        # IMPORTANT: we DO NOT treat generic positivity as satisfaction anymore
        # e.g. "that sounds great", "perfect", etc. will NOT end the conversation.

        return False


PERSONA_MINIMALIST = """You are efficient, pragmatic, and dislike small talk.
You value time and want a hotel that fits your constraints quickly.
You ask a few clear questions and then decide.
You prefer concise, factual answers rather than long explanations."""

PERSONA_EXPLORER = """You are curious and high in openness.
You care a lot about atmosphere, uniqueness, and local experiences.
You tend to ask several follow-up questions about ambiance, neighborhood, and special features.
You may consider multiple options before deciding and enjoy a bit of conversation."""
