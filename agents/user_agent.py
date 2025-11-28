from typing import Dict, Any, List
from llm_client import call_llm
import re


class UserAgent:
    def __init__(self, persona_name: str, persona_description: str,
                model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.persona_name = persona_name
        self.persona_description = persona_description
        self.model = model

    def _build_persona_system_msg(self) -> str:
        return (
            "You are a TRAVELER talking to a hotel/restaurant assistant.\n"
            f"Persona: {self.persona_description}\n\n"
            "CRITICAL RULES:\n"
            "- You are ONLY allowed to write the TRAVELER's next message.\n"
            "- NEVER write what the assistant says.\n"
            "- NEVER write a dialogue transcript.\n"
            "- NEVER prefix with 'User:', 'Assistant:', 'Traveler:' or similar.\n"
            "- Always speak in first person as the traveler (\"I\").\n"
            "- Answer in 1–2 short natural sentences only.\n"
            "- Be kind, polite, and patient.\n"
            "If you conclude the conversation is over, say so clearly "
            "(e.g., \"Thank you, that's all, goodbye.\")"
        )

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
        """
        transcript = ""
        for msg in history:
            if msg["role"] == "user":
                transcript += f"Traveler: {msg['content']}\n"
            else:
                transcript += f"Assistant: {msg['content']}\n"

        messages = [
            {"role": "system", "content": self._build_persona_system_msg()},
            {
                "role": "user",
                "content": (
                    "Here is the conversation so far:\n\n"
                    f"{transcript}\n"
                    "Now write ONLY the NEXT thing the traveler would say.\n"
                    "- Do NOT write any assistant lines.\n"
                    "- Do NOT write a dialogue transcript.\n"
                    "- One or two sentences max.\n"
                    "- No prefixes like 'User:' or 'Assistant:'."
                ),
            },
        ]
        text = call_llm(messages, model=self.model, temperature=0.2, max_tokens=80)
        return self._clean_user_text(text)

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

    def check_satisfaction(self, user_utterance: str) -> bool:
        text = user_utterance.lower()

        positive_patterns = [
            r"\bbook it\b",
            r"\bthat works\b",
            r"\bthis works\b",
            r"\bperfect\b",
            r"\bi'll take it\b",
            r"\bthat sounds great\b",
            r"\bthank you, that's all\b",
        ]

        give_up_patterns = [
            r"\bi'll look elsewhere\b",
            r"\bconversation over\b",
            r"\bi'm done here\b",
            r"\bno thanks\b",
            r"\bsearch elsewhere\b",
            r"\bgoodbye\b",
        ]

        for pat in positive_patterns + give_up_patterns:
            if re.search(pat, text):
                return True

        return False


PERSONA_MINIMALIST = """You are efficient, pragmatic, and dislike small talk.
You value time and want a hotel that fits your constraints quickly.
You ask a few clear questions and then decide.
You prefer concise, factual answers rather than long explanations."""

PERSONA_EXPLORER = """You are curious and high in openness.
You care a lot about atmosphere, uniqueness, and local experiences.
You tend to ask several follow-up questions about ambiance, neighborhood, and special features.
You may consider multiple options before deciding and enjoy a bit of conversation."""
