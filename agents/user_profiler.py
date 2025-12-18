from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from llm_client import call_llm


@dataclass
class UserProfile:
    """
    Long-term preference profile for a traveler.

    All booleans can be True / False / None (unknown).
    Lists are de-duplicated when merging.
    """

    # trip context
    trip_type: Optional[str] = None
    persona_name: Optional[str] = None

    # budget
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    currency: Optional[str] = "EUR"

    # location preferences
    wants_central_location: Optional[bool] = None
    wants_local_neighborhood: Optional[bool] = None

    # atmosphere
    prefers_quiet: Optional[bool] = None
    prefers_social: Optional[bool] = None

    # amenities
    cares_about_wifi: Optional[bool] = None
    cares_about_desk: Optional[bool] = None
    cares_about_breakfast: Optional[bool] = None
    cares_about_parking: Optional[bool] = None
    cares_about_gym: Optional[bool] = None
    cares_about_rooftop: Optional[bool] = None
    cares_about_spa: Optional[bool] = None

    # thematic preferences
    foodie: Optional[bool] = None
    romantic: Optional[bool] = None

    # hotel-specific signals
    preferred_hotels: List[str] = field(default_factory=list)
    rejected_hotels: List[str] = field(default_factory=list)

    # free-form notes the assistant can see
    free_form_notes: Optional[str] = None

    # internal: how many sessions contributed to this profile
    sessions_count: int = 0

    # --- helpers ---

    @classmethod
    def from_llm_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """
        Build a UserProfile from a dict returned by the profiler LLM.
        Unknown keys are ignored; missing keys stay as defaults.
        """
        kwargs: Dict[str, Any] = {}

        scalar_fields = [
            "trip_type",
            "persona_name",
            "budget_min",
            "budget_max",
            "currency",
            "wants_central_location",
            "wants_local_neighborhood",
            "prefers_quiet",
            "prefers_social",
            "cares_about_wifi",
            "cares_about_desk",
            "cares_about_breakfast",
            "cares_about_parking",
            "cares_about_gym",
            "cares_about_rooftop",
            "cares_about_spa",
            "foodie",
            "romantic",
            "free_form_notes",
        ]

        for key in scalar_fields:
            if key in data:
                kwargs[key] = data[key]

        preferred = data.get("preferred_hotels", []) or []
        rejected = data.get("rejected_hotels", []) or []

        kwargs["preferred_hotels"] = [str(h).strip() for h in preferred if str(h).strip()]
        kwargs["rejected_hotels"] = [str(h).strip() for h in rejected if str(h).strip()]

        kwargs["sessions_count"] = 1

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserProfile":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def merge_from(self, other: "UserProfile") -> None:
        """
        Merge another profile into this one.

        - For scalar fields: if 'other' has a non-None value, it overwrites this profile.
        - For lists: union.
        - sessions_count: incremented.
        """
        for field_name, field_def in self.__dataclass_fields__.items():  # type: ignore
            if field_name in ("sessions_count",):
                continue
            value_other = getattr(other, field_name)
            if isinstance(value_other, list):
                current_list = getattr(self, field_name) or []
                combined = current_list + [x for x in value_other if x not in current_list]
                setattr(self, field_name, combined)
            else:
                if value_other is not None:
                    setattr(self, field_name, value_other)

        self.sessions_count += max(1, other.sessions_count)

    def to_prompt_summary(self) -> str:
        """
        Short text summary for including in the assistant system prompt.
        Keeps it compact; 2-5 bullet points max.
        """
        bullets: List[str] = []

        if self.trip_type:
            bullets.append(f"Trip type: {self.trip_type}.")

        if self.budget_min is not None or self.budget_max is not None:
            if self.budget_min is not None and self.budget_max is not None:
                bullets.append(
                    f"Budget: between {int(self.budget_min)} and {int(self.budget_max)} {self.currency or 'EUR'} per night."
                )
            elif self.budget_max is not None:
                bullets.append(
                    f"Budget: up to {int(self.budget_max)} {self.currency or 'EUR'} per night."
                )

        loc_parts = []
        if self.wants_central_location:
            loc_parts.append("central location")
        if self.wants_local_neighborhood:
            loc_parts.append("local, non-touristy neighborhood")
        if loc_parts:
            bullets.append("Prefers: " + ", ".join(loc_parts) + ".")

        atmos_parts = []
        if self.prefers_quiet:
            atmos_parts.append("quiet/relaxing atmosphere")
        if self.prefers_social:
            atmos_parts.append("social/energetic vibe")
        if atmos_parts:
            bullets.append("Atmosphere: " + ", ".join(atmos_parts) + ".")

        amen_parts = []
        if self.cares_about_wifi:
            amen_parts.append("good Wi-Fi")
        if self.cares_about_desk:
            amen_parts.append("desk/workspace")
        if self.cares_about_breakfast:
            amen_parts.append("breakfast")
        if self.cares_about_parking:
            amen_parts.append("parking")
        if self.cares_about_gym:
            amen_parts.append("gym/fitness")
        if self.cares_about_rooftop:
            amen_parts.append("rooftop/terrace")
        if self.cares_about_spa:
            amen_parts.append("spa/wellness")
        if amen_parts:
            bullets.append("Cares about: " + ", ".join(amen_parts) + ".")

        theme_parts = []
        if self.foodie:
            theme_parts.append("foodie (cares about restaurants/cafés)")
        if self.romantic:
            theme_parts.append("romantic trip")
        if theme_parts:
            bullets.append("Themes: " + ", ".join(theme_parts) + ".")

        if not bullets and self.free_form_notes:
            bullets.append(self.free_form_notes)

        if not bullets:
            return "No stable preferences inferred yet."

        return "\n".join(f"- {b}" for b in bullets)



class ProfileStore:
    """
    Very simple JSON-backed profile store.
    """

    def __init__(self, path: str = "profiles.json") -> None:
        self.path = path
        self._profiles: Dict[str, UserProfile] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for user_id, prof_dict in raw.items():
                    self._profiles[user_id] = UserProfile.from_dict(prof_dict)
            except Exception:
                self._profiles = {}
        self._loaded = True

    def _save(self) -> None:
        data = {uid: prof.to_dict() for uid, prof in self._profiles.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, user_id: str) -> Optional[UserProfile]:
        self._load()
        return self._profiles.get(user_id)

    def upsert(self, user_id: str, new_profile: UserProfile) -> UserProfile:
        self._load()
        if user_id in self._profiles:
            existing = self._profiles[user_id]
            existing.merge_from(new_profile)
            self._profiles[user_id] = existing
        else:
            self._profiles[user_id] = new_profile
        self._save()
        return self._profiles[user_id]


PROFILE_SYSTEM_PROMPT = """
You are a dedicated profiling model for hotel travelers.

Your job:
- Analyze ONE finished conversation between a traveler (user) and a hotel recommendation assistant.
- Infer the traveler's stable preferences and constraints.
- Output a SINGLE JSON object with strict keys and types as specified below.
- If something is unknown, use null (NOT false or an empty string) for booleans/scalars.

You MUST:
- Base your inferences ONLY on what the traveler actually said and decided in THIS conversation.
- Prefer stable, general preferences over one-off questions.
- NOT invent preferences that are not hinted at.
- Use null when you are unsure.

JSON SCHEMA (all keys are REQUIRED, even if the value is null):

{
  "trip_type": "business | romantic | leisure | unknown",
  "persona_name": string or null,

  "budget_min": number or null,
  "budget_max": number or null,
  "currency": string or null,

  "wants_central_location": true | false | null,
  "wants_local_neighborhood": true | false | null,

  "prefers_quiet": true | false | null,
  "prefers_social": true | false | null,

  "cares_about_wifi": true | false | null,
  "cares_about_desk": true | false | null,
  "cares_about_breakfast": true | false | null,
  "cares_about_parking": true | false | null,
  "cares_about_gym": true | false | null,
  "cares_about_rooftop": true | false | null,
  "cares_about_spa": true | false | null,

  "foodie": true | false | null,
  "romantic": true | false | null,

  "preferred_hotels": array of strings (names) or [],
  "rejected_hotels": array of strings (names) or [],

  "free_form_notes": string or null
}

Output RULES:
- Output ONLY valid JSON, no backticks, no markdown, no comments.
- Do NOT include any text before or after the JSON.
"""


def _build_session_summary_for_profiler(
    history: List[Dict[str, str]],
    persona_name: str,
    persona_description: str,
) -> str:
    lines: List[str] = []

    lines.append(f"Persona name: {persona_name}")
    lines.append("Persona description:")
    lines.append(persona_description.strip())
    lines.append("")
    lines.append("Conversation transcript:")

    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if role == "user":
            lines.append(f"Traveler: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return text


def infer_profile_from_session(
    history: List[Dict[str, str]],
    persona_name: str,
    persona_description: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    temperature: float = 0.1,
) -> UserProfile:
    """
    Use an LLM (the fourth model) to infer a UserProfile from a single finished session.
    """
    user_msg = _build_session_summary_for_profiler(
        history=history,
        persona_name=persona_name,
        persona_description=persona_description,
    )

    messages = [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = call_llm(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=512,
    )

    json_str = _extract_json_block(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        fallback = {
            "trip_type": "unknown",
            "persona_name": persona_name or None,
            "budget_min": None,
            "budget_max": None,
            "currency": "EUR",

            "wants_central_location": None,
            "wants_local_neighborhood": None,

            "prefers_quiet": None,
            "prefers_social": None,

            "cares_about_wifi": None,
            "cares_about_desk": None,
            "cares_about_breakfast": None,
            "cares_about_parking": None,
            "cares_about_gym": None,
            "cares_about_rooftop": None,
            "cares_about_spa": None,

            "foodie": None,
            "romantic": None,

            "preferred_hotels": [],
            "rejected_hotels": [],

            "free_form_notes": f"Profiler JSON parse failed. Raw output (truncated): {raw[:200]}..."
        }
        return UserProfile.from_llm_dict(fallback)

    if "persona_name" not in data or data.get("persona_name") is None:
        data["persona_name"] = persona_name

    return UserProfile.from_llm_dict(data)
