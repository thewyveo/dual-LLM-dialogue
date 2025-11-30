from typing import Dict, Any, List, Optional
import uuid

from agents.satisfaction_judge import llm_check_satisfaction
from memory.memory import Memory, update_memory_with_session_profile
from agents.user_agent import UserAgent, PERSONA_MINIMALIST, PERSONA_EXPLORER
from agents.assistant_prompt import AssistantPromptAgent
from agents.assistant_ft import AssistantFineTunedAgent
from retrieval.hotel_api import HotelAPIClient


def run_conversation(
    persona: str = "minimalist",
    assistant_variant: str = "prompt",  # or "ft"
    max_turns: int = 5,
    location: str = "Amsterdam",
    initial_history: Optional[List[Dict[str, str]]] = None,
    seed_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single user-assistant conversation.
    Optionally uses a pre-designed initial history instead of letting
    the user LLM generate the first utterance.

    Now also accepts:
    - seed_id: the ID of the initial history seed (e.g. "min_business_trip").

    We use (persona, seed_id) combined as the logical "user_id" for profiling,
    so each (persona, seed) pair gets its own profile.
    """

    # Choose persona description
    if persona == "minimalist":
        persona_description = PERSONA_MINIMALIST
    else:
        persona_description = PERSONA_EXPLORER

    # Build a stable logical user key: persona__seed_id
    if seed_id:
        user_key = f"{persona}__{seed_id}"
    else:
        user_key = persona

    # Init components
    session_id = str(uuid.uuid4())
    memory = Memory()
    memory.init_session(session_id)

    user = UserAgent(persona_name=persona, persona_description=persona_description)

    if assistant_variant == "prompt":
        # Use persona+seed as a stable "user_id" so each seed has its own profile
        assistant = AssistantPromptAgent(user_id=user_key)
    else:
        assistant = AssistantFineTunedAgent()

    hotel_client = HotelAPIClient()

    # --- INITIAL HISTORY SETUP ---
    if initial_history is not None and len(initial_history) > 0:
        # Use the provided messages as the beginning of the dialogue.
        # We assume the last message is from the user (a user request).
        for msg in initial_history:
            role = msg["role"]
            content = msg["content"]
            memory.add_turn(session_id, role, content)
    else:
        # Old behavior: let the user LLM create the opener.
        user_utt = user.initial_prompt(location=location)
        memory.add_turn(session_id, "user", user_utt)

    turn_count = 0
    finished = False
    stop_reason = "max_turns"  # default

    # --- MAIN LOOP ---
    while turn_count < max_turns and not finished:
        turn_count += 1
        print(f"  Turn {turn_count}: assistant thinking...")

        # Retrieve dialogue history so far
        history = memory.get_history(session_id)

        # Simple retrieval: search in the given location, no extra constraints yet
        candidate_hotels = hotel_client.search_hotels(location=location, limit=5)

        # Assistant responds
        assistant_utt = assistant.respond(history, candidate_hotels)
        memory.add_turn(session_id, "assistant", assistant_utt)

        print(f"  Turn {turn_count}: user responding...")

        # User responds
        history = memory.get_history(session_id)
        user_utt = user.next_utterance(history)
        memory.add_turn(session_id, "user", user_utt)

        # NEW: ask the judge LLM if the conversation is done
        full_history = memory.get_history(session_id)
        if llm_check_satisfaction(full_history):
            finished = True
            stop_reason = "user_satisfied"
            print("  LLM judge: user satisfied, ending conversation.")

    final_history = memory.get_history(session_id)

    # After the conversation, if user is satisfied, update long-term profile
    if stop_reason == "user_satisfied":
        try:
            updated_profile = update_memory_with_session_profile(
                user_id=user_key,                # persona + seed
                history=final_history,
                persona_name=persona,
                persona_description=persona_description,
            )
            print(f"  Updated long-term profile for '{user_key}':")
            print(updated_profile.to_dict())
        except Exception as e:
            # Don't crash the run if profiling fails
            print(f"  WARNING: failed to update profile for '{user_key}': {e}")

    return {
        "session_id": session_id,
        "persona": persona,
        "assistant_variant": assistant_variant,
        "history": final_history,
        "finished": finished,
        "num_turns": turn_count,
        "stop_reason": stop_reason,
        # seed_id is still also attached at batch level, but we can return it here too
        "initial_seed_id": seed_id,
    }
