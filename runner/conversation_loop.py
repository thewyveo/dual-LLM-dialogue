from typing import Dict, Any
import uuid

from memory.memory import Memory
from agents.user_agent import UserAgent, PERSONA_MINIMALIST, PERSONA_EXPLORER
from agents.assistant_prompt import AssistantPromptAgent
from agents.assistant_ft import AssistantFineTunedAgent
from retrieval.hotel_api import HotelAPIClient


def run_conversation(
    persona: str = "minimalist",
    assistant_variant: str = "prompt",  # or "ft"
    max_turns: int = 5,
    location: str = "Amsterdam",
) -> Dict[str, Any]:
    """
    Run a single user-assistant conversation.
    Returns logs + some basic stats.
    """

    # Choose persona description
    if persona == "minimalist":
        persona_description = PERSONA_MINIMALIST
    else:
        persona_description = PERSONA_EXPLORER

    # Init components
    session_id = str(uuid.uuid4())
    memory = Memory()
    memory.init_session(session_id)

    user = UserAgent(persona_name=persona, persona_description=persona_description)

    if assistant_variant == "prompt":
        assistant = AssistantPromptAgent()
    else:
        assistant = AssistantFineTunedAgent()

    hotel_client = HotelAPIClient()

    # Initial user turn
    user_utt = user.initial_prompt(location=location)
    memory.add_turn(session_id, "user", user_utt)

    turn_count = 0
    finished = False

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

        # Stop condition
        if user.check_satisfaction(user_utt):
            finished = True
            print("  User satisfied, ending conversation.")

    final_history = memory.get_history(session_id)

    return {
        "session_id": session_id,
        "persona": persona,
        "assistant_variant": assistant_variant,
        "history": final_history,
        "finished": finished,
        "num_turns": turn_count,
    }
