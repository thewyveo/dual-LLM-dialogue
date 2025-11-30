from typing import List, Dict
from llm_client import call_llm

# Simple lexical patterns for “I’m done / goodbye”
_SATISFACTION_PATTERNS = [
    "that's all i needed",
    "that's all i need",
    "that's all, thanks",
    "that was very helpful",
    "this was very helpful",
    "this is very helpful",
    "thank you, that was very helpful",
    "thank you very much",
    "i'm done",
    "im done",
    "i'm good now",
    "im good now",
    "this is perfect, thanks",
    "great, i'll book that",
    "great, i’ll book that",
    "i'll book that one",
    "ill book that one",
    "i'll go with",
    "ill go with",
    "goodbye",
    "bye",
]

def llm_check_satisfaction(
    history: List[Dict[str, str]],
    model: str = "Qwen/Qwen2.5-3B-Instruct",
) -> bool:
    """
    Use the LLM as a 'satisfaction judge'.

    Input: full dialogue history as a list of {'role': 'user'|'assistant', 'content': str}.
    Output: True if the user has clearly expressed satisfaction / closure, False otherwise.
    """

    if not history:
        return False

    # --- 1) Deterministic lexical check on the last USER message ---
    last_msg = history[-1]
    if last_msg["role"] == "user":
        text = last_msg["content"].lower()
        for pat in _SATISFACTION_PATTERNS:
            if pat in text:
                return True

    # --- 2) LLM judge (original logic, unchanged in spirit) ---

    # Build a simple transcript
    transcript_lines = []
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        transcript_lines.append(f"{speaker}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    system_prompt = (
        "You are a judge evaluating whether a hotel booking conversation is finished.\n"
        "You see the full conversation between a User (traveler) and an Assistant.\n\n"
        "Your task: decide if the USER has clearly expressed that they are satisfied\n"
        "with the recommendation and closed the conversation.\n\n"
        "Examples of when the conversation IS finished:\n"
        "- User: \"Great, I’ll book that one, thanks, that’s all I needed.\"\n"
        "- User: \"Perfect, I’ll go with the Canal View Inn. Thanks, goodbye.\"\n"
        "- User: \"That works for me, I’ll handle the rest, bye.\"\n"
        "- User: \"Thank you, that was very helpful. That's all I needed, goodbye.\"\n"
        "Examples of when the conversation is NOT finished:\n"
        "- User: \"Could you also recommend something near the station?\"\n"
        "- User: \"I’m not sure yet, can you compare two more hotels?\"\n"
        "- User: \"Hmm, that sounds okay, but do you have anything cheaper?\"\n"
        "- User: \"Absolutely! I'm thinking of a place in the heart of the city that has a serene garden setting.\"\n"
        "- User: \"Absolutely! I'd love to hear more details about the Quiet Garden Hotel.\"\n\n"
        "IMPORTANT:\n"
        "- Only the USER's intent matters.\n"
        "- If they are still asking questions or asking for more options, the answer is NO.\n"
        "- If they clearly say they’re done / will book / say goodbye, the answer is YES.\n"
        "- Respond with exactly one word: YES or NO.\n\n"
    )

    user_msg = (
        "Here is the conversation:\n\n"
        f"{transcript}\n\n"
        "Question: Has the USER clearly expressed they want to close the conversation?\n"
        "Answer YES or NO."
    )

    resp = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        model=model,
        temperature=0.0,
        max_tokens=4,
    )

    answer = resp.strip().upper()
    # tolerate things like "YES." or "YES\n"
    return answer.startswith("Y")
