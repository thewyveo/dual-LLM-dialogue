from typing import List, Dict

try:
    import tiktoken
except ImportError:
    tiktoken = None


def count_tokens(messages: List[Dict[str, str]], model_name: str = "gpt-4.1-mini") -> int:
    """
    Rough token counter. If tiktoken is available, use it; otherwise, fall back to a naive split.
    """
    if tiktoken is None:
        # naive: just split on whitespace
        return sum(len(m["content"].split()) for m in messages)

    # pick an encoding; you can adjust to your exact model later
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg["content"]))
    return total
