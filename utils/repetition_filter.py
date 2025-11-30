from typing import List
import torch
from bert_score import score as bertscore_score

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# You can change this to another BERTScore model if you like.
_BERT_MODEL_TYPE = "microsoft/deberta-base-mnli"


def max_bertscore_similarity(
    candidate: str,
    previous_texts: List[str],
    model_type: str = None,
) -> float:
    """
    Compute max BERTScore F1 between `candidate` and a list of previous texts.
    Returns 0.0 if there are no previous texts.
    """
    if not previous_texts:
        return 0.0

    model_type = model_type or _BERT_MODEL_TYPE

    # candidate vs each previous text
    cands = [candidate] * len(previous_texts)
    refs = previous_texts

    with torch.no_grad():
        _, _, F1 = bertscore_score(
            cands,
            refs,
            model_type=model_type,
            verbose=False,
            device=_DEVICE,
        )

    max_sim = float(F1.max().item())
    return max_sim


def is_semantic_repeat(
    candidate: str,
    previous_texts: List[str],
    threshold: float = 0.92,
) -> bool:
    """
    Return True if `candidate` is too similar to ANY previous text
    according to BERTScore (F1 >= threshold).
    """
    if not previous_texts:
        return False

    sim = max_bertscore_similarity(candidate, previous_texts)
    return sim >= threshold
