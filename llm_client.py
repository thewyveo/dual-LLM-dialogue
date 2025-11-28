import time
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global cache
_tokenizer = None
_model = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_local_model(model_name: str):
    """
    Lazy-load the local chat model once and reuse it.
    """
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print(f"[llm_client] Loading local model '{model_name}' on device={_DEVICE}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.to(_DEVICE)
        _model.eval()

    return _tokenizer, _model


def call_llm(
    messages: List[Dict[str, str]],
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    temperature: float = 0.3,
    max_tokens: int = 128,
) -> str:
    """
    Generic chat completion helper for a local HuggingFace chat model.

    messages: list of {"role": "system" | "user" | "assistant", "content": "..."}.
    Returns only the NEW generated text (assistant or traveler turn).
    """
    tokenizer, hf_model = _load_local_model(model)

    # 1) Build prompt using chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # tells the model "now it's your turn"
        )
    else:
        # Fallback: simple role-tagged prompt
        prompt = ""
        for m in messages:
            role = m["role"]
            if role == "system":
                prompt += f"<system>\n{m['content']}\n</system>\n"
            elif role == "user":
                prompt += f"<user>\n{m['content']}\n</user>\n"
            else:
                prompt += f"<assistant>\n{m['content']}\n</assistant>\n"
        prompt += "<assistant>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 2) Slice off the prompt part
    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()
