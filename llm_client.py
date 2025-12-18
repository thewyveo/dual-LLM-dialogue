import time
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_CACHE = {}


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve a model name to an actual path / identifier for AutoModel.

    Rules:
    - "assistant-ft-qwen" -> ./models/assistant_ft_qwen
    - if ./models/<model_name> exists -> use that as local path
    - otherwise -> assume it's a HuggingFace hub id and use as-is
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # special cases: fine-tuned assistants
    if model_name == "assistant-ft-qwen":
        return os.path.join(base_dir, "models", "assistant_ft_qwen")
    elif model_name == "assistant-peft-qwen":
        return os.path.join(base_dir, "models", "assistant_peft_qwen")

    # if there's a local directory under ./models/<model_name>, use that
    local_dir = os.path.join(base_dir, "models", model_name)
    if os.path.isdir(local_dir):
        return local_dir

    # otherwise, treat the string as a HF model id
    return model_name


from peft import PeftModel


def _get_local_model_and_tokenizer(model_name: str):
    if model_name in LOCAL_MODEL_CACHE:
        return LOCAL_MODEL_CACHE[model_name]

    model_path = _resolve_model_path(model_name)
    print(f"[llm_client] Loading model '{model_name}' from '{model_path}' on device {_DEVICE}...")

    # --- TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if model_name != "assistant-peft-qwen" else "Qwen/Qwen2.5-0.5B-Instruct",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- MODEL ---
    if model_name == "assistant-peft-qwen":
        # 1. load BASE model
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # 2. attach LoRA adapters
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
        )

    else:
        # prompt or full-FT
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    model = model.to(_DEVICE)
    model.eval()

    LOCAL_MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


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
    hf_model, tokenizer = _get_local_model_and_tokenizer(model)

    # 1) build prompt using chat template if it actually exists
    use_chat_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )

    if use_chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # tells the model "now it's your turn"
        )
    else:
        # fallback: simple role-tagged prompt
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

    # 2) slice off the prompt part
    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()
