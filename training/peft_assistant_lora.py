import os
import json
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# =========================
# Paths & model config
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "assistant_ft_train.jsonl")

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "assistant_peft_qwen"
)


@dataclass
class AssistantExample:
    input: str
    output: str


class AssistantDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.examples: List[AssistantExample] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(
                    AssistantExample(
                        input=ex["input"],
                        output=ex["output"],
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # prompt + answer (instruction tuning)
        full_text = ex.input + " " + ex.output + self.tokenizer.eos_token

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # labels = LM objective
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # lora configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                       # rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = AssistantDataset(DATA_PATH, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4, # (higher LR for lora)
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # save only adapters + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
