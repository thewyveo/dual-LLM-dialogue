import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "assistant_ft_train.jsonl")

# You can change these to whatever Qwen chat model you are using now
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "assistant_ft_qwen")


@dataclass
class AssistantExample:
    input: str
    output: str


class AssistantDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.examples: List[AssistantExample] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r") as f:
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

        # Simple SFT: prompt + answer
        # "ASSISTANT:" is already in input, we just append the answer and EOS.
        full_text = ex.input + " " + ex.output + self.tokenizer.eos_token

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Language modeling: labels = input_ids
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    dataset = AssistantDataset(DATA_PATH, tokenizer)

    # You can tweak these training hyperparams
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save final checkpoint
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Fine-tuned model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
