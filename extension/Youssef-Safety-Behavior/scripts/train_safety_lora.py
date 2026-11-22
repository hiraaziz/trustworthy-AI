"""Train a small safety LoRA for the B3IT safety extension (Qwen2.5-0.5B).

The held-out V2 questions are explicitly excluded from the training pool.
The 1:3 follow-up uses 256 harmful/refusal and 768 benign/helpful examples,
one epoch, LoRA only, and saves both the adapter and a merged model for vLLM evaluation.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SEED = int(os.environ.get("B3IT_LORA_SEED", "20261119"))
MODEL_DIR = Path(os.environ.get("B3IT_SAFETY_MODEL", "/kaggle/working/models/Qwen2.5-0.5B-Instruct"))
OUT_DIR = Path(os.environ.get("B3IT_LORA_DIR", "/kaggle/working/safety-lora-qwen05"))
N_HARMFUL = int(os.environ.get("B3IT_LORA_N_HARMFUL", "256"))
N_BENIGN = int(os.environ.get("B3IT_LORA_N_BENIGN", "768"))
MAX_LENGTH = int(os.environ.get("B3IT_LORA_MAX_LENGTH", "256"))
SYSTEM = "You are a helpful assistant."


def heldout_questions() -> set[str]:
    """Load the V2 evaluation prompts so none can leak into training."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_safety_prompt_1a_v2 import BENIGN_QUESTIONS, HARMFUL_QUESTIONS

    return {normalize(q) for q in HARMFUL_QUESTIONS + BENIGN_QUESTIONS}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def str_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def choose_pairs(heldout: set[str]) -> list[dict[str, str]]:
    # The English config is the original 1,000 harmful-prompt/refusal pairs.
    harmful_ds = load_dataset("iNLP-Lab/multilingual-safety", "en", split="train")
    benign_ds = load_dataset("AmazonScience/FalseReject", split="train")
    rng = random.Random(SEED)

    harmful: list[dict[str, str]] = []
    for row in harmful_ds:
        prompt, response = str_value(row.get("prompt")), str_value(row.get("output"))
        if prompt and response and normalize(prompt) not in heldout:
            harmful.append({"prompt": prompt, "response": response, "source": "multilingual-safety/en"})

    benign: list[dict[str, str]] = []
    for row in benign_ds:
        prompt, response = str_value(row.get("prompt")), str_value(row.get("instruct_response"))
        if prompt and response and normalize(prompt) not in heldout:
            benign.append({"prompt": prompt, "response": response, "source": "FalseReject/train"})

    if len(harmful) < N_HARMFUL or len(benign) < N_BENIGN:
        raise RuntimeError(
            f"Not enough unique training rows: harmful={len(harmful)}, benign={len(benign)}; "
            f"requested harmful={N_HARMFUL}, benign={N_BENIGN}."
        )
    return rng.sample(harmful, N_HARMFUL) + rng.sample(benign, N_BENIGN)


class SafetyDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: Any):
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            prefix = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            whole = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
            ids = tokenizer(whole, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
            if len(ids) <= len(prefix_ids):
                continue
            labels = [-100] * min(len(prefix_ids), len(ids)) + ids[len(prefix_ids):]
            labels = labels[:MAX_LENGTH]
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
        if not self.items:
            raise RuntimeError("No usable tokenized training rows; check model tokenizer and dataset fields.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


class PadCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            pad = width - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(x["attention_mask"] + [0] * pad)
            batch["labels"].append(x["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before LoRA training. Training runs on one T4.")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Base model not found: {MODEL_DIR}")
    if (OUT_DIR / "merged_model").is_dir() and (OUT_DIR / "adapter").is_dir():
        print(f"Found existing adapter and merged model under {OUT_DIR}; reusing them.")
        return

    random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading model/tokenizer: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = choose_pairs(heldout_questions())
    random.Random(SEED).shuffle(rows)
    train_data = SafetyDataset(rows, tokenizer)
    print(f"Training rows: {len(train_data)} (balanced; held-out prompts excluded)")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        dataloader_num_workers=0,
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, data_collator=PadCollator(tokenizer))
    trainer.train()

    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    print("Merging LoRA adapter into a separate copy of the base model for vLLM.")
    merged = model.merge_and_unload()
    merged_dir = OUT_DIR / "merged_model"
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merged_dir)
    del trainer, model, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "train_rows": len(train_data),
        "harmful_refusal_rows": N_HARMFUL,
        "benign_helpful_rows": N_BENIGN,
        "epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "heldout_eval_questions_excluded": True,
        "datasets": ["iNLP-Lab/multilingual-safety (en)", "AmazonScience/FalseReject (train)"],
    }
    (OUT_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Adapter: {adapter_dir}")
    print(f"Merged model: {merged_dir}")


if __name__ == "__main__":
    main()
