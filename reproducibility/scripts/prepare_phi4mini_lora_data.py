
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.tinychange import FinetuningDataModule, TinyChangeConfig
from track_llm_apis.util import patch_chat_template

root = Path(sys.argv[1])
snapshot = json.loads((root / "audit/phi4mini_snapshot.json").read_text())
assert snapshot["model"] == "microsoft/Phi-4-mini-instruct"

cfg = TinyChangeConfig()
pool = cfg.finetuning_dataset
assert cfg.datasets_dir.resolve() == (root / "data/datasets").resolve()
assert len(pool) == 4096
assert (cfg.seed, cfg.finetuning_batch_size, cfg.finetuning_epochs) == (0, 1, 1)

tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
tokenizer.name_or_path = snapshot["model"]
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
patch_chat_template(tokenizer, config.chat_templates)

dm = FinetuningDataModule(
    config=cfg, n_samples=128, batch_size=cfg.finetuning_batch_size, tokenizer=tokenizer,
)
indices = np.random.RandomState(cfg.seed).permutation(len(pool))[:128].tolist()
np.random.seed(cfg.seed)
dm.prepare_data()
dm.setup("fit")

loader = dm.train_dataloader()
assert len(loader) == 128
target_counts = []
padded_lengths = set()

for position, (batch, pool_index) in enumerate(zip(loader, indices, strict=True)):
    raw = pool[pool_index]["conversation"]
    expected = tokenizer.apply_chat_template(
        raw, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
        padding=False, truncation=False,
    )
    ids = batch["input_ids"][0].tolist()
    attention = batch["attention_mask"][0].tolist()
    assistant = batch["assistant_tokens_mask"][0].tolist()
    assert len(ids) == len(attention) == len(assistant), position
    assert batch["labels"][0].tolist() == ids, position
    assert [v for v, keep in zip(ids, attention) if keep] == expected["input_ids"], position
    assert [v for v, keep in zip(assistant, attention) if keep] == expected["assistant_masks"], position
    assert all(a == 0 for a, keep in zip(assistant, attention) if not keep), position
    padded_lengths.add(len(ids))
    target_counts.append(sum(assistant[1:]))

assert len(padded_lengths) == 1, f"Observed padded lengths: {sorted(padded_lengths)}"
padded_length = padded_lengths.pop()
empty = [i for i, count in enumerate(target_counts) if count == 0]
assert not empty, f"Training positions with zero assistant targets: {empty}"

report = {
    "model": snapshot["model"],
    "revision": snapshot["revision"],
    "training_pool_size": len(pool),
    "training_pool_fingerprint": pool._fingerprint,
    "seed": cfg.seed,
    "selected_examples": len(indices),
    "selected_pool_indices_in_training_order": indices,
    "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
    "configured_max_length": cfg.finetuning_max_length,
    "padded_tokens_per_example": padded_length,
    "minimum_trainable_assistant_tokens": min(target_counts),
    "data_module_cache": str(dm._get_disk_path()),
}
path = root / "audit/step34_phi4mini_lora_data.json"
if path.exists():
    assert json.loads(path.read_text()) == report, f"Existing audit differs: {path}"
else:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)

print("Verified examples in training order:", len(indices))
print("Configured max length:", cfg.finetuning_max_length)
print("Actual padded tokens per example:", padded_length)
print("Minimum trainable assistant tokens:", min(target_counts))
print("Saved:", path)
print("Step 34 passed: Phi LoRA training data, order, and masks verified.")
