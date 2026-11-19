
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.tinychange import FinetuningDataModule, TinyChangeConfig
from track_llm_apis.util import patch_chat_template


def main():
    root = Path(sys.argv[1])
    snapshot = json.loads((root / "audit/qwen_snapshot.json").read_text())
    measured = json.loads((root / "audit/step08_qwen_lora_pool.json").read_text())

    cfg = TinyChangeConfig()
    pool = cfg.finetuning_dataset
    assert cfg.datasets_dir == root / "data/datasets"
    assert len(pool) == measured["training_pool_size"] == 4096
    assert pool._fingerprint == measured["training_pool_fingerprint"]
    assert cfg.seed == 0 and cfg.finetuning_batch_size == 1

    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
    # The authors use the Hub model ID in the data module's cache directory name.
    tokenizer.name_or_path = snapshot["model"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_chat_template(tokenizer, config.chat_templates)

    dm = FinetuningDataModule(
        config=cfg, n_samples=128,
        batch_size=cfg.finetuning_batch_size, tokenizer=tokenizer,
    )

    # TinyChange seeds NumPy before the data module creates its permutation.
    # Use a separate RNG to calculate the expected indices for verification.
    expected_indices = np.random.RandomState(cfg.seed).permutation(len(pool))[:128]
    np.random.seed(cfg.seed)
    dm.prepare_data()
    dm.setup("fit")

    selected = dm.tokenized_dataset
    assert len(selected) == 128
    padded_length = measured["longest_untruncated_sequence"]
    target_counts = []

    for position, pool_index in enumerate(expected_indices):
        raw = pool[int(pool_index)]["conversation"]
        expected = tokenizer.apply_chat_template(
            raw, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, padding=False, truncation=False,
        )
        row = selected[position]
        ids = row["input_ids"].tolist()
        attention = row["attention_mask"].tolist()
        assistant = row["assistant_tokens_mask"].tolist()
        assert len(ids) == len(attention) == len(assistant) == padded_length, position
        assert row["labels"].tolist() == ids, position
        assert [v for v, keep in zip(ids, attention) if keep] == expected["input_ids"], position
        assert [v for v, keep in zip(assistant, attention) if keep] == expected["assistant_masks"], position
        assert all(a == 0 for a, keep in zip(assistant, attention) if not keep), position
        target_counts.append(sum(assistant[1:]))

    empty = [i for i, count in enumerate(target_counts) if count == 0]
    assert not empty, f"Examples with zero trainable assistant tokens: {empty}"

    indices = expected_indices.tolist()
    report = {
        "model": snapshot["model"],
        "revision": snapshot["revision"],
        "training_pool_fingerprint": pool._fingerprint,
        "seed": cfg.seed,
        "selected_examples": len(selected),
        "selected_pool_indices_in_training_order": indices,
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "padded_tokens_per_example": padded_length,
        "minimum_trainable_assistant_tokens": min(target_counts),
        "data_module_cache": str(dm._get_disk_path()),
    }
    path = root / "audit/step09_qwen_lora_data.json"
    if path.exists():
        assert json.loads(path.read_text()) == report, "Existing training-data report differs."
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(path)

    print("Verified examples in training order:", len(selected))
    print("Padded tokens per example:", padded_length)
    print("Minimum trainable assistant tokens:", min(target_counts))
    print("Saved:", path)
    print("Step 9 passed: upstream Qwen LoRA data, order, and masks verified.")


if __name__ == "__main__":
    main()
