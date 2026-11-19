
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
    snap = json.loads((root / "audit/gemma_snapshot.json").read_text())
    model = "google/gemma-3-1b-it"
    assert snap["model"] == model
    assert Path(snap["snapshot"]).name == snap["revision"]

    cfg = TinyChangeConfig()
    pool = cfg.finetuning_dataset
    assert cfg.datasets_dir == root / "data/datasets"
    assert len(pool) == max(cfg.finetuning_samples) == 4096
    assert 128 in cfg.finetuning_samples and cfg.seed == 0
    assert cfg.finetuning_batch_size == cfg.finetuning_epochs == 1

    tok = AutoTokenizer.from_pretrained(snap["snapshot"], local_files_only=True)
    tok.name_or_path = model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    patch_chat_template(tok, config.chat_templates)

    # Independent check of the indices expected from upstream's seeded permutation.
    indices = np.random.RandomState(cfg.seed).permutation(len(pool))[:128].tolist()
    dm = FinetuningDataModule(
        config=cfg, n_samples=128, batch_size=cfg.finetuning_batch_size, tokenizer=tok,
    )
    np.random.seed(cfg.seed)
    dm.prepare_data()
    dm.setup("fit")
    selected = dm.tokenized_dataset
    assert len(selected) == 128

    padded_lengths, target_counts = set(), []
    for position, pool_index in enumerate(indices):
        expected = tok.apply_chat_template(
            pool[pool_index]["conversation"],
            tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
            padding=False, truncation=False,
        )
        row = selected[position]
        ids = row["input_ids"].tolist()
        attention = row["attention_mask"].tolist()
        assistant = row["assistant_tokens_mask"].tolist()
        assert len(ids) == len(attention) == len(assistant), position
        assert row["labels"].tolist() == ids, position
        assert [x for x, keep in zip(ids, attention) if keep] == expected["input_ids"], position
        assert [x for x, keep in zip(assistant, attention) if keep] == expected["assistant_masks"], position
        assert all(mask == 0 for mask, keep in zip(assistant, attention) if not keep), position
        padded_lengths.add(len(ids))
        target_counts.append(sum(assistant[1:]))

    assert len(padded_lengths) == 1, f"Unexpected padded lengths: {padded_lengths}"
    empty = [i for i, n in enumerate(target_counts) if n == 0]
    assert not empty, f"Selected examples with zero trainable assistant tokens: {empty}"

    record = {
        "model": model,
        "revision": snap["revision"],
        "training_pool_size": len(pool),
        "training_pool_fingerprint": pool._fingerprint,
        "seed": cfg.seed,
        "selected_examples": len(selected),
        "selected_pool_indices_in_training_order": indices,
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "padded_tokens_per_example": padded_lengths.pop(),
        "minimum_trainable_assistant_tokens": min(target_counts),
        "data_module_cache": str(dm._get_disk_path()),
    }
    path = root / "audit/step21_gemma_lora_data.json"
    if path.exists():
        assert json.loads(path.read_text()) == record, "Existing Gemma training-data report differs."
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(path)

    print("Verified examples in training order:", len(selected))
    print("Padded tokens per example:", record["padded_tokens_per_example"])
    print("Minimum trainable assistant tokens:", min(target_counts))
    print("Saved:", path)
    print("Step 21 passed: Gemma LoRA training data, order, and masks verified.")


if __name__ == "__main__":
    main()
