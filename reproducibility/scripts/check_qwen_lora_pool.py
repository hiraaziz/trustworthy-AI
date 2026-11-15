
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.tinychange import TinyChangeConfig
from track_llm_apis.util import patch_chat_template


def main():
    root = Path(sys.argv[1])
    snapshot = json.loads((root / "audit/qwen_snapshot.json").read_text())
    cfg = TinyChangeConfig()
    pool = cfg.finetuning_dataset

    assert cfg.datasets_dir == root / "data/datasets"
    assert max(cfg.finetuning_samples) == 4096 and 128 in cfg.finetuning_samples
    assert len(pool) == 4096
    assert cfg.finetuning_batch_size == 1 and cfg.finetuning_epochs == 1
    assert cfg.finetuning_lr_scale == [1e-6]
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) == (8, 8, 0.0)
    assert cfg.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
    patch_chat_template(tokenizer, config.chat_templates)

    lengths = []
    for start in range(0, len(pool), 32):
        conversations = pool[start:start + 32]["conversation"]
        encoded = tokenizer.apply_chat_template(
            conversations, tokenize=True, return_dict=True,
            padding=False, truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    longest = max(lengths)
    report = {
        "model": snapshot["model"],
        "revision": snapshot["revision"],
        "training_pool_size": len(pool),
        "training_pool_fingerprint": pool._fingerprint,
        "updates_for_selected_variant": 128,
        "configured_max_length": cfg.finetuning_max_length,
        "longest_untruncated_sequence": longest,
        "conversations_longer_than_configured_max": sum(
            n > cfg.finetuning_max_length for n in lengths
        ),
        "padded_token_positions_per_column": len(pool) * longest,
        "batch_size": cfg.finetuning_batch_size,
        "epochs": cfg.finetuning_epochs,
        "learning_rate": cfg.finetuning_lr_scale[0],
        "rank": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "target_modules": cfg.lora_target_modules,
    }
    path = root / "audit/step08_qwen_lora_pool.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)
    print(json.dumps(report, indent=2))
    print("Step 8 passed: upstream LoRA pool and sequence lengths verified.")


if __name__ == "__main__":
    main()
