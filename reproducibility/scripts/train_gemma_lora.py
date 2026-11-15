
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from track_llm_apis import tinychange as tc
from track_llm_apis.config import config
from track_llm_apis.util import get_model_hash, patch_chat_template


def load_bf16(path):
    return AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, local_files_only=True,
    )


def main():
    root = Path(sys.argv[1])
    snap = json.loads((root / "audit/gemma_snapshot.json").read_text())
    data = json.loads((root / "audit/step21_gemma_lora_data.json").read_text())
    assert snap["model"] == data["model"] == "google/gemma-3-1b-it"
    assert snap["revision"] == data["revision"]
    assert data["selected_examples"] == 128 and data["minimum_trainable_assistant_tokens"] > 0
    cache = Path(data["data_module_cache"])
    assert cache.is_dir(), f"Missing Step 21 training cache: {cache}"

    dest = root / "results/gemma/checkpoints/lora_128"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        saved = json.loads((dest / "COMPLETE.json").read_text())
        assert saved["revision"] == snap["revision"]
        assert saved["training_order_sha256"] == data["indices_sha256"]
        assert saved["optimizer_steps"] == 128
        assert get_model_hash(load_bf16(dest)) == saved["reloaded_hash"]
        print("Step 22 passed: existing Gemma LoRA checkpoint reverified.")
        print("Changed BF16 parameter values:", saved["changed_bf16_parameter_values"])
        print("Checkpoint:", dest)
        return

    assert torch.cuda.is_available(), "Training process cannot see the GPU."
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    cfg = tc.TinyChangeConfig(variants_device="cuda:0")
    assert cfg.finetuning_dataset._fingerprint == data["training_pool_fingerprint"]
    assert cfg.seed == data["seed"] == 0
    assert cfg.finetuning_lr_scale == [1e-6]
    assert cfg.finetuning_batch_size == cfg.finetuning_epochs == 1
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) == (8, 8, 0.0)
    assert cfg.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    base = load_bf16(snap["snapshot"])
    assert base.device.type == "cpu" and base.dtype == torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(snap["snapshot"], local_files_only=True)
    tokenizer.name_or_path = snap["model"]
    patch_chat_template(tokenizer, config.chat_templates)
    change = tc.TinyChange(base, tokenizer, cfg)

    report = {
        "model": snap["model"],
        "revision": snap["revision"],
        "base_hash": change.model_hash,
        "base_dtype": str(base.dtype),
        "base_device": str(base.device),
        "training_device": cfg.variants_device,
        "gpu": torch.cuda.get_device_name(0),
        "training_order_sha256": data["indices_sha256"],
        "padded_tokens_per_example": data["padded_tokens_per_example"],
        "seed": cfg.seed,
        "rank": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "target_modules": cfg.lora_target_modules,
    }

    # Observe the authors' trainer without changing its batches or optimizer.
    original_fit = tc.pl.Trainer.fit

    def observed_fit(trainer, model, datamodule):
        assert Path(datamodule._get_disk_path()) == cache
        assert not model.model.is_gradient_checkpointing
        trainable = {
            name: str(p.dtype)
            for name, p in model.model.named_parameters() if p.requires_grad
        }
        assert trainable and all("lora_" in name for name in trainable)
        report["trainable_parameters"] = trainable
        report["gradient_checkpointing"] = False
        print("GPU:", report["gpu"], flush=True)
        print("Trainable adapter tensors:", len(trainable), flush=True)
        print("Adapter dtypes:", sorted(set(trainable.values())), flush=True)

        outcome = original_fit(trainer, model=model, datamodule=datamodule)
        assert len(trainer.optimizers) == 1
        optimizer = trainer.optimizers[0]
        assert isinstance(optimizer, torch.optim.SGD)
        assert all(group["lr"] == 1e-6 for group in optimizer.param_groups)
        report["optimizer"] = type(optimizer).__name__
        report["optimizer_steps"] = int(trainer.global_step)
        assert report["optimizer_steps"] == 128
        return outcome

    torch.cuda.reset_peak_memory_stats()
    with patch.object(tc.pl.Trainer, "fit", observed_fit):
        variant = change.finetune(
            lr=cfg.finetuning_lr_scale[0], n_samples=128,
            epochs=cfg.finetuning_epochs, batch_size=cfg.finetuning_batch_size,
            use_lora=True,
        )

    report["peak_torch_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["description"] = variant.description
    report["merged_dtype"] = str(variant.model.dtype)
    report["merged_hash"] = variant.model_hash
    assert variant.model.dtype == torch.bfloat16

    # Publish the checkpoint only after verifying the saved tensors on reload.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".gemma-lora-") as tmp:
        staged = Path(tmp) / "checkpoint"
        variant.model.save_pretrained(staged, safe_serialization=True)
        tokenizer.save_pretrained(staged)
        reloaded = load_bf16(staged)

        expected = variant.model.state_dict()
        actual = reloaded.state_dict()
        assert expected.keys() == actual.keys(), "Checkpoint tensor names differ"
        for name, tensor in expected.items():
            restored = actual[name]
            assert tensor.shape == restored.shape and tensor.dtype == restored.dtype, name
            assert torch.equal(tensor.detach().cpu(), restored), f"Reload mismatch: {name}"

        base_params = dict(base.named_parameters())
        saved_params = dict(reloaded.named_parameters())
        assert base_params.keys() == saved_params.keys()
        changed = 0
        for name, original in base_params.items():
            restored = saved_params[name]
            assert original.dtype == restored.dtype == torch.bfloat16, name
            assert original.shape == restored.shape, name
            changed += int(torch.count_nonzero(original.detach() != restored.detach()).item())

        report["verified_checkpoint_tensors"] = len(actual)
        report["changed_bf16_parameter_values"] = changed
        report["merged_weights_changed"] = changed > 0
        report["reloaded_hash"] = get_model_hash(reloaded)
        (staged / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(dest)

    for key in ("optimizer_steps", "merged_dtype", "verified_checkpoint_tensors",
                "changed_bf16_parameter_values", "peak_torch_allocated_gib"):
        print(f"{key}: {report[key]}")
    print("Checkpoint:", dest)
    print("Step 22 passed: Gemma LoRA trained, merged, saved, and reload-verified.")


if __name__ == "__main__":
    main()
