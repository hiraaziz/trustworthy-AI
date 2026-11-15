
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
    snapshot = json.loads((root / "audit/phi4mini_snapshot.json").read_text())
    data = json.loads((root / "audit/step34_phi4mini_lora_data.json").read_text())
    assert data["revision"] == snapshot["revision"]
    assert data["selected_examples"] == 128
    assert Path(data["data_module_cache"]).is_dir()

    dest = root / "results/phi4mini/checkpoints/lora_128"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        saved = json.loads((dest / "COMPLETE.json").read_text())
        assert saved["revision"] == snapshot["revision"]
        assert saved["training_order_sha256"] == data["indices_sha256"]
        assert saved["optimizer_steps"] == 128
        assert get_model_hash(load_bf16(dest)) == saved["reloaded_hash"]
        print("Step 35 passed: existing Phi LoRA checkpoint verified.", flush=True)
        print("Checkpoint:", dest)
        return

    cfg = tc.TinyChangeConfig(variants_device="cuda:0")
    assert cfg.finetuning_dataset._fingerprint == data["training_pool_fingerprint"]
    assert cfg.seed == data["seed"] == 0
    assert cfg.finetuning_lr_scale == [1e-6]
    assert cfg.finetuning_batch_size == cfg.finetuning_epochs == 1

    base = load_bf16(snapshot["snapshot"])
    assert base.device.type == "cpu" and base.dtype == torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
    tokenizer.name_or_path = snapshot["model"]
    patch_chat_template(tokenizer, config.chat_templates)
    change = tc.TinyChange(base, tokenizer, cfg)

    record = {
        "model": snapshot["model"],
        "revision": snapshot["revision"],
        "training_order_sha256": data["indices_sha256"],
        "base_device": str(base.device),
        "base_dtype": str(base.dtype),
        "base_hash": change.model_hash,
        "training_device": cfg.variants_device,
        "seed": cfg.seed,
        "rank": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "target_modules": cfg.lora_target_modules,
    }

    # Observe the original trainer immediately before and after its fit call.
    original_fit = tc.pl.Trainer.fit

    def observed_fit(trainer, model, datamodule):
        assert Path(datamodule._get_disk_path()) == Path(data["data_module_cache"])
        assert not model.model.is_gradient_checkpointing
        trainable = {
            name: str(p.dtype)
            for name, p in model.model.named_parameters() if p.requires_grad
        }
        assert trainable and all("lora_" in name for name in trainable)
        audit = json.loads(
            (root / "audit/step33_phi4mini_lora_targets.json").read_text()
        )
        actual_shapes = {
            name: list(p.shape)
            for name, p in model.model.named_parameters() if p.requires_grad
        }
        assert audit["revision"] == snapshot["revision"]
        assert actual_shapes == audit["trainable_parameter_shapes"], "LoRA targets differ from Step 33"
        assert audit["matched_by_projection"] == {"o_proj": 32}
        count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        assert count == audit["trainable_parameter_count"] == 1572864
        record["trainable_parameter_count"] = count
        record["matched_by_projection"] = audit["matched_by_projection"]
        print("Verified before training: 32 o_proj adapters, 1,572,864 trainable parameters.", flush=True)
        record["trainable_parameters"] = trainable
        record["gradient_checkpointing"] = False
        print("Trainable adapter tensors:", len(trainable), flush=True)
        result = original_fit(trainer, model=model, datamodule=datamodule)
        record["optimizer_steps"] = int(trainer.global_step)
        assert record["optimizer_steps"] == 128
        return result

    torch.cuda.reset_peak_memory_stats()
    with patch.object(tc.pl.Trainer, "fit", observed_fit):
        variant = change.finetune(
            lr=cfg.finetuning_lr_scale[0], n_samples=128,
            epochs=cfg.finetuning_epochs, batch_size=cfg.finetuning_batch_size,
            use_lora=True,
        )

    record.update({
        "description": variant.description,
        "merged_dtype": str(variant.model.dtype),
        "merged_hash": variant.model_hash,
        "merged_weights_changed": variant.model_hash != change.model_hash,
        "peak_torch_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })
    assert variant.model.dtype == torch.bfloat16

    # Publish the final directory only after reloading and verifying its weights.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".phi4mini-lora-") as tmp:
        staged = Path(tmp) / "checkpoint"
        variant.model.save_pretrained(staged, safe_serialization=True)
        tokenizer.save_pretrained(staged)
        reloaded = load_bf16(staged)
        original_state = variant.model.state_dict()
        loaded_state = reloaded.state_dict()
        assert original_state.keys() == loaded_state.keys(), "Checkpoint tensor names differ"

        for name, tensor in original_state.items():
            actual = loaded_state[name]
            assert tensor.dtype == actual.dtype, f"Reload dtype mismatch: {name}"
            assert tensor.shape == actual.shape, f"Reload shape mismatch: {name}"
            assert torch.equal(tensor.detach().cpu(), actual), f"Reload value mismatch: {name}"

        before_buffers = dict(variant.model.named_buffers())
        after_buffers = dict(reloaded.named_buffers())
        buffer_differences = []
        for name in sorted((before_buffers.keys() | after_buffers.keys()) - original_state.keys()):
            before = before_buffers.get(name)
            after = after_buffers.get(name)
            if (before is None or after is None or before.dtype != after.dtype
                    or before.shape != after.shape or not torch.equal(before.cpu(), after)):
                buffer_differences.append(name)

        base_params = dict(base.named_parameters())
        loaded_params = dict(reloaded.named_parameters())
        assert base_params.keys() == loaded_params.keys(), "Parameter layout differs from baseline"
        changed = 0
        for name, original in base_params.items():
            actual = loaded_params[name]
            assert original.dtype == actual.dtype == torch.bfloat16, name
            assert original.shape == actual.shape, name
            changed += int(torch.count_nonzero(original.detach() != actual.detach()).item())

        record["reloaded_hash"] = get_model_hash(reloaded)
        record["verified_checkpoint_tensors"] = len(loaded_state)
        record["nonpersistent_buffer_differences"] = buffer_differences
        record["changed_bf16_parameter_values"] = changed
        record["merged_weights_changed"] = changed > 0
        print("Verified checkpoint tensors:", len(loaded_state), flush=True)
        print("Non-persistent buffer differences:", buffer_differences, flush=True)
        print("Changed BF16 parameter values:", changed, flush=True)
        (staged / "COMPLETE.json").write_text(json.dumps(record, indent=2) + "\n")
        staged.rename(dest)

    for key in ("optimizer_steps", "merged_dtype", "merged_weights_changed",
                "peak_torch_allocated_gib"):
        print(f"{key}: {record[key]}")
    print("Checkpoint:", dest)
    print("Step 35 passed: Phi LoRA trained, merged, saved, and reload-verified.")


if __name__ == "__main__":
    main()
