
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from track_llm_apis import tinychange as tc
from track_llm_apis.util import get_model_hash, patch_chat_template
from track_llm_apis.config import config


def load_bf16(path):
    return AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, local_files_only=True
    )


def verify_saved_weights(model, reloaded):
    expected = model.state_dict()
    actual = reloaded.state_dict()
    assert expected.keys() == actual.keys(), "Checkpoint tensor names differ"
    for name, before in expected.items():
        after = actual[name]
        assert before.dtype == after.dtype, f"Reload dtype mismatch: {name}"
        assert before.shape == after.shape, f"Reload shape mismatch: {name}"
        assert torch.equal(before.detach().cpu(), after), f"Reload value mismatch: {name}"
    return len(actual)


def main():
    root = Path(sys.argv[1])
    snapshot = json.loads((root / "audit/qwen_snapshot.json").read_text())
    checkpoints = root / "results/qwen/checkpoints"
    lora = json.loads((checkpoints / "lora_128/COMPLETE.json").read_text())
    assert snapshot["revision"] == lora["revision"]

    base = load_bf16(snapshot["snapshot"])
    baseline_hash = get_model_hash(base)
    assert baseline_hash == lora["base_hash"], "LoRA and Step 11 baselines differ"
    base_params = dict(base.named_parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
    tokenizer.name_or_path = snapshot["model"]
    patch_chat_template(tokenizer, config.chat_templates)

    cfg = tc.TinyChangeConfig(variants_device="cuda:0")
    settings = [
        ("noise", {"type": "random_noise", "scale": 2**-12}),
        ("magnitude", {"type": "weight_pruning", "method": "magnitude", "scale": 2**-8}),
        ("random", {"type": "weight_pruning", "method": "random", "scale": 2**-8}),
    ]
    assert cfg.seed == 0

    for name, description in settings:
        dest = checkpoints / name
        if dest.exists():
            saved = json.loads((dest / "COMPLETE.json").read_text())
            assert saved["base_hash"] == baseline_hash
            assert saved["revision"] == snapshot["revision"]
            assert saved["description"] == description
            assert saved["seed"] == cfg.seed
            assert get_model_hash(load_bf16(dest)) == saved["reloaded_hash"]
            print(f"{name}: existing checkpoint verified.", flush=True)
            continue

        print(f"Creating {name} from the original Qwen model...", flush=True)
        change = tc.TinyChange(base, tokenizer, cfg)
        if name == "noise":
            variant = change.random_noise(scale=description["scale"])
        else:
            variant = change.weight_pruning(scale=description["scale"], method=name)

        assert variant.description == description
        assert get_model_hash(base) == baseline_hash, "Original weights changed"
        with tempfile.TemporaryDirectory(dir=checkpoints, prefix=f".{name}-") as tmp:
            staged = Path(tmp) / "checkpoint"
            variant.model.save_pretrained(staged, safe_serialization=True)
            tokenizer.save_pretrained(staged)
            reloaded = load_bf16(staged)
            verified = verify_saved_weights(variant.model, reloaded)

            params = dict(reloaded.named_parameters())
            assert params.keys() == base_params.keys()
            changed = 0
            for key, original in base_params.items():
                actual = params[key].detach()
                assert actual.dtype == original.dtype == torch.bfloat16, key
                assert actual.shape == original.shape, key
                count = torch.count_nonzero(actual != original.detach()).item()
                if name != "noise" and count:
                    assert "mlp" in key and key.endswith(".weight"), key
                changed += int(count)

            report = {
                "model": snapshot["model"],
                "revision": snapshot["revision"],
                "description": description,
                "seed": cfg.seed,
                "rng_policy": "TinyChange initialized independently for each variant",
                "base_device": str(base.device),
                "creation_device": cfg.variants_device,
                "creation_dtype": str(variant.model.dtype),
                "base_hash": baseline_hash,
                "upstream_variant_hash": variant.model_hash,
                "reloaded_hash": get_model_hash(reloaded),
                "verified_checkpoint_tensors": verified,
                "changed_bf16_parameter_values": changed,
            }
            (staged / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
            staged.rename(dest)

        print(f"{name}: {changed:,} BF16 parameter values changed; "
              f"{verified} checkpoint tensors verified.", flush=True)
        del params, actual, reloaded, variant, change
        gc.collect()
        torch.cuda.empty_cache()

    print("Step 11 passed: all three Qwen weight-perturbation checkpoints verified.")
    print("Checkpoint directory:", checkpoints)


if __name__ == "__main__":
    main()
