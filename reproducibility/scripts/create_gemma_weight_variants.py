
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.tinychange import TinyChange, TinyChangeConfig
from track_llm_apis.util import get_model_hash, patch_chat_template


def load_bf16(path):
    return AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, local_files_only=True,
    )


def verify_reload(merged, restored, base, name):
    expected = merged.state_dict()
    actual = restored.state_dict()
    assert expected.keys() == actual.keys(), f"{name}: checkpoint tensor names differ"
    for key, tensor in expected.items():
        saved = actual[key]
        assert tensor.shape == saved.shape and tensor.dtype == saved.dtype, key
        assert torch.equal(tensor.detach().cpu(), saved), f"{name}: reload mismatch at {key}"

    base_params = dict(base.named_parameters())
    saved_params = dict(restored.named_parameters())
    assert base_params.keys() == saved_params.keys()
    changed = 0
    for key, original in base_params.items():
        saved = saved_params[key].detach()
        assert saved.shape == original.shape, key
        assert saved.dtype == original.dtype == torch.bfloat16, key
        difference = saved != original.detach()
        count = int(difference.sum().item())
        if name in ("magnitude", "random") and count:
            assert "mlp" in key and key.endswith(".weight"), f"Unexpected pruned tensor: {key}"
            assert torch.count_nonzero(saved[difference]).item() == 0, key
        changed += count
    return len(actual), changed


def main():
    root = Path(sys.argv[1])
    snap = json.loads((root / "audit/gemma_snapshot.json").read_text())
    checkpoints = root / "results/gemma/checkpoints"
    lora = json.loads((checkpoints / "lora_128/COMPLETE.json").read_text())
    assert snap["model"] == lora["model"] == "google/gemma-3-1b-it"
    assert Path(snap["snapshot"]).name == snap["revision"] == lora["revision"]

    base = load_bf16(snap["snapshot"])
    assert base.device.type == "cpu"
    baseline_hash = get_model_hash(base)
    assert baseline_hash == lora["base_hash"], "Baseline differs from Step 22"
    tokenizer = AutoTokenizer.from_pretrained(snap["snapshot"], local_files_only=True)
    tokenizer.name_or_path = snap["model"]
    patch_chat_template(tokenizer, config.chat_templates)

    cfg = TinyChangeConfig(variants_device="cuda:0")
    assert cfg.seed == 0
    settings = {
        "noise": {"type": "random_noise", "scale": 2**-12},
        "magnitude": {"type": "weight_pruning", "method": "magnitude", "scale": 2**-8},
        "random": {"type": "weight_pruning", "method": "random", "scale": 2**-8},
    }

    for name, description in settings.items():
        dest = checkpoints / name
        if dest.exists():
            report = json.loads((dest / "COMPLETE.json").read_text())
            assert report["revision"] == snap["revision"] and report["base_hash"] == baseline_hash
            assert report["description"] == description and report["seed"] == cfg.seed
            assert get_model_hash(load_bf16(dest)) == report["reloaded_hash"]
            print(f"{name}: reused verified checkpoint; "
                  f"{report['changed_bf16_parameter_values']:,} changed values.", flush=True)
            continue

        print(f"Creating {name} from the original Gemma weights...", flush=True)
        change = TinyChange(base, tokenizer, cfg)
        if name == "noise":
            variant = change.random_noise(scale=description["scale"])
        else:
            variant = change.weight_pruning(scale=description["scale"], method=name)

        assert variant.description == description
        assert variant.model.dtype == torch.bfloat16
        assert get_model_hash(base) == baseline_hash, "Original model changed"

        with tempfile.TemporaryDirectory(dir=checkpoints, prefix=f".{name}-") as tmp:
            staged = Path(tmp) / "checkpoint"
            variant.model.save_pretrained(staged, safe_serialization=True)
            tokenizer.save_pretrained(staged)
            restored = load_bf16(staged)
            verified, changed = verify_reload(variant.model, restored, base, name)

            report = {
                "model": snap["model"],
                "revision": snap["revision"],
                "description": description,
                "seed": cfg.seed,
                "rng_policy": "TinyChange initialized independently for each variant",
                "base_device": str(base.device),
                "creation_device": cfg.variants_device,
                "creation_dtype": str(variant.model.dtype),
                "base_hash": baseline_hash,
                "upstream_variant_hash": variant.model_hash,
                "reloaded_hash": get_model_hash(restored),
                "verified_checkpoint_tensors": verified,
                "changed_bf16_parameter_values": changed,
            }
            (staged / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
            staged.rename(dest)

        print(f"{name}: {changed:,} BF16 values changed; "
              f"{verified} checkpoint tensors verified.", flush=True)
        del restored, variant, change
        gc.collect()
        torch.cuda.empty_cache()

    print("Checkpoint directory:", checkpoints)
    print("Step 23 passed: all three Gemma weight-perturbation checkpoints verified.")


if __name__ == "__main__":
    main()
