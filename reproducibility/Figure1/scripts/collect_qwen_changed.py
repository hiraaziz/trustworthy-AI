
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.sampling.vllm_sampling import (
    cleanup_vllm, init_vllm, load_model_to_vllm, vllm_inference_bi,
)
from track_llm_apis.util import get_model_hash, patch_chat_template
from collect_qwen_baseline import digest, verify as verify_baseline


def restore_rng(saved):
    version, state, gauss = saved
    random.setstate((version, tuple(state), gauss))


def verify_changed(record, condition, manifest, manifest_hash, start_hash, lengths):
    assert record["manifest_sha256"] == manifest_hash
    assert record["condition"] == condition["id"]
    assert record["description"] == condition["description"]
    assert record["checkpoint_hash"] == condition["checkpoint_reloaded_hash"]
    assert record["start_rng_sha256"] == start_hash
    assert record["inference_dtype"] == "torch.bfloat16"
    assert record["output_tokens"] == 1
    assert record["prompt_lengths"] == lengths
    assert set(record["counts"]) == set(manifest["probes"])
    for prompt, counts in record["counts"].items():
        assert all(type(n) is int and n > 0 for n in counts.values()), repr(prompt)
        assert sum(counts.values()) == 5000, repr(prompt)
    version, state, gauss = record["python_random_state_after_sampling"]
    random.Random().setstate((version, tuple(state), gauss))


def main():
    root = Path(sys.argv[1])
    name = sys.argv[2]
    results = root / "results/qwen"
    folder = results / "observations"
    raw = (results / "evaluation_manifest.json").read_bytes()
    manifest = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    conditions = manifest["conditions"]
    order = [c["id"] for c in conditions]
    assert order == ["unchanged", "lora_128", "noise", "magnitude", "random"]
    index = order.index(name)
    assert index > 0
    assert len(manifest["probes"]) == len(set(manifest["probes"])) == 120
    assert digest(manifest["probes"]) == manifest["probes_sha256"]
    assert manifest["observations_per_probe_per_condition"] == 5000

    previous = json.loads((folder / "unchanged.json").read_text())
    verify_baseline(previous, manifest, manifest_hash)
    lengths = previous["prompt_lengths"]

    # Verify the saved chain before continuing its traffic randomization.
    for condition in conditions[1:index + 1]:
        name_in_chain = condition["id"]
        path = folder / f"{name_in_chain}.json"
        start_state = previous["python_random_state_after_sampling"]
        start_hash = digest(start_state)
        if path.exists():
            previous = json.loads(path.read_text())
            verify_changed(previous, condition, manifest, manifest_hash, start_hash, lengths)
            if name_in_chain == name:
                print(f"{name}: reused 600,000 verified observations.", flush=True)
                return
        else:
            assert name_in_chain == name, f"Run the missing condition first: {name_in_chain}"
            break

    bi = config.sampling.bi
    lp = config.sampling.logprob
    traffic = lp.other_prompts
    assert digest(traffic) == manifest["traffic_sha256"]
    assert (lp.batch_size, bi.per_batch_input_fraction) == (64, 0.95)
    assert bi.temperature == 0 and bi.models[manifest["model"]] == 1

    # Use the pinned baseline tokenizer for every condition, as in Step 13.
    tokenizer = AutoTokenizer.from_pretrained(conditions[0]["model_path"], local_files_only=True)
    patch_chat_template(tokenizer, config.chat_templates)
    model = AutoModelForCausalLM.from_pretrained(
        condition["model_path"], torch_dtype=torch.bfloat16, local_files_only=True,
    )
    assert get_model_hash(model) == condition["checkpoint_reloaded_hash"], name
    model = model.to("cuda:0")
    engine = init_vllm(model, tokenizer, "cuda:0")

    try:
        load_model_to_vllm(engine, model)
        dtype = engine.llm_engine.vllm_config.model_config.dtype
        assert dtype == torch.bfloat16
        engine_tokenizer = engine.get_tokenizer()
        for text in dict.fromkeys(manifest["probes"] + traffic):
            assert engine_tokenizer.encode(text) == tokenizer.encode(text), name

        restore_rng(start_state)
        print(f"Collecting {name}: 120 probes × 5,000 observations.", flush=True)
        sampled = vllm_inference_bi(
            llm=engine, inputs=manifest["probes"], traffic_prompts=traffic,
            n_samples=5000, temperature=bi.temperature,
            input_fraction=bi.per_batch_input_fraction,
            batch_size=lp.batch_size, max_tokens=1,
        )
        record = {
            "model": manifest["model"], "revision": manifest["revision"],
            "condition": name, "description": condition["description"],
            "checkpoint_hash": condition["checkpoint_reloaded_hash"],
            "manifest_sha256": manifest_hash, "inference_dtype": str(dtype),
            "start_rng_sha256": start_hash,
            "python_random_state_after_sampling": random.getstate(),
            "counts": sampled.counts, "prompt_lengths": sampled.prompt_lengths,
            "output_tokens": sampled.output_tokens,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        verify_changed(json.loads(tmp.read_text()), condition, manifest,
                       manifest_hash, start_hash, lengths)
        tmp.replace(path)
        print(f"{name}: 600,000 observations verified and saved to {path}", flush=True)
    finally:
        cleanup_vllm(engine)


if __name__ == "__main__":
    main()
