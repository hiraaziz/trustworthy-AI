
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


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def verify(record, manifest, manifest_hash):
    assert record["manifest_sha256"] == manifest_hash
    assert record["model"] == manifest["model"] and record["revision"] == manifest["revision"]
    assert record["condition"] == "unchanged"
    assert record["base_hash"] == manifest["base_hash"]
    assert record["inference_dtype"] == "torch.bfloat16" and record["output_tokens"] == 1
    probes = manifest["probes"]
    assert set(record["counts"]) == set(record["prompt_lengths"]) == set(probes)
    for prompt in probes:
        counts = record["counts"][prompt]
        assert all(type(n) is int and n > 0 for n in counts.values()), repr(prompt)
        assert sum(counts.values()) == 5000, repr(prompt)
        assert record["prompt_lengths"][prompt] > 0
    version, state, gauss = record["python_random_state_after_sampling"]
    random.Random().setstate((version, tuple(state), gauss))


def main():
    root = Path(sys.argv[1])
    folder = root / "results/gemma"
    raw = (folder / "evaluation_manifest.json").read_bytes()
    manifest = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()

    assert manifest["model"] == "google/gemma-3-1b-it"
    probes = manifest["probes"]
    assert len(probes) == len(set(probes)) == 120
    assert digest(probes) == manifest["probes_sha256"]
    assert manifest["observations_per_probe_per_condition"] == 5000

    out = folder / "observations/unchanged.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        verify(json.loads(out.read_text()), manifest, manifest_hash)
        print("Step 25 passed: existing Gemma baseline reverified — 600,000 observations.")
        print("Saved:", out)
        return

    assert config.data_dir == root / "data"
    bi, lp = config.sampling.bi, config.sampling.logprob
    traffic = lp.other_prompts
    assert digest(traffic) == manifest["traffic_sha256"]
    assert (lp.batch_size, bi.per_batch_input_fraction) == (64, 0.95)
    assert bi.temperature == manifest["temperature"] == 0
    assert bi.models[manifest["model"]] == manifest["max_tokens"] == 1

    baseline, = [c for c in manifest["conditions"] if c["id"] == "unchanged"]
    path = Path(baseline["model_path"])
    assert path.name == manifest["revision"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    patch_chat_template(tokenizer, config.chat_templates)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    assert get_model_hash(model) == manifest["base_hash"], "Gemma baseline hash differs"
    model = model.to("cuda:0")

    engine = init_vllm(model, tokenizer, "cuda:0")
    try:
        load_model_to_vllm(engine, model)
        dtype = engine.llm_engine.vllm_config.model_config.dtype
        assert dtype == torch.bfloat16, f"Unexpected engine dtype: {dtype}"

        engine_tokenizer = engine.get_tokenizer()
        for text in dict.fromkeys(probes + traffic):
            assert engine_tokenizer.encode(text) == tokenizer.encode(text), (
                "Engine tokenizer differs from the verified Gemma tokenizer."
            )

        random.seed(config.seed)
        print("Collecting unchanged Gemma: 120 probes × 5,000 observations.", flush=True)
        sampled = vllm_inference_bi(
            llm=engine, inputs=probes, traffic_prompts=traffic,
            n_samples=5000, temperature=bi.temperature,
            input_fraction=bi.per_batch_input_fraction,
            batch_size=lp.batch_size, max_tokens=1,
        )
        record = {
            "model": manifest["model"], "revision": manifest["revision"],
            "condition": "unchanged", "description": baseline["description"],
            "base_hash": manifest["base_hash"], "manifest_sha256": manifest_hash,
            "inference_dtype": str(dtype), "sampling_seed": config.seed,
            "counts": sampled.counts, "prompt_lengths": sampled.prompt_lengths,
            "output_tokens": sampled.output_tokens,
            "python_random_state_after_sampling": random.getstate(),
        }
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        verify(json.loads(tmp.read_text()), manifest, manifest_hash)
        tmp.replace(out)
    finally:
        cleanup_vllm(engine)

    print("Step 25 passed: 120 unchanged Gemma probes × 5,000 observations verified.")
    print("Total fresh baseline observations: 600,000")
    print("Saved:", out)


if __name__ == "__main__":
    main()
