
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
from track_llm_apis.util import patch_chat_template


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def verify(record, manifest, manifest_hash):
    probes = manifest["probes"]
    assert record["manifest_sha256"] == manifest_hash
    assert record["condition"] == "unchanged"
    assert record["output_tokens"] == 1
    assert set(record["counts"]) == set(record["prompt_lengths"]) == set(probes)
    for p in probes:
        counts = record["counts"][p]
        assert all(type(n) is int and n > 0 for n in counts.values()), repr(p)
        assert sum(counts.values()) == 5000, repr(p)
        assert record["prompt_lengths"][p] > 0
    version, state, gauss = record["python_random_state_after_sampling"]
    random.Random().setstate((version, tuple(state), gauss))


def main():
    root = Path(sys.argv[1])
    results = root / "results/qwen"
    manifest_file = results / "evaluation_manifest.json"
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    out = results / "observations/unchanged.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        verify(json.loads(out.read_text()), manifest, manifest_hash)
        print("Step 13 passed: saved baseline reverified — 600,000 observations.")
        print("Saved:", out)
        return

    probes = manifest["probes"]
    n = manifest["observations_per_probe_per_condition"]
    assert len(set(probes)) == len(probes) == 120 and n == 5000
    assert digest(probes) == manifest["probes_sha256"]
    assert config.data_dir == root / "data"

    lp = config.sampling.logprob
    bi = config.sampling.bi
    traffic = lp.other_prompts
    assert digest(traffic) == manifest["traffic_sha256"]
    assert (lp.batch_size, bi.per_batch_input_fraction) == (64, 0.95)
    assert bi.temperature == manifest["temperature"] == 0
    assert bi.models[manifest["model"]] == manifest["max_tokens"] == 1

    baseline = next(c for c in manifest["conditions"] if c["id"] == "unchanged")
    path = Path(baseline["model_path"])
    assert path.name == manifest["revision"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    patch_chat_template(tokenizer, config.chat_templates)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda:0")

    # Upstream evaluation initializes the engine, then loads condition weights via IPC.
    engine = init_vllm(model, tokenizer, "cuda:0")
    try:
        load_model_to_vllm(engine, model)
        actual_dtype = engine.llm_engine.vllm_config.model_config.dtype
        assert actual_dtype == torch.bfloat16, f"Evaluation engine dtype: {actual_dtype}"

        # Check the engine tokenizer as well as the checkpoint tokenizers from Step 12.
        engine_tokenizer = engine.get_tokenizer()
        for text in dict.fromkeys(probes + traffic):
            assert engine_tokenizer.encode(text) == tokenizer.encode(text), (
                "Evaluation engine tokenizer differs from the pinned tokenizer."
            )

        random.seed(config.seed)
        print("Collecting unchanged Qwen: 120 probes × 5,000 observations.", flush=True)
        sampled = vllm_inference_bi(
            llm=engine, inputs=probes, traffic_prompts=traffic, n_samples=n,
            temperature=bi.temperature, input_fraction=bi.per_batch_input_fraction,
            batch_size=lp.batch_size, max_tokens=manifest["max_tokens"],
        )
        record = {
            "model": manifest["model"],
            "revision": manifest["revision"],
            "condition": "unchanged",
            "description": baseline["description"],
            "manifest_sha256": manifest_hash,
            "inference_dtype": str(actual_dtype),
            "sampling_seed": config.seed,
            "counts": sampled.counts,
            "prompt_lengths": sampled.prompt_lengths,
            "output_tokens": sampled.output_tokens,
            "python_random_state_after_sampling": random.getstate(),
        }
        verify(record, manifest, manifest_hash)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        verify(json.loads(tmp.read_text()), manifest, manifest_hash)
        tmp.replace(out)
    finally:
        cleanup_vllm(engine)

    print("Step 13 passed: 120 unchanged Qwen probes × 5,000 observations verified.")
    print("Total fresh baseline observations: 600,000")
    print("Saved:", out)


if __name__ == "__main__":
    main()
