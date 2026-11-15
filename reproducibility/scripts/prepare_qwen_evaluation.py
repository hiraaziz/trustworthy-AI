
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import BIPhase1bOutput


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def encode_all(path, texts):
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return tokenizer(texts, padding=False, truncation=False)["input_ids"]


def main():
    root = Path(sys.argv[1])
    results = root / "results/qwen"
    snap = json.loads((root / "audit/qwen_snapshot.json").read_text())
    selection = json.loads((results / "selected_probes.json").read_text())
    prep = json.loads((root / "audit/lmsys_preparation.json").read_text())
    model = snap["model"]

    assert config.data_dir == root / "data"
    assert selection["model"] == model and selection["revision"] == snap["revision"]
    bi = config.sampling.bi
    lp = config.sampling.logprob

    calibration = BIPhase1bOutput.load_model(model, temperature=bi.temperature)
    probes = calibration.get_border_inputs()[:bi.max_border_inputs]
    assert probes == selection["prompts"] and len(set(probes)) == len(probes) == 120
    assert digest(probes) == selection["prompts_sha256"]

    traffic = lp.other_prompts
    from datasets import load_from_disk
    from track_llm_apis.util import slugify

    cache = (
        Path(prep["datasets_dir"])
        / slugify(prep["dataset"], hash_length=0)
    ).resolve()
    state = json.loads((cache / "state.json").read_text())
    stored_fp = state["_fingerprint"]
    print("Step 4 fingerprint:", prep["dataset_fingerprint"], flush=True)
    print("On-disk fingerprint:", stored_fp, flush=True)
    print("Reloaded fingerprint:", lp.other_prompts_dataset._fingerprint, flush=True)
    assert stored_fp == prep["dataset_fingerprint"], "On-disk provenance differs from Step 4"

    expected_files = [(cache / f["filename"]).resolve() for f in state["_data_files"]]
    opened_files = [Path(f["filename"]).resolve() for f in lp.other_prompts_dataset.cache_files]
    assert expected_files and opened_files == expected_files, "Evaluation opened different cache files"

    reference = load_from_disk(str(cache))
    expected_traffic = [row["conversation"][0]["content"] for row in reference]
    assert traffic == expected_traffic, "Evaluation traffic content or order differs"
    print(f"Verified {len(traffic):,} ordered traffic prompts from the Step 4 cache.", flush=True)
    assert len(traffic) == prep["filtered_conversations"] == 4502
    assert (bi.temperature, bi.models[model], lp.batch_size) == (0, 1, 64)
    assert bi.per_batch_input_fraction == 0.95
    assert config.sampling.n_samples == 5000
    assert bi.reference_samples_per_prompt == 50

    texts = list(dict.fromkeys(probes + traffic))
    baseline = encode_all(snap["snapshot"], texts)
    conditions = [{
        "id": "unchanged", "model_path": snap["snapshot"],
        "description": {"type": "unchanged"},
    }]
    expected = {
        "lora_128": {"type": "finetune", "lr": 1e-6, "n_samples": 128, "lora": True},
        "noise": {"type": "random_noise", "scale": 2**-12},
        "magnitude": {"type": "weight_pruning", "method": "magnitude", "scale": 2**-8},
        "random": {"type": "weight_pruning", "method": "random", "scale": 2**-8},
    }
    base_hashes = set()
    for name, description in expected.items():
        path = results / "checkpoints" / name
        report = json.loads((path / "COMPLETE.json").read_text())
        assert report["model"] == model and report["revision"] == snap["revision"], name
        assert report["description"] == description, name
        assert report["changed_bf16_parameter_values"] > 0, name
        assert encode_all(path, texts) == baseline, f"{name}: tokenizer mismatch"
        base_hashes.add(report["base_hash"])
        conditions.append({
            "id": name, "model_path": str(path), "description": description,
            "checkpoint_reloaded_hash": report["reloaded_hash"],
        })
        print(f"{name}: variant metadata and evaluation token IDs verified.", flush=True)
    assert len(base_hashes) == 1, "Checkpoint baseline hashes differ"

    manifest = {
        "model": model, "revision": snap["revision"],
        "probes": probes, "probes_sha256": digest(probes),
        "traffic_count": len(traffic), "traffic_sha256": digest(traffic),
        "traffic_dataset_fingerprint": prep["dataset_fingerprint"],
        "conditions": conditions, "base_hash": base_hashes.pop(),
        "temperature": bi.temperature, "max_tokens": bi.models[model],
        "batch_size": lp.batch_size, "input_fraction": bi.per_batch_input_fraction,
        "observations_per_probe_per_condition": config.sampling.n_samples,
        "reference_samples_per_prompt": bi.reference_samples_per_prompt,
        "planned_target_observations": len(probes) * len(conditions) * config.sampling.n_samples,
    }
    assert manifest["planned_target_observations"] == 3_000_000
    path = results / "evaluation_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Existing evaluation manifest differs"
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)

    print("Saved:", path)
    print("Step 12 passed: 120 probes × 5 conditions × 5,000 fresh observations planned.")


if __name__ == "__main__":
    main()
