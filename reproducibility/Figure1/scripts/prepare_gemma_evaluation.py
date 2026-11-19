
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import BIPhase1bOutput
from track_llm_apis.util import slugify


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def encode(path, texts):
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return tokenizer(texts, padding=False, truncation=False)["input_ids"]


def main():
    root = Path(sys.argv[1])
    results = root / "results/gemma"
    snap = json.loads((root / "audit/gemma_snapshot.json").read_text())
    selection = json.loads((results / "selected_probes.json").read_text())
    prep = json.loads((root / "audit/lmsys_preparation.json").read_text())
    model = "google/gemma-3-1b-it"

    assert model == snap["model"] == selection["model"]
    assert snap["revision"] == selection["revision"] == Path(snap["snapshot"]).name
    assert config.data_dir == root / "data"
    bi, lp = config.sampling.bi, config.sampling.logprob

    calibration = BIPhase1bOutput.load_model(model, temperature=bi.temperature)
    probes = calibration.get_border_inputs()[:bi.max_border_inputs]
    assert probes == selection["prompts"]
    assert len(probes) == len(set(probes)) == 120
    assert digest(probes) == selection["prompts_sha256"]

    # Check the stored fingerprint and the ordered content used by evaluation.
    cache = (Path(prep["datasets_dir"]) / slugify(prep["dataset"], hash_length=0)).resolve()
    state = json.loads((cache / "state.json").read_text())
    assert state["_fingerprint"] == prep["dataset_fingerprint"] == selection["traffic_dataset_fingerprint"]
    traffic = lp.other_prompts
    expected_files = [(cache / f["filename"]).resolve() for f in state["_data_files"]]
    opened_files = [Path(f["filename"]).resolve() for f in lp.other_prompts_dataset.cache_files]
    assert opened_files == expected_files and expected_files
    assert traffic == [r["conversation"][0]["content"] for r in load_from_disk(str(cache))]
    assert len(traffic) == prep["filtered_conversations"] == 4502
    print("Verified 120 ordered Gemma probes and 4,502 ordered traffic prompts.", flush=True)

    assert (bi.temperature, bi.models[model], lp.batch_size) == (0, 1, 64)
    assert bi.per_batch_input_fraction == 0.95
    assert config.sampling.n_samples == 5000 and bi.reference_samples_per_prompt == 50

    texts = list(dict.fromkeys(probes + traffic))
    baseline_ids = encode(snap["snapshot"], texts)
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
        assert report["reloaded_hash"], name
        assert encode(path, texts) == baseline_ids, f"{name}: tokenizer IDs differ"
        base_hashes.add(report["base_hash"])
        conditions.append({
            "id": name, "model_path": str(path), "description": description,
            "checkpoint_reloaded_hash": report["reloaded_hash"],
        })
        print(f"{name}: checkpoint metadata and token IDs verified.", flush=True)
    assert len(base_hashes) == 1, "The variants have different baseline hashes"

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
        assert json.loads(path.read_text()) == manifest, "Existing Gemma manifest differs"
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)

    print("Saved:", path)
    print("Step 24 passed: 120 probes × 5 conditions × 5,000 fresh observations planned.")


if __name__ == "__main__":
    main()
