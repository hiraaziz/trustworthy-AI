
import gc
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from track_llm_apis.sampling.common import CompressedOutput, DataSource
from track_llm_apis.tinychange import TinyChangeModel


def main():
    root = Path(sys.argv[1])
    results = root / "results/qwen"
    raw = (results / "evaluation_manifest.json").read_bytes()
    manifest = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    probes = manifest["probes"]
    assert len(probes) == len(set(probes)) == 120
    assert [c["id"] for c in manifest["conditions"]] == [
        "unchanged", "lora_128", "noise", "magnitude", "random"
    ]

    destination = results / "figure1_merged"
    expected = Counter()
    records = []
    baseline_lengths = None

    for condition in manifest["conditions"]:
        name = condition["id"]
        path = results / "observations" / f"{name}.json"
        record = json.loads(path.read_text())
        assert record["manifest_sha256"] == manifest_hash, name
        assert record["model"] == manifest["model"] and record["revision"] == manifest["revision"], name
        assert record["condition"] == name and record["description"] == condition["description"], name
        assert record["inference_dtype"] == "torch.bfloat16" and record["output_tokens"] == 1, name
        if name != "unchanged":
            assert record["checkpoint_hash"] == condition["checkpoint_reloaded_hash"], name

        counts, lengths = record["counts"], record["prompt_lengths"]
        assert set(counts) == set(lengths) == set(probes), name
        if baseline_lengths is None:
            baseline_lengths = lengths
        assert lengths == baseline_lengths, name

        label = TinyChangeModel.description_to_str(condition["description"])
        for prompt, outputs in counts.items():
            assert all(type(n) is int and n > 0 for n in outputs.values()), (name, prompt)
            assert sum(outputs.values()) == 5000, (name, prompt)
            for text, n in outputs.items():
                expected[(label, (prompt, lengths[prompt]), (text, 1))] = n
        records.append((label, counts, lengths))
        print(f"{name}: 120 probes × 5,000 observations verified.", flush=True)

    assert sum(expected.values()) == 3_000_000

    def verify_archive(folder):
        loaded = CompressedOutput.from_json_dir(folder)
        assert loaded.model_name == manifest["model"]
        assert len(loaded.rows) == 3_000_000
        actual = Counter()
        for row in loaded.rows:
            assert row.source == DataSource.BI.value
            actual[(row.variant, row.prompt, row.text)] += 1
        assert actual == expected, "Reloaded archive differs from the observation files."

    if destination.exists():
        verify_archive(destination)
    else:
        with tempfile.TemporaryDirectory(dir=results, prefix=".qwen-merge-") as temp:
            staged = Path(temp) / "figure1_merged"
            merged = CompressedOutput(model_name=manifest["model"], gpus=["NVIDIA A40"])
            for label, counts, lengths in records:
                merged.add_bi_results(counts, lengths, variant=label, output_tokens=1)
            merged.dump_json(staged)
            del merged
            gc.collect()
            verify_archive(staged)
            staged.rename(destination)

    archive, = destination.glob("*.json.gz")
    print("Scoring input:", archive)
    print("Step 15 passed: 3,000,000 Qwen observations reload-verified in upstream format.")


if __name__ == "__main__":
    main()
