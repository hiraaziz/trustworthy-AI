import gzip
import json
from collections import Counter
from pathlib import Path

from track_llm_apis.sampling.common import CompressedOutput
from track_llm_apis.tinychange import TinyChange


def main():
    root = Path(__file__).resolve().parents[1]
    detection = root / "results/detection"
    probes = json.loads((detection / "probes.json").read_text())
    assert len(probes) == len(set(probes)) == 120

    def settings(condition):
        return json.loads((detection / condition / "settings.json").read_text())

    baseline = settings("original")
    guarded = settings("guard_on")
    assert baseline["condition"] == "guard_off" and guarded["condition"] == "guard_on"
    for key in (
        "model", "revision", "system_prompt", "probes_in_column_order",
        "rounds", "block_size", "round_seed_base", "temperature",
        "batch_size", "input_fraction", "max_tokens",
    ):
        if baseline[key] != guarded[key]:
            raise ValueError(f"Collection settings differ for {key}")

    assert baseline["probes_in_column_order"] == probes
    assert baseline["rounds"] == 5000 and baseline["max_tokens"] == 1

    # Use the original request as the probe identity in both conditions.
    # These lengths describe baseline requests and exclude the hidden guard's costs.
    calibration = json.loads((root / "results/calibration/phase_1b.json").read_text())
    lengths = {q: calibration["prompt_lengths"][q] for q in probes}

    labels = {
        "original": TinyChange.unchanged_str(),
        "guard_on": json.dumps({
            "type": "hidden_input_guard",
            "model": guarded["guard"]["model"],
            "trigger": "Unsafe",
            "prefix": guarded["guard"]["prefix"],
        }),
    }
    merged = CompressedOutput(model_name=baseline["model"], gpus=["NVIDIA A40"])
    rounds, block_size = baseline["rounds"], baseline["block_size"]

    for condition, label in labels.items():
        counts = {q: Counter() for q in probes}
        for start in range(0, rounds, block_size):
            path = detection / condition / f"rounds_{start:04d}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as f:
                rows = json.load(f)

            if len(rows) != min(block_size, rounds - start):
                raise ValueError(f"Unexpected round count in {path}")
            for row in rows:
                for q, token in zip(probes, row, strict=True):
                    if not isinstance(token, str):
                        raise ValueError(f"Invalid token in {path}")
                    counts[q][token] += 1

        assert all(sum(c.values()) == rounds for c in counts.values())
        merged.add_bi_results(counts, lengths, label, output_tokens=1)
        print(f"Verified {condition}: {len(probes)} × {rounds} = 600,000", flush=True)

    assert len(merged.rows) == 1_200_000
    out = detection / "merged"
    merged.dump_json(out)
    (out / "condition_mapping.json").write_text(json.dumps(labels, indent=2))
    print("Merged observations:", len(merged.rows))
    print("Saved:", out)


if __name__ == "__main__":
    main()
