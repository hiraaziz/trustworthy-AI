
import gzip
import json
from pathlib import Path

from track_llm_apis.sampling.common import CompressedOutput
from track_llm_apis.tinychange import TinyChange


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/experiment.json").read_text())
    probes = json.loads((root / "results/detection/probes.json").read_text())
    detection = root / "results/detection"

    labels = {"original": TinyChange.unchanged_str()}
    labels.update({
        v["label"]: json.dumps({
            "type": "system_prompt",
            "label": v["label"],
            "replacement": v["replacement_word"],
            "tokens_replaced": 1,
        })
        for v in plan["variants"]
    })

    merged = CompressedOutput(
        model_name=plan["model"], gpus=["NVIDIA RTX A6000"]
    )
    baseline_lengths = None

    for condition, variant in labels.items():
        folder = detection / condition
        settings = json.loads((folder / "settings.json").read_text())
        assert settings["condition"] == condition
        assert settings["probes"] == probes
        assert settings["rounds"] == 5000
        assert settings["model"] == plan["model"]
        assert settings["revision"] == plan["model_revision"]
        assert settings["dtype"] == plan["dtype"]

        expected_prompt = (
            plan["baseline_system_prompt"] if condition == "original"
            else next(
                v["system_prompt"] for v in plan["variants"]
                if v["label"] == condition
            )
        )
        assert settings["system_prompt"] == expected_prompt

        seen_rounds = set()
        condition_lengths = None
        before = len(merged.rows)

        for start in range(0, 5000, 250):
            path = folder / f"block_{start:05d}.json.gz"
            with gzip.open(path, "rt") as f:
                block = json.load(f)

            lengths = block["prompt_lengths"]
            assert set(lengths) == set(probes)
            if condition_lengths is None:
                condition_lengths = lengths
            assert lengths == condition_lengths

            rows = block["rows"]
            assert [r["round"] for r in rows] == list(range(start, start + 250))

            for row in rows:
                assert row["round"] not in seen_rounds
                seen_rounds.add(row["round"])
                assert set(row["tokens"]) == set(probes)

                merged.add_bi_results(
                    counts={p: {row["tokens"][p]: 1} for p in probes},
                    prompt_lengths=lengths,
                    variant=variant,
                    output_tokens=1,
                )

        assert seen_rounds == set(range(5000))
        assert len(merged.rows) - before == 600000

        if baseline_lengths is None:
            baseline_lengths = condition_lengths
        assert condition_lengths == baseline_lengths, (
            f"Prompt lengths differ for {condition}"
        )

        print(
            f"Verified {condition}: 120 probes × 5000 = 600,000 observations",
            flush=True,
        )

    assert len(merged.rows) == 3000000
    out = detection / "merged"
    merged.dump_json(out)
    (out / "condition_mapping.json").write_text(json.dumps(labels, indent=2))

    print("Merged output:", out, flush=True)
    print("STEP 14 COMPLETE | 3,000,000 observations verified", flush=True)


if __name__ == "__main__":
    main()
