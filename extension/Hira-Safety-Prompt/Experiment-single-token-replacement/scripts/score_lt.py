
import csv
import gzip
import json
import random
from pathlib import Path

import numpy as np
import orjson
import torch
from sklearn.metrics import roc_auc_score

from track_llm_apis.config import config
from track_llm_apis.sampling.common import (
    CompressedOutput, DataSource, SingleVariantResult,
)
from track_llm_apis.sampling.analyze_param_sweeps import LTParameterSweep
from track_llm_apis.tinychange import TinyChange


def main():
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    config.analysis.device = "cpu"
    config.analysis.param_chunk_index = None
    torch.set_num_threads(1)

    collection = root / "results/lt/collection"
    settings = json.loads((collection / "settings.json").read_text())
    mapping = settings["variant_mapping"]
    assert mapping["original"] == TinyChange.unchanged_str()
    assert settings["observations_per_condition"] == 5000
    assert settings["topk"] == 20
    assert settings["prompt"] == config.sampling.logprob.default_prompt

    data = CompressedOutput(
        model_name=settings["model"], gpus=[settings["gpu"]]
    )

    for condition in ("original", "A", "B", "C", "D"):
        count = 0
        for start in range(0, 5000, 100):
            path = collection / condition / f"block_{start:05d}.json.gz"
            with gzip.open(path, "rb") as f:
                block = CompressedOutput.from_json(orjson.loads(f.read()))

            assert len(block.rows) == 100
            assert block.model_name == settings["model"]
            assert block.gpus == [settings["gpu"]]

            for row in block.rows:
                assert row.variant == mapping[condition]
                assert row.source == DataSource.LT.value
                assert row.prompt[0] == settings["prompt"]
                assert row.text[1] == 1
                assert len(row.first_token_logprobs) >= 20

            data.merge(block)
            count += len(block.rows)

        assert count == 5000
        print(f"Verified {condition}: {count} LT observations", flush=True)

    assert len(data.rows) == 25000
    directory = root / "results/lt/merged"
    data.dump_json(directory)

    out = root / "results/lt/scoring"
    out.mkdir(parents=True, exist_ok=True)
    score_settings = {
        "reference_observations": 50,
        "detection_observations": 10,
        "n_tests": 1000,
        "seed": 42,
        "prompt": settings["prompt"],
        "topk": 20,
        "gpu_used_for_collection": settings["gpu"],
        "h0": "original/original and edited/edited, pooled by repository scorer",
        "h1": "original reference versus edited detection",
    }
    (out / "settings.json").write_text(json.dumps(score_settings, indent=2))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    LTParameterSweep.compute_stats(
        directory=directory,
        data=data,
        n_samples_values=[10],
        reference_samples_per_prompt=50,
        n_tests=1000,
    )

    results = {}
    for path in (directory / "analysis/lt").glob("*.json"):
        result = SingleVariantResult.model_validate_json(path.read_bytes())
        if result.params != {"detection_samples_per_prompt": 10}:
            continue

        metadata = json.loads(result.variant)
        label = metadata["label"]
        assert result.variant == mapping[label]
        assert len(result.h0_stats.stats) == 2000
        assert len(result.h1_stats.stats) == 1000

        h0 = np.asarray(result.h0_stats.stats)
        h1 = np.asarray(result.h1_stats.stats)
        assert np.isfinite(h0).all() and np.isfinite(h1).all()
        auc = roc_auc_score(
            np.r_[np.zeros(len(h0)), np.ones(len(h1))],
            np.r_[h0, h1],
        )
        results[label] = {
            "variant": label,
            "replacement": metadata["replacement"],
            "roc_auc": float(auc),
            "highest_no_change": float(h0.max()),
            "lowest_change": float(h1.min()),
            "every_change_higher": bool(h1.min() > h0.max()),
        }

    assert set(results) == set("ABCD")
    path = out / "lt_auc_sanity_check.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results["A"]))
        writer.writeheader()
        for label in "ABCD":
            row = results[label]
            writer.writerow(row)
            print(
                f'{label} | AUC {row["roc_auc"]:.8f} | '
                f'max H0 {row["highest_no_change"]:.8f} | '
                f'min H1 {row["lowest_change"]:.8f} | '
                f'every change higher: {row["every_change_higher"]}',
                flush=True,
            )

    print("Saved:", path)
    print("Raw scores:", directory / "analysis/lt")
    print("STEP 20 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
