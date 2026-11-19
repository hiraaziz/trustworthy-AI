
import csv
import gzip
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from track_llm_apis.config import config
from track_llm_apis.sampling.common import CompressedOutput, SingleVariantResult
from track_llm_apis.sampling.analyze_param_sweeps import BIParameterSweep
from track_llm_apis.tinychange import TinyChange


def main():
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    config.analysis.param_chunk_index = None

    directory = root / "results/detection/merged"
    paths = list(directory.glob("*.json.gz"))
    assert len(paths) == 1
    with gzip.open(paths[0], "rt") as f:
        data = CompressedOutput.from_json(json.load(f))

    mapping = json.loads((directory / "condition_mapping.json").read_text())
    assert mapping["original"] == TinyChange.unchanged_str()
    assert set(data.get_rows_by_variant()) == set(mapping.values())

    out = root / "results/scoring"
    out.mkdir(exist_ok=True)
    settings = {
        "n_prompts": 5,
        "detection_samples_per_prompt": 3,
        "reference_samples_per_prompt": 50,
        "n_tests": 1000,
        "seed": 42,
        "h0": "original/original and edited/edited, pooled by repository scorer",
        "h1": "original reference versus edited detection",
        "condition_mapping": mapping,
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])

    BIParameterSweep.compute_stats(
        directory=directory,
        data=data,
        n_prompts_values=[5],
        detection_samples_values=[3],
        reference_samples_per_prompt=50,
        n_tests=1000,
    )

    results = {}
    for path in (directory / "analysis/bi").glob("*.json"):
        result = SingleVariantResult.model_validate_json(path.read_bytes())
        if result.params != {"n_prompts": 5, "detection_samples": 3}:
            continue
        metadata = json.loads(result.variant)
        label = metadata["label"]
        assert result.variant == mapping[label]
        assert len(result.h0_stats.stats) == 2000
        assert len(result.h1_stats.stats) == 1000

        auc = roc_auc_score(
            [0] * 2000 + [1] * 1000,
            result.h0_stats.stats + result.h1_stats.stats,
        )
        results[label] = {
            "variant": label,
            "replacement": metadata["replacement"],
            "tokens_replaced": 1,
            "roc_auc": float(auc),
        }

    assert set(results) == set("ABCD")
    summary = out / "b3it_auc.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results["A"]))
        writer.writeheader()
        for label in "ABCD":
            writer.writerow(results[label])
            print(
                f'{label} | helpful → {results[label]["replacement"]} | '
                f'ROC-AUC {results[label]["roc_auc"]:.8f}',
                flush=True,
            )

    print("Saved:", summary)
    print("Raw scores:", directory / "analysis/bi")
    print("STEP 15 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
