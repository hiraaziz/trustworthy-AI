
import hashlib
import itertools
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_param_sweeps import BIParameterSweep, compute_auc_from_stats
from track_llm_apis.sampling.common import (
    CompressedOutput, DataSource, get_analysis_dir, load_single_variant_results,
)
from track_llm_apis.tinychange import TinyChangeModel


def main():
    results = Path(sys.argv[1])
    manifest_file = results / "evaluation_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    directory = results / "figure1_merged"
    archive, = directory.glob("*.json.gz")
    analysis = config.analysis
    analysis.device = "cpu"
    prompts = analysis.bi_n_prompts_values
    samples = analysis.bi_detection_samples_values
    reference = manifest["reference_samples_per_prompt"]
    assert prompts == [1, 3, 5, 10] and samples == [1, 3, 10, 20, 50]
    assert reference == 50 and analysis.n_tests == 1000
    assert analysis.param_chunk_index is None and config.seed == 0

    variants = [
        TinyChangeModel.description_to_str(c["description"])
        for c in manifest["conditions"] if c["id"] != "unchanged"
    ]
    expected = set(itertools.product(variants, prompts, samples))
    assert len(expected) == 80

    def verify(folder):
        rows = load_single_variant_results([folder], DataSource.BI)
        assert len(rows) == 80, f"Expected 80 result files, found {len(rows)}"
        aucs = {}
        for row in rows:
            assert row.model_name == manifest["model"] and row.method == DataSource.BI
            key = (row.variant, row.params["n_prompts"], row.params["detection_samples"])
            assert key in expected and key not in aucs, key
            h0, h1 = row.h0_stats.stats, row.h1_stats.stats
            assert row.n_tests == 1000 and (len(h0), len(h1)) == (2000, 1000), key
            assert all(math.isfinite(x) for x in itertools.chain(h0, h1)), key
            auc = float(compute_auc_from_stats(h0, h1, sampling=False))
            assert 0 <= auc <= 1, key
            aucs[key] = auc
        assert set(aucs) == expected
        return aucs

    inputs = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "seed": config.seed,
        "reference_samples": reference,
        "n_tests": analysis.n_tests,
        "device": analysis.device,
    }
    destination = get_analysis_dir(directory, DataSource.BI)
    marker = destination / "scoring_inputs.meta"
    if destination.exists():
        assert marker.is_file() and json.loads(marker.read_text()) == inputs, (
            f"Existing analysis has different or unverified provenance: {destination}"
        )
        aucs = verify(directory)
    else:
        data = CompressedOutput.from_json_dir(directory)
        assert data.model_name == manifest["model"] and len(data.rows) == 3_000_000
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        with tempfile.TemporaryDirectory(dir=results, prefix=".scoring-") as tmp:
            stage = Path(tmp)
            BIParameterSweep.compute_stats(
                directory=stage, data=data,
                n_prompts_values=prompts, detection_samples_values=samples,
                reference_samples_per_prompt=reference, n_tests=analysis.n_tests,
            )
            aucs = verify(stage)
            staged_results = get_analysis_dir(stage, DataSource.BI)
            (staged_results / marker.name).write_text(json.dumps(inputs, indent=2))
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_results.rename(destination)

    print(f"Verified {len(aucs)} comparisons across {len(prompts) * len(samples)} settings.")
    print("Statistics per comparison: 2,000 H0 and 1,000 H1")
    print(f"Per-variant ROC-AUC range: {min(aucs.values()):.6f} to {max(aucs.values()):.6f}")
    print("Saved statistics:", destination)
    print("Step 41 passed: all 80 Phi B3IT comparisons scored and verified.")


if __name__ == "__main__":
    main()
