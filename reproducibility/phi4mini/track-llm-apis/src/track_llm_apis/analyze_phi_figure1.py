
import gc
import gzip
import json
import random
from pathlib import Path

import numpy as np
import torch

from track_llm_apis.config import config
from track_llm_apis.sampling.common import (
    CompressedOutput, DataSource, get_analysis_dir, load_single_variant_results,
)
from track_llm_apis.sampling.analyze_param_sweeps import BIParameterSweep

def main():
    root = Path("/workspace/B3IT_Figure1_Phi4Mini")
    directory = root / "results/figure1_merged"
    files = list(directory.glob("*.json.gz"))
    assert len(files) == 1

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    config.analysis.device = "cpu"
    config.analysis.param_chunk_index = None

    print("Loading verified observations for CPU analysis.", flush=True)
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    data = CompressedOutput.from_json(raw)
    del raw
    gc.collect()
    assert len(data.rows) == 3_000_000

    print("Running B3IT: 20 settings × 4 perturbations × 1,000 tests.", flush=True)
    BIParameterSweep.compute_stats(
        directory=directory,
        data=data,
        n_prompts_values=[1, 3, 5, 10],
        detection_samples_values=[1, 3, 10, 20, 50],
        reference_samples_per_prompt=50,
        n_tests=1000,
    )

    results = load_single_variant_results([directory], DataSource.BI)
    assert len(results) == 80, f"Expected 80 results, found {len(results)}."
    for result in results:
        assert result.n_tests == 1000
        assert len(result.h0_stats.stats) == 2000
        assert len(result.h1_stats.stats) == 1000
        assert np.isfinite(result.h0_stats.stats).all()
        assert np.isfinite(result.h1_stats.stats).all()

    report = {
        "model": data.model_name,
        "parameter_combinations": 20,
        "perturbations": 4,
        "verified_results": len(results),
        "reference_samples_per_prompt": 50,
        "n_tests": 1000,
        "seed": 0,
        "analysis_directory": str(get_analysis_dir(directory, DataSource.BI)),
    }
    destination = root / "results/figure1_statistics_verification.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(destination)
    print(json.dumps(report, indent=2), flush=True)
    print("Phi Figure 1 statistics verified and saved.", flush=True)

if __name__ == "__main__":
    main()
