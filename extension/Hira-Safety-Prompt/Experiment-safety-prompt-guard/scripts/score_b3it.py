import json
import random
from pathlib import Path

import numpy as np
import torch
from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_param_sweeps import (
    BIParameterSweep, compute_auc_from_stats,
)
from track_llm_apis.sampling.common import (
    CompressedOutput, DataSource, SingleVariantResult,
    get_analysis_dir, get_variant_filename,
)
from track_llm_apis.tinychange import TinyChange

root = Path(__file__).resolve().parents[1]
merged = root / "results/detection/merged"
out = root / "results/scoring"
out.mkdir(parents=True, exist_ok=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
config.analysis.param_chunk_index = None

labels = json.loads((merged / "condition_mapping.json").read_text())
assert labels["original"] == TinyChange.unchanged_str()
data = CompressedOutput.from_json_dir(merged)
assert len(data.rows) == 1_200_000
assert set(data.references.variants) == {labels["original"], labels["guard_on"]}

params = {"n_prompts": 5, "detection_samples": 3}
reference_samples = 50
n_tests = 1000
BIParameterSweep.compute_stats(
    directory=merged,
    data=data,
    n_prompts_values=[params["n_prompts"]],
    detection_samples_values=[params["detection_samples"]],
    reference_samples_per_prompt=reference_samples,
    n_tests=n_tests,
)

raw_path = get_analysis_dir(merged, DataSource.BI) / get_variant_filename(
    labels["guard_on"], params
)
result = SingleVariantResult.model_validate_json(raw_path.read_bytes())
assert result.variant == labels["guard_on"] and result.n_tests == n_tests
h0, h1 = result.h0_stats.stats, result.h1_stats.stats
assert len(h0) == 2 * n_tests and len(h1) == n_tests
auc = float(compute_auc_from_stats(h0, h1, sampling=False))

summary = {
    "roc_auc": auc,
    "seed": seed,
    "probes_per_test": 5,
    "reference_observations_per_probe": reference_samples,
    "detection_observations_per_probe": 3,
    "tests_per_comparison": n_tests,
    "no_change_scores": len(h0),
    "change_scores": len(h1),
    "raw_scores": str(raw_path.relative_to(root)),
}
path = out / "hidden_guard_b3it.json"
path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Hidden-guard B3IT ROC-AUC: {auc:.8f}")
print(f"No-change scores: {len(h0)} | change scores: {len(h1)}")
print("Saved:", path)
