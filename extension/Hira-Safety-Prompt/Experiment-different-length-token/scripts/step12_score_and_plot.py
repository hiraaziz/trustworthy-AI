import csv
import gzip
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import orjson
import torch

from track_llm_apis.config import config
from track_llm_apis.tinychange import TinyChange
from track_llm_apis.sampling.common import (
    CompressedOutput, DataSource, SingleVariantResult,
    get_analysis_dir, get_variant_filename,
)
from track_llm_apis.sampling.analyze_param_sweeps import (
    BIParameterSweep, compute_auc_from_stats,
)
from track_llm_apis.util import ci

root = Path(__file__).resolve().parents[1]
config.data_dir = root / "data"
config.analysis.device = "cpu"
out = root / "results/scoring_with_ci"
out.mkdir(parents=True, exist_ok=True)

# All scoring settings are explicit and saved alongside the results.
settings = {
    "probes_per_test": 5,
    "reference_per_probe": 50,
    "detection_per_probe": 3,
    "tests_per_comparison": 1000,
    "bootstrap_resamples_per_variant": 250,
    "confidence_level": 0.95,
    "scoring_seed": 0,
    "bootstrap_seed": 2026,
}
lengths = {"A": 3, "B": 4, "C": 5, "D": 8}
labels = {"original": TinyChange.unchanged_str()}
for label, length in lengths.items():
    labels[label] = json.dumps({
        "type": "system_prompt", "label": label, "added_tokens": length
    })
(out / "settings.json").write_text(json.dumps(
    {"settings": settings, "baseline_and_variant_mapping": labels}, indent=2
))

# Keep each observation's index and translate only its condition label.
merged = root / "results/detection/merged"
data_file, = merged.glob("*.json.gz")
with gzip.open(data_file, "rb") as f:
    raw = orjson.loads(f.read())
names = raw["references"]["variants"]
assert len(names) == 5 and set(names) == set(labels)
assert len(raw["rows"]) == 3_000_000
raw["references"]["variants"] = [labels[name] for name in names]
data = CompressedOutput.from_json(raw)
del raw

random.seed(settings["scoring_seed"])
torch.manual_seed(settings["scoring_seed"])
BIParameterSweep.compute_stats(
    directory=out,
    data=data,
    n_prompts_values=[settings["probes_per_test"]],
    detection_samples_values=[settings["detection_per_probe"]],
    reference_samples_per_prompt=settings["reference_per_probe"],
    n_tests=settings["tests_per_comparison"],
)
del data

previous_csv = root / "results/scoring/short_prompt_auc.csv"
previous = {}
if previous_csv.exists():
    with previous_csv.open(newline="") as f:
        previous = {r["variant"]: float(r["roc_auc"]) for r in csv.DictReader(f)}

params = {"n_prompts": 5, "detection_samples": 3}
stats_dir = get_analysis_dir(out, DataSource.BI)
rows = []
draws = {}
random.seed(settings["bootstrap_seed"])

for label, length in lengths.items():
    path = stats_dir / get_variant_filename(labels[label], params)
    result = SingleVariantResult.model_validate_json(path.read_bytes())
    h0 = result.h0_stats.stats  # Original/original plus variant/variant
    h1 = result.h1_stats.stats  # Original/variant
    assert len(h0) == 2000 and len(h1) == 1000

    auc = compute_auc_from_stats(h0, h1, sampling=False)
    draws[label] = [
        compute_auc_from_stats(h0, h1, sampling=True)
        for _ in range(settings["bootstrap_resamples_per_variant"])
    ]
    low, high = ci(draws[label], alpha=0.05)
    rows.append((label, length, auc, low, high))
    comparison = ""
    if label in previous:
        comparison = f" | previous {previous[label]:.6f} | delta {auc-previous[label]:+.6f}"
    print(f"{label}: AUC {auc:.6f} | 95% CI [{low:.6f}, {high:.6f}]{comparison}",
          flush=True)

with (out / "short_prompt_auc_with_ci.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variant", "added_tokens", "roc_auc", "ci_low", "ci_high"])
    writer.writerows(rows)
(out / "bootstrap_draws.json").write_text(json.dumps(draws))

fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
for label, length, auc, low, high in rows:
    ax.vlines(length, low, high, color="#205781", linewidth=1.5)
    ax.hlines([low, high], length - 0.08, length + 0.08, color="#205781")
    ax.scatter(length, auc, color="#205781", zorder=3)
ax.set_xticks(list(lengths.values()), [f"{k} (+{v})" for k, v in lengths.items()])
ax.set_xlabel("Variant and tokens added to the original system prompt")
ax.set_ylabel("Detection ROC-AUC")
# Zoom so the bootstrap intervals near 1.0 are visible.
ax.set_ylim(min(row[3] for row in rows) - 0.004, 1.001)
ax.grid(axis="y", alpha=0.25)
ax.set_title("Qwen2.5-0.5B-Instruct: short safety instructions\n"
             "5 probes per test · 95% bootstrap intervals")
for ext in ("png", "pdf"):
    fig.savefig(out / f"short_prompt_auc_with_ci.{ext}", dpi=300)
plt.close(fig)
print("Saved:", out)
print("SCORING AND BOOTSTRAP COMPLETE")
