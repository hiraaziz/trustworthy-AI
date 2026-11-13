
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
    LTParameterSweep, compute_auc_from_stats,
)
from track_llm_apis.util import ci

root = Path(__file__).resolve().parents[1]
collection = root / "results/lt/collection"
out = root / "results/lt/comparison"
out.mkdir(parents=True, exist_ok=True)
config.data_dir = root / "data"
config.analysis.device = "cpu"

# Use the B3IT scores from the corrected run with bootstrap intervals.
with (root / "results/scoring_with_ci/short_prompt_auc_with_ci.csv").open() as f:
    b3it = list(csv.DictReader(f))
assert [(r["variant"], int(r["added_tokens"])) for r in b3it] == [
    ("A", 3), ("B", 4), ("C", 5), ("D", 8)
]
rows = [
    ("B3IT", r["variant"], int(r["added_tokens"]), float(r["roc_auc"]),
     float(r["ci_low"]), float(r["ci_high"]))
    for r in b3it
]

settings = json.loads((collection / "settings.json").read_text())
assert settings["probe"] == config.sampling.logprob.default_prompt == "x"
assert settings["rounds_per_condition"] == 5000
assert settings["checkpoint_rounds"] == 100
assert settings["top_logprobs"] == 20

labels = {"original": TinyChange.unchanged_str()}
for r in b3it:
    labels[r["variant"]] = json.dumps({
        "type": "system_prompt", "label": r["variant"],
        "added_tokens": int(r["added_tokens"]),
    })

# Translate the saved condition label into the format the authors' sweep expects.
data = CompressedOutput(model_name=settings["model"])
for label, repo_label in labels.items():
    for start in range(0, 5000, 100):
        folder = collection / label / f"rounds_{start:04d}"
        file, = folder.glob("*.json.gz")
        with gzip.open(file, "rb") as f:
            raw = orjson.loads(f.read())
        assert raw["references"]["variants"] == [label], file
        raw["references"]["variants"] = [repo_label]
        block = CompressedOutput.from_json(raw)
        assert len(block.rows) == 100, file
        assert all(
            r.prompt[0] == "x" and r.source == DataSource.LT.value
            and r.text[1] == 1 and len(r.first_token_logprobs) >= 20
            for r in block.rows
        ), file
        data.merge(block)
    print(f"Verified {label}: 5000 LT observations", flush=True)

assert len(data.rows) == 25000
random.seed(0)
torch.manual_seed(0)
LTParameterSweep.compute_stats(
    directory=out, data=data, n_samples_values=[10],
    reference_samples_per_prompt=50, n_tests=1000,
)

params = {"detection_samples_per_prompt": 10}
stats_dir = get_analysis_dir(out, DataSource.LT)
draws = {}
random.seed(2026)
for r in b3it:
    label = r["variant"]
    path = stats_dir / get_variant_filename(labels[label], params)
    result = SingleVariantResult.model_validate_json(path.read_bytes())
    h0, h1 = result.h0_stats.stats, result.h1_stats.stats
    assert len(h0) == 2000 and len(h1) == 1000
    auc = compute_auc_from_stats(h0, h1, sampling=False)
    draws[label] = [
        compute_auc_from_stats(h0, h1, sampling=True) for _ in range(250)
    ]
    low, high = ci(draws[label], alpha=0.05)
    rows.append(("LT", label, int(r["added_tokens"]), auc, low, high))
    print(f"{label} (+{r['added_tokens']} tokens) | "
          f"B3IT {float(r['roc_auc']):.4f} | LT {auc:.4f} "
          f"[{low:.4f}, {high:.4f}]", flush=True)

(out / "lt_bootstrap_draws.json").write_text(json.dumps(draws))
with (out / "b3it_vs_lt.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "variant", "added_tokens", "roc_auc", "ci_low", "ci_high"])
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
for method, color, marker in [("B3IT", "#205781", "o"), ("LT", "#b45309", "s")]:
    points = [r for r in rows if r[0] == method]
    x = [r[2] for r in points]
    ax.plot(x, [r[3] for r in points], marker=marker, color=color, label=method)
    for _, _, length, _, low, high in points:
        ax.vlines(length, low, high, color=color)
        ax.hlines([low, high], length - 0.07, length + 0.07, color=color)

ax.set_xticks([3, 4, 5, 8], ["A (+3)", "B (+4)", "C (+5)", "D (+8)"])
ax.set(xlabel="Safety-prompt variant (added Qwen tokens)", ylabel="Detection ROC-AUC")
ax.set_ylim(min(r[4] for r in rows) - 0.005, 1.005)
ax.set_title("Detecting safety-prompt changes in Qwen2.5-0.5B-Instruct\n"
             "Points show ROC-AUC; bars show 95% bootstrap intervals")
ax.grid(axis="y", alpha=0.25)
ax.legend()
for ext in ("png", "pdf"):
    fig.savefig(out / f"b3it_vs_lt.{ext}", dpi=300)
plt.close(fig)
print("Saved:", out)
print("LT STEP 3 COMPLETE", flush=True)
