
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from statistics import fmean

from track_llm_apis.config import config
from track_llm_apis.sampling import analyze_param_sweeps as sweep

root = Path(sys.argv[1])
out = root / "results/figure1_combined"
cache = out / "bootstrap"
cache.mkdir(exist_ok=True)

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def key(row):
    return int(row["n_prompts"]), int(row["samples_per_prompt"])

def pair_id(model, variant):
    return model, json.dumps(json.loads(variant), sort_keys=True, separators=(",", ":"))

# Linux fork workers inherit the current setting's score arrays read-only.
# Each task passes only replicate indices between processes.
def bootstrap_batch(bounds):
    start, stop = bounds
    values = []
    for index in range(start, stop):
        random.seed(seed_base + index)
        values.append(float(sweep._bootstrap_one_hierarchical(score_pairs)))
    return values

if __name__ == "__main__":
    summaries = read_csv(out / "figure1_three_models_summary.csv")
    pairs = read_csv(out / "figure1_three_models_per_variant.csv")
    grid = {(p, s) for p in (1, 3, 5, 10) for s in (1, 3, 10, 20, 50)}
    assert len(summaries) == 20 and {key(r) for r in summaries} == grid
    assert len(pairs) == 240

    expected = {k: {} for k in grid}
    for row in pairs:
        k = key(row)
        pair = pair_id(row["model"], row["variant"])
        assert pair not in expected[k]
        expected[k][pair] = float(row["roc_auc"])

    directories = [
        root / "results" / folder / "figure1_merged"
        for folder in ("qwen", "gemma", "phi4mini")
    ]
    raw = sweep.load_single_variant_results(directories, sweep.BIParameterSweep.get_data_source())
    grouped = {k: {} for k in grid}
    for result in raw:
        k = (result.params["n_prompts"], result.params["detection_samples"])
        pair = pair_id(result.model_name, result.variant)
        assert pair in expected[k] and pair not in grouped[k], (k, pair)
        h0, h1 = result.h0_stats.stats, result.h1_stats.stats
        assert (len(h0), len(h1)) == (2000, 1000), (k, pair)
        auc = sweep.compute_auc_from_stats(h0, h1, sampling=False)
        assert math.isclose(auc, expected[k][pair], abs_tol=1e-9), (k, pair)
        grouped[k][pair] = (h0, h1)

    n = config.analysis.n_bootstrap
    alpha = config.analysis.results_alpha
    assert n == 10000 and alpha == 0.05
    workers = min(9, len(os.sched_getaffinity(0)))
    print(f"Verified score files. Using {workers} CPU workers and {n:,} replicates per setting.",
          flush=True)

    completed = []
    new_times = []
    for setting_index, row in enumerate(summaries):
        p, s = k = key(row)
        assert len(grouped[k]) == 12 and grouped[k].keys() == expected[k].keys()
        assert math.isclose(fmean(expected[k].values()), float(row["mean_roc_auc"]),
                            abs_tol=1e-9)
        ordered_pairs = sorted(grouped[k])
        score_pairs = [grouped[k][pair] for pair in ordered_pairs]
        seed_base = (p * 100 + s) * n
        fingerprint = hashlib.sha256(
            json.dumps([ordered_pairs, score_pairs, seed_base, n, alpha],
                       separators=(",", ":")).encode()
        ).hexdigest()
        path = cache / f"prompts_{p}_samples_{s}.json"

        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["fingerprint"] == fingerprint, f"Saved inputs differ: {path}"
            values = saved["bootstrap_aucs"]
            status = "reused"
        else:
            started = time.monotonic()
            batches = [(i, min(i + 100, n)) for i in range(0, n, 100)]
            with mp.get_context("fork").Pool(workers) as pool:
                values = [v for batch in pool.imap(bootstrap_batch, batches) for v in batch]
            assert len(values) == n and all(0 <= v <= 1 for v in values)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fingerprint": fingerprint, "bootstrap_aucs": values}))
            tmp.replace(path)
            new_times.append(time.monotonic() - started)
            status = "saved"

        assert len(values) == n and all(0 <= v <= 1 for v in values)
        lower, upper = sweep.ci(values, alpha)
        completed.append(dict(row, auc_lower=lower, auc_upper=upper, n_bootstrap=n))
        remaining = len(summaries) - setting_index - 1
        eta = f" | rough remaining {remaining * fmean(new_times) / 60:.1f} min" if new_times else ""
        print(f"{setting_index + 1:2d}/20 | {p} prompts × {s} samples | "
              f"95% interval [{lower:.6f}, {upper:.6f}] | {status}{eta}", flush=True)

    destination = out / "figure1_three_models_summary_with_ci.csv"
    tmp = destination.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    tmp.replace(destination)
    assert len(read_csv(destination)) == 20
    print("Saved:", destination)
    print("Step 44 passed: 20 bootstrap intervals saved using 10,000 replicates each.")
