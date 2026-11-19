import os
import subprocess
import time
from pathlib import Path

root = Path("/workspace/B3IT_Figure1_Corrected")
script = root / "scripts/export_figure1_csvs.py"

script.write_text(r'''
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

from track_llm_apis.sampling.analyze_param_sweeps import (
    BIParameterSweep, compute_auc_from_stats,
)
from track_llm_apis.sampling.common import DataSource, load_single_variant_results
from track_llm_apis.tinychange import TinyChangeModel


def write_csv(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with tmp.open(newline="", encoding="utf-8") as f:
        saved = list(csv.DictReader(f))
    assert saved == [{k: str(v) for k, v in row.items()} for row in rows]
    tmp.replace(path)


def main():
    folder = Path(sys.argv[1])
    manifest = json.loads((folder / "evaluation_manifest.json").read_text())
    model = manifest["model"]
    variants = [
        TinyChangeModel.description_to_str(c["description"])
        for c in manifest["conditions"] if c["id"] != "unchanged"
    ]
    grid = list(product([1, 3, 5, 10], [1, 3, 10, 20, 50]))
    assert len(set(variants)) == 4
    assert manifest["reference_samples_per_prompt"] == 50

    results = load_single_variant_results([folder / "figure1_merged"], DataSource.BI)
    indexed = {}
    for r in results:
        assert r.model_name == model and r.method == DataSource.BI
        key = (r.params["n_prompts"], r.params["detection_samples"], r.variant)
        assert key not in indexed, f"Duplicate comparison: {key}"
        assert r.n_tests == 1000
        assert len(r.h0_stats.stats) == 2000 and len(r.h1_stats.stats) == 1000
        indexed[key] = r
    assert set(indexed) == {(p, s, v) for p, s in grid for v in variants}

    per_variant, summary = [], []
    for p, s in grid:
        setting_rows = []
        input_tokens, output_tokens = [], []
        for v in variants:
            r = indexed[p, s, v]
            auc = float(compute_auc_from_stats(r.h0_stats.stats, r.h1_stats.stats, sampling=False))
            assert math.isfinite(auc) and 0 <= auc <= 1
            inp, out = r.h0_stats.input_token_avg, r.h0_stats.output_token_avg
            point = BIParameterSweep.create_result((p, s), inp, out)
            row = {
                "model": model, "variant": v, "n_prompts": p,
                "samples_per_prompt": s, "roc_auc": auc,
                "yearly_cost_usd": BIParameterSweep._compute_yearly_cost(point),
            }
            setting_rows.append(row)
            input_tokens.append(inp)
            output_tokens.append(out)

        per_variant.extend(setting_rows)
        # Upstream averages token counts across pairs before calculating cost.
        mean_point = BIParameterSweep.create_result(
            (p, s), sum(input_tokens) / 4, sum(output_tokens) / 4,
        )
        summary.append({
            "n_prompts": p, "samples_per_prompt": s,
            "mean_roc_auc": sum(row["roc_auc"] for row in setting_rows) / 4,
            "yearly_cost_usd": BIParameterSweep._compute_yearly_cost(mean_point),
            "number_of_variants": 4,
        })

    output = folder / "figure1"
    output.mkdir(exist_ok=True)
    for filename, rows in [
        ("figure1_per_variant.csv", per_variant),
        ("figure1_summary.csv", summary),
    ]:
        path = output / filename
        write_csv(path, rows)
        print(f"{len(rows)} rows saved and reread: {path}")

    for row in (summary[0], summary[-1]):
        print(f'{row["n_prompts"]} prompt(s) × {row["samples_per_prompt"]} detection '
              f'sample(s): mean AUC {row["mean_roc_auc"]:.6f}, '
              f'annual cost ${row["yearly_cost_usd"]:.4f}')
    print("step 42 passed: 80 per-variant rows and 20 four-variant means verified.")


if __name__ == "__main__":
    main()
''', encoding="utf-8")

started = time.monotonic()
subprocess.run(
    ["/opt/b3it-corrected-venv/bin/python", "-u", str(script), str(root / "results/phi4mini")],
    cwd=root / "authors_repo/track-llm-apis",
    env={
        **os.environ,
        "TRACKLLM__DATA_DIR": str(root / "data"),
        "CUDA_VISIBLE_DEVICES": "",
        "MPLBACKEND": "Agg",
        "HF_HUB_OFFLINE": "1",
    },
    check=True,
)
print(f"Elapsed: {(time.monotonic() - started) / 60:.1f} minutes")