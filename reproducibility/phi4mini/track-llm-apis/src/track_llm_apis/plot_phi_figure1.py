
import csv
import inspect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from track_llm_apis.sampling.common import DataSource, load_single_variant_results
from track_llm_apis.sampling.analyze_param_sweeps import compute_auc_from_stats
from track_llm_apis.util import compute_yearly_cost

ROOT = Path("/workspace/B3IT_Figure1_Phi4Mini")

def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    output = ROOT / "results/figure1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "configs/phi_figure1_evaluation_manifest.json").read_text())
    expected_variants = {
        json.dumps(v["description"], sort_keys=True)
        for v in manifest["variants"] if v["id"] != "unchanged"
    }
    results = load_single_variant_results(
        [ROOT / "results/figure1_merged"], DataSource.BI
    )
    assert len(results) == 80

    print("Repository cost calculation used by Qwen and Phi:")
    print(inspect.getsource(compute_yearly_cost))

    grouped = defaultdict(list)
    records = []
    for result in results:
        p = int(result.params["n_prompts"])
        s = int(result.params["detection_samples"])
        variant = json.dumps(json.loads(result.variant), sort_keys=True)
        assert variant in expected_variants
        assert result.model_name == manifest["model"]

        auc = float(compute_auc_from_stats(
            result.h0_stats.stats, result.h1_stats.stats, sampling=False,
        ))
        # The saved averages already total all prompts in one test.
        # Use Qwen's exact cost call, without initialization/reference costs.
        cost = float(compute_yearly_cost(
            input_tokens_per_sample=result.h1_stats.input_token_avg / s,
            output_tokens_per_sample=result.h1_stats.output_token_avg / s,
            n_samples=s,
        ))
        assert np.isfinite(auc) and 0 <= auc <= 1
        assert np.isfinite(cost) and cost > 0

        row = dict(
            model=result.model_name, variant=variant, n_prompts=p,
            samples_per_prompt=s, roc_auc=auc, yearly_cost_usd=cost,
        )
        records.append(row)
        grouped[p, s].append(row)

    expected_settings = {(p, s) for p in [1, 3, 5, 10]
                         for s in [1, 3, 10, 20, 50]}
    assert set(grouped) == expected_settings
    summary = []
    for (p, s), rows in sorted(grouped.items()):
        assert len(rows) == 4
        assert {r["variant"] for r in rows} == expected_variants
        assert np.allclose([r["yearly_cost_usd"] for r in rows],
                           rows[0]["yearly_cost_usd"])
        summary.append(dict(
            n_prompts=p, samples_per_prompt=s,
            mean_roc_auc=float(np.mean([r["roc_auc"] for r in rows])),
            yearly_cost_usd=rows[0]["yearly_cost_usd"], number_of_variants=4,
        ))

    save_csv(output / "figure1_per_variant.csv",
             sorted(records, key=lambda r: (r["variant"], r["n_prompts"],
                                           r["samples_per_prompt"])))
    save_csv(output / "figure1_summary.csv", summary)

    colors = {1: "#f4cf65", 3: "#ffa52f", 5: "#ed6818", 10: "#923510"}
    sizes = {1: 20, 3: 35, 10: 55, 20: 80, 50: 115}
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(right=0.74, bottom=0.19)
    for p, color in colors.items():
        rows = sorted([r for r in summary if r["n_prompts"] == p],
                      key=lambda r: r["samples_per_prompt"])
        x = [r["yearly_cost_usd"] for r in rows]
        y = [r["mean_roc_auc"] for r in rows]
        ax.plot(x, y, color=color, linewidth=1.5)
        ax.scatter(x, y, s=[sizes[r["samples_per_prompt"]] for r in rows],
                   color=color, zorder=3)

    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    costs = [r["yearly_cost_usd"] for r in summary]
    ax.set_xlim(min(0.07, min(costs) * 0.8), max(120, max(costs) * 1.1))
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Cost per year ($)")
    ax.set_ylabel("Average ROC AUC")
    ax.set_title("B3IT — Phi-4-mini-instruct\nFour perturbation variants")
    ax.grid(True, which="major", alpha=0.25)

    legend = ax.legend(
        handles=[Line2D([0], [0], color=c, marker="o", label=str(p))
                 for p, c in colors.items()],
        title="Prompts per test", loc="upper left",
        bbox_to_anchor=(1.02, 1), frameon=False,
    )
    ax.add_artist(legend)
    ax.legend(
        handles=[Line2D([0], [0], linestyle="", marker="o",
                        markerfacecolor="none", markeredgecolor="black",
                        markersize=np.sqrt(size), label=str(s))
                 for s, size in sizes.items()],
        title="Samples per prompt", loc="upper left",
        bbox_to_anchor=(1.02, 0.55), frameon=False,
    )
    fig.text(
        0.12, 0.035,
        "Mean across LoRA, Gaussian noise, magnitude pruning and random pruning.\n"
        "50 reference samples per prompt; hourly monitoring; initialization cost excluded.",
        fontsize=9,
    )
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"figure1_phi.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nPrompts | Samples | Mean ROC AUC | Annual cost")
    for r in summary:
        print(f'{r["n_prompts"]:7d} | {r["samples_per_prompt"]:7d} | '
              f'{r["mean_roc_auc"]:.4f} | ${r["yearly_cost_usd"]:.4f}')
    print("\nPhi figure and CSV files saved:", output)

if __name__ == "__main__":
    main()
