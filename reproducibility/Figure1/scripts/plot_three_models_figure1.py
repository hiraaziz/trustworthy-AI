
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

out = Path(sys.argv[1])
source = out / "figure1_three_models_summary_with_ci.csv"
with source.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

prompts = [1, 3, 5, 10]
samples = [1, 3, 10, 20, 50]
points = {}
for row in rows:
    key = int(row["n_prompts"]), int(row["samples_per_prompt"])
    assert key not in points, f"Duplicate setting: {key}"
    assert int(row["number_of_model_variant_pairs"]) == 12
    assert int(row["n_bootstrap"]) == 10000
    cost = float(row["yearly_cost_usd"])
    auc = float(row["mean_roc_auc"])
    lo, hi = float(row["auc_lower"]), float(row["auc_upper"])
    assert math.isfinite(cost) and cost > 0 and 0 <= auc <= 1
    assert 0 <= lo <= hi <= 1
    points[key] = cost, auc

assert set(points) == {(p, s) for p in prompts for s in samples}

title = "Scaled reproduction of Figure 1: three models, four perturbation variants per model."
colors = {p: plt.get_cmap("turbo")(i / 3) for i, p in enumerate(prompts)}
diameters = {s: 8 + 2 * i for i, s in enumerate(samples)}

fig = plt.figure(figsize=(10.5, 5.3), layout="constrained")
grid = fig.add_gridspec(1, 2, width_ratios=[4, 1.25])
ax = fig.add_subplot(grid[0, 0])
legend_ax = fig.add_subplot(grid[0, 1])
legend_ax.axis("off")
fig.suptitle(title.replace(": ", ":\n", 1), fontsize=14)

for p in prompts:
    ordered = sorted((points[p, s][0], points[p, s][1], s) for s in samples)
    costs, aucs, sample_counts = zip(*ordered)
    ax.plot(costs, aucs, color=colors[p], linewidth=1.8, zorder=2)
    ax.scatter(costs, aucs, s=[diameters[s] ** 2 for s in sample_counts],
               color=colors[p], linewidths=0, zorder=3)

ax.set_xscale("log")
ax.set_xlim(0.1, 100)
ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
ax.set_xlabel("Cost per year ($)")
ax.set_ylabel("Average ROC AUC")
ax.margins(y=0.12)
ax.grid(True, which="major", alpha=0.22)

color_handles = [
    Line2D([], [], color=colors[p], marker="o", markersize=5, label=str(p))
    for p in prompts
]
size_handles = [
    Line2D([], [], linestyle="none", marker="o", markersize=diameters[s],
           markerfacecolor="none", markeredgecolor="black", label=str(s))
    for s in samples
]
first = legend_ax.legend(handles=color_handles, title="Prompts per test",
                         loc="upper left", frameon=False)
legend_ax.add_artist(first)
legend_ax.legend(handles=size_handles, title="Detection samples\nper prompt",
                 loc="lower left", frameon=False, labelspacing=0.9)

for ext in ("png", "pdf"):
    path = out / f"figure1_three_models.{ext}"
    tmp = path.with_name(f".{path.stem}.tmp.{ext}")
    fig.savefig(tmp, format=ext, dpi=300, bbox_inches="tight", facecolor="white")
    with tmp.open("rb") as f:
        signature = f.read(8)
    assert signature.startswith(b"%PDF-" if ext == "pdf" else b"\x89PNG\r\n\x1a\n")
    tmp.replace(path)
    print(f"{ext.upper()}: {path}")

plt.close(fig)
print("Combined per-variant data:", out / "figure1_three_models_per_variant.csv")
print("Combined point estimates:", out / "figure1_three_models_summary.csv")
print("Combined estimates and intervals:", source)
print("Step 45 passed: combined Figure 1 saved as PNG and PDF.")
