
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    root = Path(__file__).resolve().parents[1]
    score_dir = root / "results/detection/merged/analysis/bi"
    out = root / "results/scoring"
    figures = root / "results/figures"
    figures.mkdir(exist_ok=True)

    rng = np.random.default_rng(2026)
    n_bootstraps = 1000
    results = {}
    bootstrap_values = {}

    for path in sorted(score_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record["params"] != {"n_prompts": 5, "detection_samples": 3}:
            continue

        variant = json.loads(record["variant"])
        label = variant["label"]
        h0 = np.asarray(record["h0_stats"]["stats"])
        h1 = np.asarray(record["h1_stats"]["stats"])
        assert len(h0) == 2000 and len(h1) == 1000
        targets = np.r_[np.zeros(len(h0)), np.ones(len(h1))]

        auc = roc_auc_score(targets, np.r_[h0, h1])
        boot = np.empty(n_bootstraps)

        # Preserve the scorer's two equally sized no-change groups.
        for i in range(n_bootstraps):
            sampled_h0 = np.r_[
                rng.choice(h0[:1000], size=1000, replace=True),
                rng.choice(h0[1000:], size=1000, replace=True),
            ]
            sampled_h1 = rng.choice(h1, size=1000, replace=True)
            boot[i] = roc_auc_score(
                targets, np.r_[sampled_h0, sampled_h1]
            )

        low, high = np.quantile(boot, [0.025, 0.975])
        results[label] = {
            "variant": label,
            "replacement": variant["replacement"],
            "tokens_replaced": 1,
            "roc_auc": float(auc),
            "ci_lower": float(low),
            "ci_upper": float(high),
        }
        bootstrap_values[label] = boot

    assert set(results) == set("ABCD")
    rows = [results[label] for label in "ABCD"]

    with (out / "b3it_auc_with_ci.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out / "b3it_bootstrap_values.npz", **bootstrap_values
    )
    (out / "bootstrap_settings.json").write_text(json.dumps({
        "replicates": n_bootstraps,
        "seed": 2026,
        "interval": "95% percentile",
        "resampling": "within each saved H0 group and H1",
        "scope": "conditional on collected observations and selected probes",
    }, indent=2))

    x = np.arange(4)
    aucs = np.array([r["roc_auc"] for r in rows])
    lower = np.array([r["ci_lower"] for r in rows])
    upper = np.array([r["ci_upper"] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        x, aucs,
        yerr=np.vstack([
            np.maximum(aucs - lower, 0),
            np.maximum(upper - aucs, 0),
        ]),
        fmt="o", capsize=5, color="#245a81",
        label="B3IT",
    )
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xticks(x, [f'{r["variant"]}\n{r["replacement"]}' for r in rows])
    ax.set_ylim(0.48, 1.025)
    ax.set_xlabel('Replacement for “helpful” — one token replaced per variant')
    ax.set_ylabel("Detection ROC-AUC")
    ax.set_title("Qwen2.5-0.5B-Instruct — system-prompt token edits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    caption = (
        "Baseline system prompt: “You are a helpful assistant.”\n"
        "Per detection test: 5 probes × 3 generated tokens; "
        "50 reference observations per probe.\n"
        "1,000 tests per comparison; error bars show 95% bootstrap intervals "
        "from 1,000 resamples."
    )
    fig.text(0.08, 0.025, caption, fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.19, 1, 1))

    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"b3it_token_edits.{extension}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)

    for row in rows:
        print(
            f'{row["variant"]}: {row["replacement"]} | '
            f'AUC {row["roc_auc"]:.8f} | '
            f'95% interval [{row["ci_lower"]:.8f}, {row["ci_upper"]:.8f}]'
        )

    print("Figures:", figures)
    print("STEP 16 COMPLETE")


if __name__ == "__main__":
    main()
