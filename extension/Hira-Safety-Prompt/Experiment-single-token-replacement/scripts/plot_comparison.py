
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
    out = root / "results/comparison"
    out.mkdir(exist_ok=True)

    with (root / "results/scoring/b3it_auc_with_ci.csv").open() as f:
        b3it = {r["variant"]: r for r in csv.DictReader(f)}

    rng = np.random.default_rng(2026)
    rows = []
    bootstraps = {}

    for path in sorted(
        (root / "results/lt/merged/analysis/lt").glob("*.json")
    ):
        record = json.loads(path.read_text())
        if record["params"] != {"detection_samples_per_prompt": 10}:
            continue

        metadata = json.loads(record["variant"])
        label = metadata["label"]
        h0 = np.asarray(record["h0_stats"]["stats"])
        h1 = np.asarray(record["h1_stats"]["stats"])
        assert len(h0) == 2000 and len(h1) == 1000

        targets = np.r_[np.zeros(2000), np.ones(1000)]
        auc = roc_auc_score(targets, np.r_[h0, h1])
        values = np.empty(1000)

        for i in range(1000):
            sampled = np.r_[
                rng.choice(h0[:1000], 1000, replace=True),
                rng.choice(h0[1000:], 1000, replace=True),
                rng.choice(h1, 1000, replace=True),
            ]
            values[i] = roc_auc_score(targets, sampled)

        low, high = np.quantile(values, [0.025, 0.975])
        bootstraps[label] = values
        rows.append({
            "variant": label,
            "replacement": metadata["replacement"],
            "b3it_auc": float(b3it[label]["roc_auc"]),
            "b3it_lower": float(b3it[label]["ci_lower"]),
            "b3it_upper": float(b3it[label]["ci_upper"]),
            "lt_auc": float(auc),
            "lt_lower": float(low),
            "lt_upper": float(high),
        })

    rows.sort(key=lambda r: r["variant"])
    assert [r["variant"] for r in rows] == list("ABCD")

    with (out / "b3it_vs_lt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(out / "lt_bootstrap_values.npz", **bootstraps)
    (out / "bootstrap_settings.json").write_text(json.dumps({
        "replicates": 1000,
        "seed": 2026,
        "interval": "95% percentile",
        "scope": "resampling saved scores, conditional on collected data",
    }, indent=2))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)

    for method, color, marker in (
        ("b3it", "#245a81", "o"),
        ("lt", "#bd5700", "s"),
    ):
        y = np.array([r[f"{method}_auc"] for r in rows])
        low = np.array([r[f"{method}_lower"] for r in rows])
        high = np.array([r[f"{method}_upper"] for r in rows])
        ax.errorbar(
            x, y,
            yerr=np.vstack([
                np.maximum(y - low, 0),
                np.maximum(high - y, 0),
            ]),
            fmt=marker + "-",
            color=color, capsize=4,
            label="B3IT" if method == "b3it" else "LT",
        )

    ax.set_xticks(
        x, [f'{r["variant"]}\n{r["replacement"]}' for r in rows]
    )
    ax.set_ylim(0.98, 1.002)
    ax.set_ylabel("Detection ROC-AUC")
    ax.set_xlabel('Replacement for “helpful” — one token replaced')
    ax.set_title("Qwen2.5-0.5B-Instruct — detecting system-prompt edits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")

    caption = (
        "Per detection test: B3IT used 15 generated tokens (5 probes × 3); "
        "LT used 10 (1 prompt × 10)\n"
        "with access to top-20 log probabilities. Both used 50 reference observations per probe.\n"
        "B3IT collection: RTX A6000; LT collection: A40. "
        "1,000 tests per comparison; 95% bootstrap intervals.\n"
        "Vertical axis is zoomed. Intervals describe resampling uncertainty "
        "conditional on the collected data."
    )
    fig.text(0.07, 0.025, caption, fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.21, 1, 1))

    for extension in ("png", "pdf"):
        fig.savefig(
            out / f"b3it_vs_lt.{extension}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)

    for row in rows:
        print(
            f'{row["variant"]}: {row["replacement"]} | '
            f'B3IT {row["b3it_auc"]:.8f} | LT {row["lt_auc"]:.8f}'
        )
    print("Saved PNG, PDF and CSV:", out)
    print("STEP 21 COMPLETE")


if __name__ == "__main__":
    main()
