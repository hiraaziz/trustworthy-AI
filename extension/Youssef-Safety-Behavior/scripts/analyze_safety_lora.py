"""Analyze matched base-vs-LoRA detector runs and create reviewable figures."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_safety_prompt_1a as base

OUT_DIR = Path(os.environ.get("B3IT_LORA_RESULTS", "/kaggle/working/safety-lora-results"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_samples_path = OUT_DIR / "base_border_samples.json"
    lora_samples_path = OUT_DIR / "lora_border_samples.json"
    if not base_samples_path.is_file() or not lora_samples_path.is_file():
        raise FileNotFoundError("Run the notebook's base and LoRA inference cells before analysis.")
    base_samples = json.loads(base_samples_path.read_text(encoding="utf-8"))
    lora_samples = json.loads(lora_samples_path.read_text(encoding="utf-8"))
    detector_rows = base.detection_curves(
        base_samples["neutral_reference"],
        base_samples["neutral_control"],
        lora_samples["lora_treatment"],
    )
    save_rows(OUT_DIR / "detector_metrics.csv", detector_rows)

    eval_rows = read_rows(OUT_DIR / "base_safety_eval.csv") + read_rows(OUT_DIR / "lora_safety_eval.csv")
    save_rows(OUT_DIR / "safety_eval_for_review.csv", eval_rows)
    summaries = []
    for phase in ("base", "lora"):
        for category in ("harmful", "benign"):
            subset = [r for r in eval_rows if r["condition"] == phase and r["category"] == category]
            refusals = sum(str(r["refusal_heuristic"]).lower() == "true" for r in subset)
            summaries.append({
                "condition": phase,
                "category": category,
                "n_questions": len(subset),
                "heuristic_refusal_count": refusals,
                "heuristic_refusal_rate": refusals / len(subset) if subset else float("nan"),
                "human_review_required": True,
            })
    save_rows(OUT_DIR / "heuristic_refusal_summary.csv", summaries)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [r["samples_per_prompt"] for r in detector_rows]
    axes[0].plot(x, [r["b3it_detection_rate"] for r in detector_rows], marker="o", label="B3IT detection")
    axes[0].plot(x, [r["lt_detection_rate"] for r in detector_rows], marker="o", label="LT detection")
    axes[0].plot(x, [r["b3it_false_alarm_rate"] for r in detector_rows], "--", label="B3IT false alarm")
    axes[0].plot(x, [r["lt_false_alarm_rate"] for r in detector_rows], "--", label="LT false alarm")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(x, labels=[str(v) for v in x])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Samples per frozen border prompt")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Detecting the safety LoRA change")
    axes[0].grid(alpha=.25)
    axes[0].legend()

    labels = [f"{r['condition']}\n{r['category']}" for r in summaries]
    axes[1].bar(labels, [r["heuristic_refusal_rate"] for r in summaries], color=["#4063d8", "#12a68a"] * 2)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Refusal heuristic rate")
    axes[1].set_title("Automatic screen only (human review needed)")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    path = OUT_DIR / "safety_lora_results.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Detector metrics: {OUT_DIR / 'detector_metrics.csv'}")
    print(f"Review responses: {OUT_DIR / 'safety_eval_for_review.csv'}")
    print(f"Heuristic refusal summary: {OUT_DIR / 'heuristic_refusal_summary.csv'}")
    print(f"Figure: {path}")
    print("Do not treat refusal keywords as a safety score; manually label harmful compliance and benign over-refusals.")


if __name__ == "__main__":
    main()
