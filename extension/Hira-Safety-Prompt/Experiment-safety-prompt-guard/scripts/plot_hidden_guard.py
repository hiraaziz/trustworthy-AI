import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "results/scoring/hidden_guard_b3it.json").read_text())
raw = json.loads((root / summary["raw_scores"]).read_text())
h0 = raw["h0_stats"]["stats"]
h1 = raw["h1_stats"]["stats"]
assert len(h0) == 2000 and len(h1) == 1000

fpr, tpr, _ = roc_curve([0] * len(h0) + [1] * len(h1), h0 + h1)
area = auc(fpr, tpr)
assert math.isclose(area, summary["roc_auc"], abs_tol=1e-10)

out = root / "results/figures"
out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.3, 5.2), layout="constrained")
ax.plot(fpr, tpr, color="#205c85", linewidth=2.4, label=f"B3IT  (AUC {area:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Chance")
ax.set(
    title="Detecting a hidden input guard on Qwen2.5-0.5B",
    xlabel="False positive rate",
    ylabel="True positive rate",
    xlim=(0, 1), ylim=(0, 1.01),
)
ax.grid(alpha=0.18)
ax.legend(loc="lower right")
for ext in ("png", "pdf"):
    fig.savefig(out / f"hidden_guard_roc.{ext}", dpi=200)
plt.close(fig)
print(f"Verified ROC-AUC: {area:.8f}")
print("Figures saved in:", out)
