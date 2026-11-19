
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from track_llm_apis.tinychange import TinyChangeConfig

root = Path(sys.argv[1])
record = json.loads((root / "audit/phi4mini_snapshot.json").read_text())
snapshot = Path(record["snapshot"])
assert record["model"] == "microsoft/Phi-4-mini-instruct"
assert snapshot.is_dir() and snapshot.name == record["revision"]

cfg = TinyChangeConfig()
model_cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) == (8, 8, 0.0)
assert set(cfg.lora_target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

# Build the real module hierarchy with shape-only tensors for this inspection.
with torch.device("meta"):
    base = AutoModelForCausalLM.from_config(model_cfg, dtype=torch.bfloat16)

fused = [name for name, _ in base.named_modules() if name.endswith(".qkv_proj")]

# Same LoraConfig arguments and PEFT entry point as upstream tinychange.py.
adapter = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    target_modules=cfg.lora_target_modules,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, adapter)

matched = [
    name for name, module in model.named_modules()
    if "default" in getattr(module, "lora_A", {})
]
trainable = {
    name: list(param.shape)
    for name, param in model.named_parameters()
    if param.requires_grad
}
assert matched and trainable, "PEFT attached no trainable adapters."
expected = {
    f"{name}.lora_{side}.default.weight"
    for name in matched for side in ("A", "B")
}
assert set(trainable) == expected, "Trainable parameters differ from the adapter pairs."

by_projection = dict(Counter(name.rsplit(".", 1)[-1] for name in matched))
report = {
    "model": record["model"],
    "revision": record["revision"],
    "inspection": "Shape-only model before training",
    "configured_targets": cfg.lora_target_modules,
    "rank": cfg.lora_r,
    "alpha": cfg.lora_alpha,
    "dropout": cfg.lora_dropout,
    "fused_qkv_modules": fused,
    "matched_by_projection": by_projection,
    "matched_module_names": matched,
    "trainable_parameter_shapes": trainable,
    "trainable_parameter_count": sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ),
}
path = root / "audit/step33_phi4mini_lora_targets.json"
if path.exists():
    assert json.loads(path.read_text()) == report, f"Existing audit differs: {path}"
else:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)

print("Configured targets:", cfg.lora_target_modules)
print("Fused qkv_proj modules in Phi:", len(fused))
print("Actual adapter matches:", by_projection)
print("Trainable parameter count:", f"{report['trainable_parameter_count']:,}")
print("Full module and parameter names:", path)
print("Step 33 passed: Phi LoRA targets inspected and saved before training.")
