
from pathlib import Path
from collections import defaultdict
import json
import sys

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

root = Path(sys.argv[1])
model = "microsoft/Phi-4-mini-instruct"
manifest = json.loads((root / "audit/hub_revisions.json").read_text())
revision = manifest["models"][model]["revision"]
assert revision == "cfbefacb99257ffa30c83adab238a50856ac3083", revision

snapshot = Path(snapshot_download(
    repo_id=model,
    revision=revision,
    cache_dir=str(root / "cache/huggingface/hub"),
    allow_patterns=[
        "*.json", "*.safetensors", "*.txt", "*.model",
        "*.jinja", "*.tiktoken", "*.py",
    ],
))
assert snapshot.name == revision

index = snapshot / "model.safetensors.index.json"
expected = defaultdict(set)
if index.is_file():
    for tensor, shard in json.loads(index.read_text())["weight_map"].items():
        expected[shard].add(tensor)
    assert expected, "Empty checkpoint index"
    shard_names = sorted(expected)
else:
    shard_names = ["model.safetensors"]

sizes = {}
tensor_count = 0
for name in shard_names:
    path = snapshot / name
    assert path.is_file(), f"Missing weight shard: {name}"
    with safe_open(str(path), framework="pt", device="cpu") as shard:
        keys = set(shard.keys())
    assert keys, f"Empty weight shard: {name}"
    if index.is_file():
        assert keys == expected[name], f"Weight names differ from index: {name}"
    tensor_count += len(keys)
    sizes[name] = path.stat().st_size

config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
assert tokenizer.encode("Hello", add_special_tokens=False)

record = {
    "model": model,
    "revision": revision,
    "snapshot": str(snapshot),
    "weight_files": sizes,
    "checkpoint_tensor_count": tensor_count,
    "model_type": config.model_type,
    "architectures": config.architectures,
    "tokenizer_class": type(tokenizer).__name__,
}
report = root / "audit/phi4mini_snapshot.json"
if report.exists():
    assert json.loads(report.read_text()) == record, f"Existing audit record differs: {report}"
else:
    tmp = report.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(report)

print("Revision:", revision)
print("Architecture:", config.architectures)
print("Tokenizer:", type(tokenizer).__name__)
print(f"Weights: {sum(sizes.values()) / 2**30:.2f} GiB across {len(sizes)} shard(s)")
print("Audit:", report)
print("Step 30 passed: pinned Phi-4-mini checkpoint and local tokenizer verified.")
