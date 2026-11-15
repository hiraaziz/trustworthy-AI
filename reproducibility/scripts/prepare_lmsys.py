
import json
from pathlib import Path

from track_llm_apis import util

root = Path("/workspace/B3IT_Figure1_Corrected")
revisions = json.loads((root / "audit/hub_revisions.json").read_text())
dataset_revision = revisions["dataset"]["revision"]
datasets_dir = root / "data/datasets"
report_path = root / "audit/lmsys_preparation.json"

# Preserve the original loader and filters.
# Supply only the explicitly recorded Hugging Face dataset revision.
original_load_dataset = util.load_dataset

def load_pinned_dataset(*args, **kwargs):
    dataset_name = args[0] if args else kwargs.get("path")
    if dataset_name == "lmsys/lmsys-chat-1m":
        assert kwargs.get("revision", dataset_revision) == dataset_revision
        kwargs["revision"] = dataset_revision
    return original_load_dataset(*args, **kwargs)

util.load_dataset = load_pinned_dataset

print("Preparing the full LMSYS dataset with upstream filters...", flush=True)
dataset = util.load_lmsys_chat_1m(
    gpt4_filter=True,
    redacted_filter=True,
    flagged_filter=True,
    first_turn_only=True,
    use_cache=True,
    datasets_dir=datasets_dir,
)

assert len(dataset) >= 128, "Too few filtered conversations."
assert dataset.column_names == ["conversation"]

for index, conversation in enumerate(dataset["conversation"]):
    assert len(conversation) == 2, f"Unexpected turn count at {index}"
    assert conversation[0]["role"] == "user"
    assert conversation[1]["role"] == "assistant"

record = {
    "dataset": "lmsys/lmsys-chat-1m",
    "revision": dataset_revision,
    "loader": "track_llm_apis.util.load_lmsys_chat_1m",
    "filters": {
        "gpt4_filter": True,
        "redacted_filter": True,
        "flagged_filter": True,
        "first_turn_only": True,
    },
    "full_dataset_processed": True,
    "filtered_conversations": len(dataset),
    "dataset_fingerprint": dataset._fingerprint,
    "datasets_dir": str(datasets_dir),
    "training_subset_selected": False,
    "token_length_filter_added": False,
}

if report_path.exists():
    saved = json.loads(report_path.read_text())
    assert saved == record, "Prepared dataset differs from its saved report."
else:
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(report_path)

print(json.dumps(record, indent=2), flush=True)
print("Step 4 passed: full upstream LMSYS preparation completed.", flush=True)
