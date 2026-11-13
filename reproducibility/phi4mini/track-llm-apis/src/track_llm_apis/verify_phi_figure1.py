
import gzip
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from track_llm_apis.sampling.common import CompressedOutput, DataSource

ROOT = Path("/workspace/B3IT_Figure1_Phi4Mini")
raw_manifest = (ROOT / "configs/phi_figure1_evaluation_manifest.json").read_bytes()
manifest = json.loads(raw_manifest)
prompts = set(manifest["prompts"])
merged = CompressedOutput(model_name=manifest["model"])
common_provenance = None
summary = {}

for variant in manifest["variants"]:
    variant_id = variant["id"]
    totals = Counter()
    signatures = set()

    for start in range(0, 5000, 100):
        folder = ROOT / "results/observations" / variant_id / f"{start:05d}_{start+100:05d}"
        metadata = json.loads((folder / "COMPLETE.json").read_text())
        assert metadata["variant"] == variant_id
        assert metadata["start"] == start
        assert metadata["samples_per_prompt"] == 100
        assert metadata["prompt_count"] == 120
        provenance = metadata["provenance"]
        assert provenance["manifest_sha256"] == hashlib.sha256(raw_manifest).hexdigest()
        assert provenance["variant"] == variant_id
        expected_signature = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()
        assert metadata["signature"] == expected_signature
        signatures.add(expected_signature)

        common = {k: v for k, v in provenance.items() if k != "variant"}
        if common_provenance is None:
            common_provenance = common
        assert common == common_provenance, "Mixed collection environments or code."

        files = list(folder.glob("*.json.gz"))
        assert len(files) == 1
        raw = files[0].read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata["output_sha256"]
        chunk = CompressedOutput.from_json(json.loads(gzip.decompress(raw)))
        assert chunk.model_name == manifest["model"]
        assert len(chunk.rows) == 12000

        counts = Counter()
        for row in chunk.rows:
            assert row.source == DataSource.BI.value
            assert json.loads(row.variant) == variant["description"]
            assert row.text[1] == 1
            prompt = row.prompt[0]
            assert row.prompt[1] == manifest["prompt_lengths"][prompt]
            counts[prompt] += 1
        assert set(counts) == prompts
        assert all(n == 100 for n in counts.values())
        totals.update(counts)
        merged.merge(chunk)

    assert len(signatures) == 1
    assert set(totals) == prompts and all(n == 5000 for n in totals.values())
    summary[variant_id] = dict(prompts=120, samples_per_prompt=5000,
                               observations=sum(totals.values()))
    print(f"Verified {variant_id}: 600,000 observations.", flush=True)

assert len(merged.rows) == 3_000_000
destination = ROOT / "results/figure1_merged"
if destination.exists():
    raise RuntimeError(f"Merged directory already exists: {destination}. Report this.")

with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
    staged = Path(tmp) / "merged"
    merged.dump_json(staged)
    files = list(staged.glob("*.json.gz"))
    assert len(files) == 1 and files[0].stat().st_size > 0
    merged_hash = hashlib.sha256(files[0].read_bytes()).hexdigest()
    staged.rename(destination)

report = dict(
    model=manifest["model"], variants=summary, total_observations=len(merged.rows),
    collection_provenance=common_provenance, merged_sha256=merged_hash,
)
path = ROOT / "results/collection_verification.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2))
temporary.replace(path)
print("Verification complete: 3,000,000 observations.", flush=True)
print("Merged dataset:", destination, flush=True)
