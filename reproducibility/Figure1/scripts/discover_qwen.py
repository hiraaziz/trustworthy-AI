
import json
import sys
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.sampling import bi_phase_1 as phase
from track_llm_apis.util import slugify

def main():
    root = Path(sys.argv[1])
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    snapshot_record = json.loads((root / "audit/qwen_snapshot.json").read_text())
    traffic_record = json.loads((root / "audit/lmsys_preparation.json").read_text())
    snapshot = Path(snapshot_record["snapshot"])
    assert snapshot_record["model"] == model
    assert snapshot.is_dir() and snapshot.name == snapshot_record["revision"]

    bi = config.sampling.bi
    assert (bi.phase_1_tokens_per_endpoint, bi.phase_1_queries_per_token) == (20000, 3)
    assert bi.temperature == 0 and bi.models[model] == 1
    assert (bi.batch_size, bi.per_batch_input_fraction) == (64, 0.95)
    assert config.datasets_dir == Path(traffic_record["datasets_dir"])
    traffic_cache = config.datasets_dir / slugify("lmsys/lmsys-chat-1m", hash_length=0)
    assert (traffic_cache / "state.json").is_file(), f"Missing Step 4 cache: {traffic_cache}"

    # Keep the upstream model ID in the results while loading the exact local revision.
    original_init = phase.init_vllm_bi
    class PinnedTokenizer:
        @staticmethod
        def from_pretrained(name):
            assert name == model
            return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)

    def pinned_engine(name, device):
        assert name == model
        return original_init(str(snapshot), device)

    try:
        result = phase.BIPhase1aOutput.load_model(model)
        print("Found saved discovery output. Verifying its counts.", flush=True)
    except FileNotFoundError:
        print("Starting upstream phase_1a on all 20,000 candidates.", flush=True)
        with patch.object(phase, "AutoTokenizer", PinnedTokenizer), \
             patch.object(phase, "init_vllm_bi", pinned_engine):
            phase.phase_1a(device="cuda:0", model=model)
        result = phase.BIPhase1aOutput.load_model(model)

    assert result.model_name == model and result.phase == "1a"
    assert result.bi_config == bi.model_dump(mode="json")
    assert len(result.results) == 20000
    assert all(sum(counts.values()) == 3 for counts in result.results.values())
    assert result.prompt_lengths.keys() == result.results.keys()

    border_inputs = result.get_border_inputs()
    report = {
        "model": model,
        "revision": snapshot_record["revision"],
        "traffic_dataset_fingerprint": traffic_record["dataset_fingerprint"],
        "candidates": len(result.results),
        "candidate_observations": sum(sum(c.values()) for c in result.results.values()),
        "discovered_border_inputs": len(border_inputs),
        "upstream_result": str(result._get_path()),
    }
    audit = root / "audit/step06_qwen_discovery.json"
    tmp = audit.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(audit)
    print(json.dumps(report, indent=2), flush=True)
    print("Step 6 passed: 20,000 Qwen candidates and 60,000 target observations verified.")


if __name__ == "__main__":
    main()
