
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
    model = "google/gemma-3-1b-it"
    snap = json.loads((root / "audit/gemma_snapshot.json").read_text())
    prep = json.loads((root / "audit/lmsys_preparation.json").read_text())
    snapshot = Path(snap["snapshot"])
    assert snap["model"] == model and snapshot.is_dir()
    assert snapshot.name == snap["revision"]

    bi = config.sampling.bi
    assert config.data_dir == root / "data"
    assert (bi.phase_1_tokens_per_endpoint, bi.phase_1_queries_per_token) == (20000, 3)
    assert (bi.temperature, bi.models[model]) == (0, 1)
    assert (bi.batch_size, bi.per_batch_input_fraction) == (64, 0.95)

    cache = config.datasets_dir / slugify(prep["dataset"], hash_length=0)
    state = json.loads((cache / "state.json").read_text())
    assert state["_fingerprint"] == prep["dataset_fingerprint"]

    try:
        result = phase.BIPhase1aOutput.load_model(model)
    except FileNotFoundError:
        result = None

    if result is None:
        # Keep the upstream model ID in saved results and load Step 18's exact revision.
        original_init = phase.init_vllm_bi

        class PinnedTokenizer:
            @staticmethod
            def from_pretrained(name):
                assert name == model
                return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)

        def pinned_engine(name, device):
            assert name == model
            return original_init(str(snapshot), device)

        print("Starting upstream Gemma discovery on all 20,000 candidates.", flush=True)
        with (
            patch.object(phase, "AutoTokenizer", PinnedTokenizer),
            patch.object(phase, "init_vllm_bi", pinned_engine),
        ):
            phase.phase_1a(device="cuda:0", model=model)
        result = phase.BIPhase1aOutput.load_model(model)
    else:
        print("Found saved Gemma discovery. Verifying every candidate.", flush=True)

    assert result.model_name == model and result.phase == "1a"
    assert result.bi_config == bi.model_dump(mode="json")
    assert len(result.results) == 20000
    assert result.results.keys() == result.prompt_lengths.keys()
    assert all(sum(outputs.values()) == 3 for outputs in result.results.values())

    report = {
        "model": model,
        "revision": snap["revision"],
        "traffic_dataset_fingerprint": prep["dataset_fingerprint"],
        "candidates": len(result.results),
        "candidate_observations": sum(sum(v.values()) for v in result.results.values()),
        "discovered_border_inputs": len(result.get_border_inputs()),
        "upstream_result": str(result._get_path()),
    }
    path = root / "audit/step19_gemma_discovery.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)
    print(json.dumps(report, indent=2))
    print("Step 19 passed: 20,000 Gemma candidates and 60,000 target observations verified.")


if __name__ == "__main__":
    main()
