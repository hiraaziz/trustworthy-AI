
import json
import sys
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer
from track_llm_apis.config import config
from track_llm_apis.sampling import bi_phase_1 as phase

def main():
    root = Path(sys.argv[1])
    model = "microsoft/Phi-4-mini-instruct"
    record = json.loads((root / "audit/phi4mini_snapshot.json").read_text())
    snapshot = Path(record["snapshot"])
    assert record["model"] == model
    assert snapshot.name == record["revision"] == "cfbefacb99257ffa30c83adab238a50856ac3083"
    assert config.data_dir.resolve() == (root / "data").resolve()

    bi = config.sampling.bi
    assert (bi.phase_1_tokens_per_endpoint, bi.phase_1_queries_per_token) == (20000, 3)
    assert bi.temperature == 0 and bi.models[model] == 1

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    candidates = phase.get_single_token_inputs(tokenizer, bi.phase_1_tokens_per_endpoint)
    assert len(candidates) == len(set(candidates)) == 20000, len(candidates)

    traffic = phase._load_traffic_prompts()
    assert len(traffic) == 4502, f"Unexpected traffic pool size: {len(traffic)}"

    try:
        result = phase.BIPhase1aOutput.load_model(model)
    except FileNotFoundError:
        result = None

    if result is None:
        original_init = phase.init_vllm_bi

        def init_pinned(model_name, device):
            assert model_name == model
            return original_init(str(snapshot), device)

        print("Starting Phi discovery: 20,000 candidates in one sampler call.", flush=True)
        with patch.object(phase, "init_vllm_bi", init_pinned):
            phase._run_phase(
                output=phase.BIPhase1aOutput(bi_config=bi.model_dump(mode="json")),
                model_inputs={model: candidates},
                model_max_tokens=bi.models,
                n_samples=bi.phase_1_queries_per_token,
                traffic_prompts=traffic,
                device="cuda:0",
            )
        result = phase.BIPhase1aOutput.load_model(model)
    else:
        print("Found saved Phi discovery. Verifying it before reuse.", flush=True)

    assert result.model_name == model and result.phase == "1a"
    assert result.bi_config == bi.model_dump(mode="json"), "Saved discovery settings differ."
    assert set(result.results) == set(candidates), "Saved candidate set differs."
    assert all(sum(counts.values()) == 3 for counts in result.results.values())
    border_inputs = result.get_border_inputs()

    audit = {
        "model": model,
        "revision": record["revision"],
        "candidates": len(candidates),
        "candidate_observations": sum(sum(c.values()) for c in result.results.values()),
        "traffic_prompts": len(traffic),
        "discovered_border_inputs": len(border_inputs),
        "upstream_result": str(result._get_path()),
    }
    path = root / "audit/step31_phi4mini_discovery.json"
    if path.exists():
        assert json.loads(path.read_text()) == audit, f"Existing audit differs: {path}"
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(audit, indent=2) + "\n")
        tmp.replace(path)

    print(json.dumps(audit, indent=2))
    print("Step 31 passed: 20,000 Phi candidates and 60,000 target observations verified.")

if __name__ == "__main__":
    main()
