
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from track_llm_apis.config import config
from track_llm_apis.sampling import bi_phase_1 as phase


def main():
    root = Path(sys.argv[1])
    model = "google/gemma-3-1b-it"
    snap = json.loads((root / "audit/gemma_snapshot.json").read_text())
    discovery = json.loads((root / "audit/step19_gemma_discovery.json").read_text())
    snapshot = Path(snap["snapshot"])
    assert snap["model"] == discovery["model"] == model
    assert snapshot.is_dir() and snapshot.name == snap["revision"] == discovery["revision"]
    assert config.data_dir == root / "data"

    bi = config.sampling.bi
    assert (bi.phase_1_queries_per_bi, bi.max_border_inputs) == (1000, 120)
    assert (bi.temperature, bi.models[model]) == (0, 1)
    assert (bi.batch_size, bi.per_batch_input_fraction) == (64, 0.95)

    phase_a = phase.BIPhase1aOutput.load_model(model)
    discovered = phase_a.get_border_inputs()
    assert phase_a.bi_config == bi.model_dump(mode="json")
    assert len(discovered) == discovery["discovered_border_inputs"]
    assert len(discovered) >= bi.phase_1b_min_bis, "Below the upstream calibration minimum"

    try:
        result = phase.BIPhase1bOutput.load_model(model)
    except FileNotFoundError:
        result = None

    if result is None:
        original_init = phase.init_vllm_bi

        def pinned_engine(name, device):
            assert name == model
            return original_init(str(snapshot), device)

        print(f"Calibrating {len(discovered):,} Gemma inputs × 1,000 observations.", flush=True)
        with patch.object(phase, "init_vllm_bi", pinned_engine):
            phase.phase_1b(device="cuda:0", model=model)
        result = phase.BIPhase1bOutput.load_model(model)
    else:
        print("Found saved Gemma calibration. Verifying every input.", flush=True)

    assert result.model_name == model and result.phase == "1b"
    assert result.bi_config == bi.model_dump(mode="json")
    assert set(result.results) == set(discovered), "Calibration input set differs from discovery"
    assert result.prompt_lengths.keys() == result.results.keys()
    totals = [sum(counts.values()) for counts in result.results.values()]
    assert all(n == 1000 for n in totals), "Incomplete calibration count"

    eligible = result.get_border_inputs()
    selected = eligible[:bi.max_border_inputs]
    assert len(set(selected)) == len(selected)
    record = {
        "model": model,
        "revision": snap["revision"],
        "traffic_dataset_fingerprint": discovery["traffic_dataset_fingerprint"],
        "discovered_inputs": len(discovered),
        "calibration_observations": sum(totals),
        "eligible_after_calibration": len(eligible),
        "selected_count": len(selected),
        "selection_rule": "First 120 returned by phase 1b get_border_inputs()",
        "prompts": selected,
        "prompt_lengths": {p: result.prompt_lengths[p] for p in selected},
        "prompts_sha256": hashlib.sha256(
            json.dumps(selected, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "upstream_result": str(result._get_path()),
    }
    path = root / "results/gemma/selected_probes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == record, "Existing probe selection differs"
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)

    print(f"Calibration observations verified: {sum(totals):,}")
    print(f"Eligible after calibration: {len(eligible):,}")
    print(f"Selected probes: {len(selected)}")
    print("Saved:", path)
    if len(selected) < bi.max_border_inputs:
        raise RuntimeError(f"Only {len(selected)} probes qualified. Actual results saved.")
    print("Step 20 passed: full Gemma calibration verified and 120 ordered probes saved.")


if __name__ == "__main__":
    main()
