
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from track_llm_apis.config import config
from track_llm_apis.sampling import bi_phase_1 as phase


def main():
    root = Path(sys.argv[1])
    model = "microsoft/Phi-4-mini-instruct"
    snapshot_record = json.loads((root / "audit/phi4mini_snapshot.json").read_text())
    discovery = json.loads((root / "audit/step31_phi4mini_discovery.json").read_text())
    snapshot = Path(snapshot_record["snapshot"])

    assert snapshot.is_dir()
    assert snapshot.name == snapshot_record["revision"] == discovery["revision"]
    assert model == snapshot_record["model"] == discovery["model"]
    assert config.data_dir.resolve() == (root / "data").resolve()

    bi = config.sampling.bi
    assert (bi.phase_1_queries_per_bi, bi.max_border_inputs) == (1000, 120)
    assert (bi.temperature, bi.models[model]) == (0, 1)
    assert (bi.batch_size, bi.per_batch_input_fraction) == (64, 0.95)

    phase_a = phase.BIPhase1aOutput.load_model(model)
    discovered = phase_a.get_border_inputs()
    assert phase_a.bi_config == bi.model_dump(mode="json")
    assert len(discovered) == discovery["discovered_border_inputs"]

    try:
        result = phase.BIPhase1bOutput.load_model(model)
    except FileNotFoundError:
        result = None

    if result is None:
        original_init = phase.init_vllm_bi

        def init_pinned(name, device):
            assert name == model
            return original_init(str(snapshot), device)

        print(f"Calibrating {len(discovered):,} Phi inputs × 1,000 observations.", flush=True)
        with patch.object(phase, "init_vllm_bi", init_pinned):
            phase.phase_1b(device="cuda:0", model=model)
        result = phase.BIPhase1bOutput.load_model(model)
    else:
        print("Found saved Phi calibration. Verifying every input.", flush=True)

    assert result.model_name == model and result.phase == "1b"
    assert result.bi_config == bi.model_dump(mode="json")
    assert set(result.results) == set(discovered), "Calibration and discovery inputs differ."
    assert all(sum(counts.values()) == 1000 for counts in result.results.values())
    assert result.prompt_lengths.keys() == result.results.keys()

    eligible = result.get_border_inputs()
    selected = eligible[:bi.max_border_inputs]
    report = {
        "model": model,
        "revision": snapshot_record["revision"],
        "traffic_prompts": discovery["traffic_prompts"],
        "discovered_inputs": len(discovered),
        "calibration_observations": sum(sum(c.values()) for c in result.results.values()),
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

    destination = root / "results/phi4mini/selected_probes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert json.loads(destination.read_text()) == report, "Existing probe selection differs."
    else:
        tmp = destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(destination)

    print(f"Calibration observations verified: {report['calibration_observations']:,}")
    print(f"Eligible after calibration: {len(eligible):,}")
    print(f"Selected probes: {len(selected)}")
    print("Saved:", destination)
    if len(selected) < 120:
        raise RuntimeError(f"Only {len(selected)} probes qualified. Actual results saved.")
    print("Step 32 passed: full Phi calibration verified and 120 ordered probes saved.")


if __name__ == "__main__":
    main()
