
import json
import random
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import (
    BIPhase1aOutput,
    BIPhase1ModelResult,
    get_single_token_inputs,
    _load_traffic_prompts,
)
from track_llm_apis.sampling.vllm_sampling import (
    vllm_inference_bi,
    cleanup_vllm,
)



def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/experiment.json").read_text())
    config.data_dir = root / "data"
    bi = config.sampling.bi
    assert bi.model_dump(mode="json") == plan["repository_bi_settings"]

    discovery = BIPhase1aOutput.load_model(
        plan["model"], temperature=bi.temperature
    )
    inputs = discovery.get_border_inputs()
    assert len(inputs) >= bi.phase_1b_min_bis
    rounds = bi.phase_1_queries_per_bi
    out = root / "results/calibration_blocks"
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "model": plan["model"],
        "revision": plan["model_revision"],
        "baseline": plan["baseline_system_prompt"],
        "inputs": inputs,
        "rounds": rounds,
        "block_size": 50,
        "seed_base": 10000,
        "bi_settings": bi.model_dump(mode="json"),
        "dtype": plan["dtype"],
    }
    manifest = out / "settings.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == settings
    else:
        manifest.write_text(json.dumps(settings, indent=2))

    traffic = _load_traffic_prompts()
    llm = None
    counts = {p: {} for p in inputs}
    lengths = {}
    total_seconds = 0.0

    print(f"Calibrating {len(inputs)} inputs × {rounds} observations.", flush=True)

    for start in range(0, rounds, 50):
        path = out / f"block_{start:04d}.json"
        n = min(50, rounds - start)

        if path.exists():
            block = json.loads(path.read_text())
        else:
            if llm is None:
                llm = LLM(
                    model=plan["model"],
                    revision=plan["model_revision"],
                    tokenizer_revision=plan["model_revision"],
                    dtype=plan["dtype"],
                    enforce_eager=True,
                    enable_prefix_caching=False,
                    gpu_memory_utilization=0.80,
                    seed=0,
                )

            random.seed(settings["seed_base"] + start)
            started = time.monotonic()
            result = vllm_inference_bi(
                llm=llm,
                inputs=inputs,
                traffic_prompts=traffic,
                n_samples=n,
                temperature=bi.temperature,
                input_fraction=bi.per_batch_input_fraction,
                batch_size=bi.batch_size,
                max_tokens=bi.models[plan["model"]],
                system_prompt=plan["baseline_system_prompt"],
            )
            block = {
                "counts": result.counts,
                "prompt_lengths": result.prompt_lengths,
                "seconds": time.monotonic() - started,
            }
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(block))
            temp.replace(path)

        assert set(block["counts"]) == set(inputs)
        for prompt, token_counts in block["counts"].items():
            assert sum(token_counts.values()) == n
            for token, count in token_counts.items():
                counts[prompt][token] = counts[prompt].get(token, 0) + count

        lengths.update(block["prompt_lengths"])
        total_seconds += block["seconds"]
        print(f"Calibrated {start + n}/{rounds} rounds saved", flush=True)

    assert all(sum(c.values()) == rounds for c in counts.values())
    record = BIPhase1ModelResult(
        model_name=plan["model"],
        bi_config=bi.model_dump(mode="json"),
        phase="1b",
        results=counts,
        prompt_lengths=lengths,
        generation_time=total_seconds,
    )
    record.save()
    print(
        f"Varied again: {len(record.get_border_inputs())}/{len(inputs)}",
        flush=True,
    )

    if llm is not None:
        cleanup_vllm(llm)
    print("STEP 10 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
