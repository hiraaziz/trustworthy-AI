
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

    marker = root / "results/discovery_complete.json"
    if marker.exists():
        record = BIPhase1aOutput.load_model(
            plan["model"], temperature=bi.temperature
        )
        print("Using completed discovery from this experiment.")
        print("Candidates screened:", len(record.results))
        print("Border inputs found:", len(record.get_border_inputs()))
        print("STEP 9 COMPLETE", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], revision=plan["model_revision"]
    )
    candidates = get_single_token_inputs(
        tokenizer, bi.phase_1_tokens_per_endpoint
    )
    assert len(candidates) == 20_000
    traffic = _load_traffic_prompts()

    print(
        f"Screening {len(candidates)} candidates × "
        f"{bi.phase_1_queries_per_token} observations.",
        flush=True,
    )

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

    random.seed(0)
    started = time.monotonic()
    result = vllm_inference_bi(
        llm=llm,
        inputs=candidates,
        traffic_prompts=traffic,
        n_samples=bi.phase_1_queries_per_token,
        temperature=bi.temperature,
        input_fraction=bi.per_batch_input_fraction,
        batch_size=bi.batch_size,
        max_tokens=bi.models[plan["model"]],
        system_prompt=plan["baseline_system_prompt"],
    )

    assert set(result.counts) == set(candidates)
    assert all(
        sum(counts.values()) == bi.phase_1_queries_per_token
        for counts in result.counts.values()
    )

    record = BIPhase1ModelResult(
        model_name=plan["model"],
        bi_config=bi.model_dump(mode="json"),
        phase="1a",
        results=result.counts,
        prompt_lengths=result.prompt_lengths,
        generation_time=time.monotonic() - started,
    )
    record.save()

    borders = record.get_border_inputs()
    details = {
        "model": plan["model"],
        "baseline_system_prompt": plan["baseline_system_prompt"],
        "candidates_screened": len(candidates),
        "border_inputs_found": len(borders),
        "generation_seconds": record.generation_time,
    }
    temp = marker.with_suffix(".tmp")
    temp.write_text(json.dumps(details, indent=2))
    temp.replace(marker)

    print("Candidates screened:", len(candidates), flush=True)
    print("Border inputs found:", len(borders), flush=True)
    print("Saved under:", config.bi_phase_1_dir, flush=True)
    cleanup_vllm(llm)
    print("STEP 9 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
