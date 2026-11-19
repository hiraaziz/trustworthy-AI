
import argparse
import gzip
import json
import random
import time
from pathlib import Path

from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import _load_traffic_prompts
from track_llm_apis.sampling.vllm_sampling import (
    vllm_inference_bi,
    cleanup_vllm,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/experiment.json").read_text())
    probes = json.loads((root / "results/detection/probes.json").read_text())
    config.data_dir = root / "data"
    bi = config.sampling.bi
    assert bi.model_dump(mode="json") == plan["repository_bi_settings"]

    prompts = {"original": plan["baseline_system_prompt"]}
    prompts.update({
        v["label"]: v["system_prompt"] for v in plan["variants"]
    })
    prompt = prompts[args.condition]
    rounds = 5000
    block_size = 250
    out = root / "results/detection" / args.condition
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "condition": args.condition,
        "system_prompt": prompt,
        "model": plan["model"],
        "revision": plan["model_revision"],
        "dtype": plan["dtype"],
        "probes": probes,
        "rounds": rounds,
        "block_size": block_size,
        "seed_base": 200000,
        "bi_settings": bi.model_dump(mode="json"),
        "prefix_caching": False,
        "enforce_eager": True,
    }
    manifest = out / "settings.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == settings
    else:
        manifest.write_text(json.dumps(settings, indent=2))

    traffic = _load_traffic_prompts()
    llm = None
    started = time.monotonic()
    new_rounds = 0

    for start in range(0, rounds, block_size):
        path = out / f"block_{start:05d}.json.gz"
        stop = min(start + block_size, rounds)

        if path.exists():
            with gzip.open(path, "rt") as f:
                saved = json.load(f)
            assert [r["round"] for r in saved["rows"]] == list(range(start, stop))
            assert all(set(r["tokens"]) == set(probes) for r in saved["rows"])
            print(f"{args.condition}: verified saved {stop}/{rounds}", flush=True)
            continue

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

        rows = []
        lengths = None
        for i in range(start, stop):
            # Match batch randomization across conditions and on resume.
            random.seed(settings["seed_base"] + i)
            result = vllm_inference_bi(
                llm=llm,
                inputs=probes,
                traffic_prompts=traffic,
                n_samples=1,
                temperature=bi.temperature,
                input_fraction=bi.per_batch_input_fraction,
                batch_size=bi.batch_size,
                max_tokens=bi.models[plan["model"]],
                system_prompt=prompt,
            )
            assert all(sum(c.values()) == 1 for c in result.counts.values())
            rows.append({
                "round": i,
                "tokens": {
                    p: next(iter(result.counts[p])) for p in probes
                },
            })
            if lengths is None:
                lengths = result.prompt_lengths
            assert result.prompt_lengths == lengths

        temp = path.with_suffix(".tmp")
        with gzip.open(temp, "wt") as f:
            json.dump({"rows": rows, "prompt_lengths": lengths}, f)
        temp.replace(path)

        new_rounds += len(rows)
        elapsed = time.monotonic() - started
        remaining = elapsed / new_rounds * (rounds - stop) / 60
        print(
            f"{args.condition}: saved {stop}/{rounds} | "
            f"about {remaining:.1f} min left",
            flush=True,
        )

    print(f"Saved: {out}", flush=True)
    if llm is not None:
        cleanup_vllm(llm)
    print(f"COLLECTION COMPLETE | {args.condition} | "
          f"{len(probes)} probes × {rounds} observations", flush=True)


if __name__ == "__main__":
    main()
