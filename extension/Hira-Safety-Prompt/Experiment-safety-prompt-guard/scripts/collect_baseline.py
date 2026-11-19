import gzip
import json
import random
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import _load_traffic_prompts
from track_llm_apis.sampling.vllm_sampling import vllm_inference_bi


def main():
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    bi = config.sampling.bi
    probes = json.loads((root / "results/detection/probes.json").read_text())
    assert len(probes) == len(set(probes)) == 120

    out = root / "results/detection/original"
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "condition": "guard_off",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "system_prompt": "You are a helpful assistant.",
        "probes_in_column_order": probes,
        "rounds": 5000,
        "block_size": 250,
        "round_seed_base": 200000,
        "temperature": bi.temperature,
        "batch_size": bi.batch_size,
        "input_fraction": bi.per_batch_input_fraction,
        "max_tokens": 1,
    }
    manifest = out / "settings.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise RuntimeError("Saved baseline settings differ from this run.")
    else:
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        temp.replace(manifest)

    tokenizer = AutoTokenizer.from_pretrained(plan["model"], revision=plan["revision"])

    def format_request(text):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": plan["system_prompt"]},
             {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )

    inputs = [format_request(q) for q in probes]
    traffic = [format_request(q) for q in _load_traffic_prompts()]
    llm = None
    verified = 0

    for start in range(0, plan["rounds"], plan["block_size"]):
        stop = min(start + plan["block_size"], plan["rounds"])
        path = out / f"rounds_{start:04d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as f:
                rows = json.load(f)
        else:
            if llm is None:
                llm = LLM(
                    model=plan["model"], revision=plan["revision"],
                    tokenizer_revision=plan["revision"], dtype="bfloat16",
                    enforce_eager=True, enable_prefix_caching=False,
                    gpu_memory_utilization=0.80, seed=0,
                )

            rows = []
            began = time.monotonic()
            for i in range(start, stop):
                random.seed(plan["round_seed_base"] + i)
                result = vllm_inference_bi(
                    llm=llm, inputs=inputs, traffic_prompts=traffic,
                    n_samples=1, temperature=plan["temperature"],
                    input_fraction=plan["input_fraction"],
                    batch_size=plan["batch_size"], max_tokens=1,
                )
                row = []
                for prompt in inputs:
                    counts = result.counts[prompt]
                    if len(counts) != 1 or sum(counts.values()) != 1:
                        raise ValueError(f"Expected one observation in round {i}.")
                    row.append(next(iter(counts)))
                rows.append(row)

            temp = path.with_suffix(".tmp")
            with gzip.open(temp, "wt", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            temp.replace(path)
            elapsed = time.monotonic() - began
            print(f"Saved {stop}/5000 | about {elapsed * (5000-stop) / (stop-start) / 60:.1f} min left",
                  flush=True)

        if len(rows) != stop-start or any(len(row) != len(probes) for row in rows):
            raise ValueError(f"Incomplete block: {path}")
        verified += len(rows) * len(probes)

    assert verified == 600000
    print("Verified baseline observations:", verified)
    print("Saved:", out)


if __name__ == "__main__":
    main()
