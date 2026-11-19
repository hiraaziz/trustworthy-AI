import json
import random
import time
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import BIPhase1ModelResult, _load_traffic_prompts
from track_llm_apis.sampling.vllm_sampling import vllm_inference_bi


def save_text(path, text):
    temp = path.with_suffix(".tmp")
    temp.write_text(text)
    temp.replace(path)


def main():
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    bi = config.sampling.bi
    discovery = root / "results/discovery/phase_1a.json"
    previous = BIPhase1ModelResult.model_validate_json(discovery.read_bytes())
    questions = previous.get_border_inputs()
    if not questions:
        raise ValueError("Discovery contains no border candidates.")

    model = previous.model_name
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    rounds = bi.phase_1_queries_per_bi
    assert rounds == 1000 and bi.temperature == 0.0
    block_size = 100

    out = root / "results/calibration"
    blocks = out / "blocks"
    blocks.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    def format_request(text):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )

    formatted = [format_request(q) for q in questions]
    traffic = [format_request(q) for q in _load_traffic_prompts()]
    totals = {q: Counter() for q in questions}
    seconds = 0.0
    llm = None

    for start in range(0, rounds, block_size):
        path = blocks / f"rounds_{start:04d}.json"
        if path.exists():
            block = json.loads(path.read_text())
        else:
            if llm is None:
                llm = LLM(
                    model=model, revision=revision, tokenizer_revision=revision,
                    dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False,
                    gpu_memory_utilization=0.80, seed=0,
                )

            random.seed(10_000 + start)
            began = time.monotonic()
            result = vllm_inference_bi(
                llm=llm, inputs=formatted, traffic_prompts=traffic,
                n_samples=block_size, temperature=bi.temperature,
                input_fraction=bi.per_batch_input_fraction,
                batch_size=bi.batch_size, max_tokens=1,
            )
            block = {
                "counts": {q: result.counts[p] for q, p in zip(questions, formatted)},
                "seconds": time.monotonic() - began,
            }

        counts = block["counts"]
        if counts.keys() != totals.keys() or any(sum(c.values()) != block_size for c in counts.values()):
            raise ValueError(f"Incomplete or mismatched calibration block: {path}")

        if not path.exists():
            save_text(path, json.dumps(block, ensure_ascii=False))
        for q in questions:
            totals[q].update(counts[q])
        seconds += block["seconds"]
        print(f"Verified {start + block_size}/{rounds} rounds", flush=True)

    assert all(sum(c.values()) == rounds for c in totals.values())
    record = BIPhase1ModelResult(
        model_name=model, bi_config=bi.model_dump(mode="json"), phase="1b",
        results=totals,
        prompt_lengths={q: previous.prompt_lengths[q] for q in questions},
        generation_time=seconds,
    )
    confirmed = record.get_border_inputs()
    save_text(out / "phase_1b.json", record.model_dump_json())
    save_text(out / "confirmed_questions.json", json.dumps(confirmed, ensure_ascii=False, indent=2))
    print("Verified calibration observations:", len(questions) * rounds)
    print(f"Varied again: {len(confirmed)}/{len(questions)}")
    print("Saved:", out / "confirmed_questions.json")


if __name__ == "__main__":
    main()
