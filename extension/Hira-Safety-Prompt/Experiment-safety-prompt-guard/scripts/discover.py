import json
import random
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import (
    BIPhase1ModelResult,
    _load_traffic_prompts,
)
from track_llm_apis.sampling.vllm_sampling import vllm_inference_bi


def main():
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    bi = config.sampling.bi
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    system_prompt = "You are a helpful assistant."

    candidates = json.loads((root / "data/candidate_questions.json").read_text())
    questions = candidates["questions"]
    assert len(questions) == len(set(questions)) == 5000
    assert bi.phase_1_queries_per_token == 3 and bi.temperature == 0.0

    out = root / "results/discovery"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase_1a.json"

    if path.exists():
        record = BIPhase1ModelResult.model_validate_json(path.read_bytes())
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

        def format_request(text):
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )

        # Qwen receives the same chat format for candidates and background traffic.
        formatted = [format_request(q) for q in questions]
        traffic = [format_request(q) for q in _load_traffic_prompts()]
        print(f"Screening {len(questions)} questions × 3 with the guard off.", flush=True)

        llm = LLM(
            model=model, revision=revision, tokenizer_revision=revision,
            dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False,
            gpu_memory_utilization=0.80, seed=0,
        )
        random.seed(0)
        started = time.monotonic()
        result = vllm_inference_bi(
            llm=llm, inputs=formatted, traffic_prompts=traffic,
            n_samples=bi.phase_1_queries_per_token, temperature=bi.temperature,
            input_fraction=bi.per_batch_input_fraction, batch_size=bi.batch_size,
            max_tokens=1,
        )

        # Save results under the original question text for calibration and testing.
        record = BIPhase1ModelResult(
            model_name=model, bi_config=bi.model_dump(mode="json"), phase="1a",
            results={q: result.counts[p] for q, p in zip(questions, formatted)},
            prompt_lengths={q: result.prompt_lengths[p] for q, p in zip(questions, formatted)},
            generation_time=time.monotonic() - started,
        )

    assert set(record.results) == set(questions)
    assert all(sum(counts.values()) == 3 for counts in record.results.values())

    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json())
        tmp.replace(path)

    borders = record.get_border_inputs()
    border_path = out / "border_candidates.json"
    border_path.write_text(json.dumps(borders, ensure_ascii=False, indent=2))
    print("Verified first-token observations:", 3 * len(questions))
    print("Border candidates found:", len(borders))
    print("Saved:", border_path)


if __name__ == "__main__":
    main()
