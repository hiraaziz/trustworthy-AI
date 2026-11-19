
from pathlib import Path
import json
import random
import time

from transformers import AutoTokenizer
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import (
    BIPhase1ModelResult,
    _load_traffic_prompts,
    get_single_token_inputs,
)
from track_llm_apis.sampling.vllm_sampling import vllm_inference_bi


class SystemPromptLLM(LLM):
    """Apply the endpoint's system prompt to probe and traffic requests alike."""

    def generate(self, prompts, *args, **kwargs):
        tokenizer = self.get_tokenizer()
        formatted = []
        for text in prompts:
            ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            formatted.append({"prompt_token_ids": ids})
        return super().generate(formatted, *args, **kwargs)


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/experiment.json").read_text())
    config.data_dir = root / "data"
    bi = config.sampling.bi
    out = root / "results/discovery"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "phase_1a.json"

    if result_path.exists():
        record = BIPhase1ModelResult.model_validate_json(result_path.read_bytes())
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"], revision=plan["model_revision"]
        )
        # Use the authors' default candidate-order seed.
        candidates = get_single_token_inputs(tokenizer, plan["discovery_candidates"])
        assert len(candidates) == 20_000
        traffic = _load_traffic_prompts()
        print(f"Screening {len(candidates)} candidates with {len(traffic)} traffic prompts.",
              flush=True)

        llm = SystemPromptLLM(
            model=plan["model"],
            revision=plan["model_revision"],
            tokenizer_revision=plan["model_revision"],
            dtype=plan["dtype"],
            enforce_eager=True,
            enable_prefix_caching=False,
            seed=0,
        )
        llm.system_prompt = plan["baseline_system_prompt"]
        random.seed(0)
        start = time.monotonic()

        result = vllm_inference_bi(
            llm=llm,
            inputs=candidates,
            traffic_prompts=traffic,
            n_samples=plan["discovery_queries_per_candidate"],
            temperature=plan["temperature"],
            input_fraction=bi.per_batch_input_fraction,
            batch_size=bi.batch_size,
            max_tokens=plan["max_output_tokens"],
        )
        record = BIPhase1ModelResult(
            model_name=plan["model"],
            bi_config=bi.model_dump(mode="json"),
            phase="1a",
            results=result.counts,
            prompt_lengths=result.prompt_lengths,
            generation_time=time.monotonic() - start,
        )
        assert set(record.results) == set(candidates)
        assert all(sum(c.values()) == 3 for c in record.results.values())
        temp = result_path.with_suffix(".tmp")
        temp.write_text(record.model_dump_json())
        temp.replace(result_path)

    borders = record.get_border_inputs()
    (out / "border_inputs.json").write_text(json.dumps(borders, indent=2))
    print("Candidates screened:", len(record.results))
    print("Border inputs found:", len(borders))
    print("Saved:", result_path)
    print("STEP 6 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
