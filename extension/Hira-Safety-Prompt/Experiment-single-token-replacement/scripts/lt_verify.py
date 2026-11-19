
import json
import random
from pathlib import Path

import torch
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import _load_traffic_prompts
from track_llm_apis.sampling.common import CompressedOutput, DataSource
from track_llm_apis.sampling.vllm_sampling import (
    vllm_inference_random_traffic,
    cleanup_vllm,
)
from track_llm_apis.sampling.analyze_logprobs import (
    logprob_two_sample_test_from_compressed_output_row,
)
from track_llm_apis.tinychange import TinyChange


class VerifiedLLM(LLM):
    def generate(self, prompts, *args, **kwargs):
        outputs = super().generate(prompts, *args, **kwargs)
        assert len(outputs) == len(prompts)

        # Verify the actual token IDs used by vLLM.
        for request, output in zip(prompts, outputs):
            assert list(output.prompt_token_ids) == request["prompt_token_ids"]

            if request["prompt_token_ids"] == self.expected_probe_ids:
                completion = output.outputs[0]
                assert len(completion.token_ids) == 1
                assert completion.logprobs is not None
                self.observed_logprobs.append({
                    int(token): value.logprob
                    for token, value in completion.logprobs[0].items()
                })
        return outputs


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/experiment.json").read_text())
    config.data_dir = root / "data"
    config.analysis.device = "cpu"
    config.sampling.vllm_use_tqdm = False
    lt = config.sampling.logprob
    prompt = lt.default_prompt

    llm = VerifiedLLM(
        model=plan["model"],
        revision=plan["model_revision"],
        tokenizer_revision=plan["model_revision"],
        dtype=plan["dtype"],
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.80,
        seed=0,
    )
    llm.expected_probe_ids = llm.get_tokenizer().apply_chat_template(
        [
            {"role": "system", "content": plan["baseline_system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    llm.observed_logprobs = []
    data = CompressedOutput(
        model_name=plan["model"],
        gpus=[torch.cuda.get_device_name(0)],
    )

    random.seed(600000)
    vllm_inference_random_traffic(
        llm=llm,
        prompts=[prompt],
        other_prompts=_load_traffic_prompts(),
        batch_size=lt.batch_size,
        n_samples=60,
        max_tokens=1,
        temperature=lt.temperature,
        logprobs_topk=lt.topk,
        variant=TinyChange.unchanged_str(),
        source=DataSource.LT,
        compressed_output=data,
        system_prompt=plan["baseline_system_prompt"],
    )

    assert len(data.rows) == 60
    assert len(llm.observed_logprobs) == 60
    for row, observed in zip(data.rows, llm.observed_logprobs):
        assert row.prompt == (prompt, len(llm.expected_probe_ids))
        assert row.first_token_logprobs == observed
        assert len(observed) >= lt.topk

    out = root / "results/lt/verification"
    data.dump_json(out)
    loaded = CompressedOutput.from_json_dir(out)
    assert len(loaded.rows) == 60
    for row, observed in zip(loaded.rows, llm.observed_logprobs):
        assert row.first_token_logprobs == observed

    score = logprob_two_sample_test_from_compressed_output_row(
        {prompt: loaded.rows[:50]},
        {prompt: loaded.rows[50:]},
        pvalue_b=0,
    )
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "system_prompt": plan["baseline_system_prompt"],
        "prompt": prompt,
        "observations": 60,
        "topk_requested": lt.topk,
        "prompt_token_ids_verified": True,
        "saved_logprobs_match_vllm": True,
        "reload_verified": True,
        "original_vs_original_statistic": score.statistic,
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2), flush=True)
    cleanup_vllm(llm)
    print("STEP 18 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
