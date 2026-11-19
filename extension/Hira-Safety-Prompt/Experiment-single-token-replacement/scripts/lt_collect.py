
import gzip
import json
import random
import time
from pathlib import Path

import orjson
import torch
from vllm import LLM
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import _load_traffic_prompts
from track_llm_apis.sampling.common import CompressedOutput, DataSource
from track_llm_apis.sampling.vllm_sampling import (
    vllm_inference_random_traffic,
    cleanup_vllm,
)


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/experiment.json").read_text())
    mapping = json.loads(
        (root / "results/detection/merged/condition_mapping.json").read_text()
    )
    verification = json.loads(
        (root / "results/lt/verification/verification.json").read_text()
    )
    assert verification["saved_logprobs_match_vllm"]
    assert verification["prompt_token_ids_verified"]

    config.data_dir = root / "data"
    config.sampling.vllm_use_tqdm = False
    lt = config.sampling.logprob
    gpu = torch.cuda.get_device_name(0)

    conditions = {"original": plan["baseline_system_prompt"]}
    conditions.update({
        v["label"]: v["system_prompt"] for v in plan["variants"]
    })
    out = root / "results/lt/collection"
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "model": plan["model"],
        "revision": plan["model_revision"],
        "dtype": plan["dtype"],
        "gpu": gpu,
        "prompt": lt.default_prompt,
        "conditions": conditions,
        "variant_mapping": mapping,
        "observations_per_condition": 5000,
        "block_size": 100,
        "batch_size": lt.batch_size,
        "topk": lt.topk,
        "temperature": lt.temperature,
        "max_output_tokens": 1,
        "reference_samples_per_prompt": lt.reference_samples_per_prompt,
        "seed_base": 700000,
        "prefix_caching": False,
        "enforce_eager": True,
    }
    manifest = out / "settings.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == settings, (
            "Saved LT settings or GPU differ from this run."
        )
    else:
        manifest.write_text(json.dumps(settings, indent=2))

    traffic = _load_traffic_prompts()
    llm = None

    for condition, system_prompt in conditions.items():
        folder = out / condition
        folder.mkdir(exist_ok=True)
        started = time.monotonic()
        fresh_observations = 0

        for start in range(0, 5000, 100):
            path = folder / f"block_{start:05d}.json.gz"

            if path.exists():
                with gzip.open(path, "rb") as f:
                    saved = CompressedOutput.from_json(orjson.loads(f.read()))
                assert len(saved.rows) == 100
                assert all(
                    row.variant == mapping[condition]
                    and row.prompt[0] == lt.default_prompt
                    and row.logprobs_idx is not None
                    for row in saved.rows
                )
                print(
                    f"{condition}: verified saved {start + 100}/5000",
                    flush=True,
                )
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

            data = CompressedOutput(model_name=plan["model"], gpus=[gpu])
            random.seed(settings["seed_base"] + start)

            vllm_inference_random_traffic(
                llm=llm,
                prompts=[lt.default_prompt],
                other_prompts=traffic,
                batch_size=lt.batch_size,
                n_samples=100,
                max_tokens=1,
                temperature=lt.temperature,
                logprobs_topk=lt.topk,
                variant=mapping[condition],
                source=DataSource.LT,
                compressed_output=data,
                system_prompt=system_prompt,
            )

            expected_ids = llm.get_tokenizer().apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": lt.default_prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            assert len(data.rows) == 100
            for row in data.rows:
                assert row.prompt == (lt.default_prompt, len(expected_ids))
                assert row.variant == mapping[condition]
                assert row.text[1] == 1
                assert len(row.first_token_logprobs) >= lt.topk

            temp = path.with_suffix(".tmp")
            with gzip.open(temp, "wb") as f:
                f.write(orjson.dumps(data.to_json()))
            temp.replace(path)

            fresh_observations += 100
            remaining = (
                (time.monotonic() - started) / fresh_observations
                * (5000 - start - 100) / 60
            )
            print(
                f"{condition}: saved {start + 100}/5000 | "
                f"about {remaining:.1f} min left for this condition",
                flush=True,
            )

    if llm is not None:
        cleanup_vllm(llm)
    print("Saved:", out, flush=True)
    print("STEP 19 COMPLETE | 5 conditions × 5000 LT observations", flush=True)


if __name__ == "__main__":
    main()
