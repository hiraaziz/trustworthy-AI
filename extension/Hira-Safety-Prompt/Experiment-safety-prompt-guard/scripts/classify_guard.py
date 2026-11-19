import json
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import _load_traffic_prompts


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    probes = json.loads((root / "results/detection/probes.json").read_text())
    traffic = list(_load_traffic_prompts())
    assert len(set(probes)) == len(probes) == 120
    requests = list(dict.fromkeys([*probes, *traffic]))

    out = root / "results/guard"
    out.mkdir(parents=True, exist_ok=True)
    model = "Qwen/Qwen3Guard-Gen-0.6B"
    settings_path = out / "settings.json"
    if settings_path.exists():
        revision = json.loads(settings_path.read_text())["revision"]
    else:
        revision = HfApi().model_info(model).sha

    settings = {
        "model": model, "revision": revision, "dtype": "bfloat16",
        "temperature": 0.0, "max_guard_output_tokens": 32,
        "trigger_label": "Unsafe", "prefix": "unsafe question ",
    }
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Existing guard decisions use different settings.")
    save_json(settings_path, settings)

    path = out / "decisions.json"
    decisions = json.loads(path.read_text()) if path.exists() else {}
    allowed = {"Safe", "Unsafe", "Controversial"}
    if not set(decisions).issubset(requests) or any(
        label not in allowed for label in decisions.values()
    ):
        raise ValueError("The saved guard decisions do not match this experiment.")

    pending = [q for q in requests if q not in decisions]
    print(f"Unique requests: {len(requests)} | remaining: {len(pending)}", flush=True)

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        llm = LLM(
            model=model, revision=revision, tokenizer_revision=revision,
            dtype="bfloat16", enforce_eager=True,
            gpu_memory_utilization=0.80, seed=0,
        )
        params = SamplingParams(temperature=0.0, max_tokens=32)
        label_line = re.compile(
            r"^\s*Safety:\s*(Safe|Unsafe|Controversial)\s*$", re.MULTILINE
        )

        for start in range(0, len(pending), 128):
            batch = pending[start:start + 128]
            # Use the guard's own user-prompt moderation template.
            inputs = [
                {"prompt_token_ids": tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=True, add_generation_prompt=False,
                )}
                for q in batch
            ]
            outputs = llm.generate(inputs, params, use_tqdm=False)
            for q, output in zip(batch, outputs, strict=True):
                reply = output.outputs[0].text
                match = label_line.search(reply)
                if match is None:
                    raise ValueError(f"Could not read the guard's safety label: {reply!r}")
                decisions[q] = match.group(1)

            save_json(path, decisions)
            print(f"Saved {len(decisions)}/{len(requests)} decisions", flush=True)

    assert set(decisions) == set(requests)
    probe_counts = Counter(decisions[q] for q in probes)
    traffic_counts = Counter(decisions[q] for q in traffic)
    print("Probe labels:", dict(probe_counts))
    print("Background labels:", dict(traffic_counts))
    print(f"Probes that will receive the prefix: {probe_counts['Unsafe']}/{len(probes)}")
    print("Saved:", path)


if __name__ == "__main__":
    main()
