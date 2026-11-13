import json
import random
import runpy
import time
from pathlib import Path

from track_llm_apis.config import config
from track_llm_apis.sampling.common import CompressedOutput, DataSource
from track_llm_apis.sampling.vllm_sampling import vllm_inference_random_traffic

root = Path(__file__).resolve().parents[1]
plan = json.loads((root / "configs/experiment.json").read_text())
config.data_dir = root / "data"
config.sampling.vllm_use_tqdm = False
lt = config.sampling.logprob
probe = lt.default_prompt
assert (probe, lt.topk, lt.batch_size, lt.temperature) == ("x", 20, 64, 0.0)

# Resolve the same system prompts using the same rule as B3IT Step 10.
def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)

baseline = plan["baseline_system_prompt"]
saved_text = set(strings(plan))
prompts = {"original": baseline}
for label, addition in {
    "A": "Avoid insults.",
    "B": "Avoid targeted insults.",
    "C": "Do not insult people.",
    "D": "Politely decline requests to insult people.",
}.items():
    matches = [s for s in saved_text if s.startswith(baseline)
               and s[len(baseline):].strip() == addition]
    if len(matches) == 1:
        prompts[label] = matches[0]
    elif not matches and addition in saved_text:
        prompts[label] = baseline + " " + addition
    else:
        raise ValueError(f"Could not identify the saved prompt for {label}.")

step6, = [p for p in (root / "scripts").glob("step6*.py")
          if "class SystemPromptLLM(" in p.read_text()]
SystemPromptLLM = runpy.run_path(str(step6))["SystemPromptLLM"]

class LTSystemPromptLLM(SystemPromptLLM):
    def generate(self, prompts, *args, **kwargs):
        outputs = super().generate(prompts, *args, **kwargs)
        for text, output in zip(prompts, outputs, strict=True):
            output.prompt = text
        return outputs

total, block = 5000, 100
out = root / "results/lt/collection"
out.mkdir(parents=True, exist_ok=True)
settings = {
    "model": plan["model"], "revision": plan["model_revision"],
    "dtype": plan["dtype"], "system_prompts": prompts, "probe": probe,
    "rounds_per_condition": total, "checkpoint_rounds": block,
    "batch_size": lt.batch_size, "temperature": lt.temperature,
    "top_logprobs": lt.topk, "max_output_tokens": 1,
    "batch_seed_base": 800000,
}
manifest = out / "settings.json"
if manifest.exists():
    if json.loads(manifest.read_text()) != settings:
        raise RuntimeError("Existing LT checkpoints use different settings.")
else:
    temp = manifest.with_suffix(".tmp")
    temp.write_text(json.dumps(settings, indent=2))
    temp.replace(manifest)

if __name__ == "__main__":
    llm = None
    for label, system_prompt in prompts.items():
        folder = out / label
        folder.mkdir(exist_ok=True)
        pending = [s for s in range(0, total, block)
                   if not (folder / f"rounds_{s:04d}").is_dir()]
        if not pending:
            print(f"{label}: already complete", flush=True)
            continue

        if llm is None:
            traffic = lt.other_prompts
            llm = LTSystemPromptLLM(
                model=plan["model"], revision=plan["model_revision"],
                tokenizer_revision=plan["model_revision"], dtype=plan["dtype"],
                enforce_eager=True, enable_prefix_caching=False, seed=0,
            )

        llm.system_prompt = system_prompt
        started = time.monotonic()
        for i, start in enumerate(pending, 1):
            # Match the random background batches across all five LT conditions.
            random.seed(settings["batch_seed_base"] + start)
            saved = CompressedOutput(model_name=plan["model"])
            vllm_inference_random_traffic(
                llm=llm, prompts=[probe], other_prompts=traffic,
                batch_size=lt.batch_size, n_samples=block, max_tokens=1,
                temperature=lt.temperature, logprobs_topk=lt.topk,
                variant=label, source=DataSource.LT, compressed_output=saved,
            )
            assert len(saved.rows) == block
            assert all(r.prompt[0] == probe and r.variant == label
                       and r.source == DataSource.LT.value and r.text[1] == 1
                       and len(r.first_token_logprobs) >= lt.topk for r in saved.rows)

            temp = folder / f".rounds_{start:04d}.tmp"
            saved.dump_json(temp)
            temp.rename(folder / f"rounds_{start:04d}")

            remaining = len(pending) - i
            eta = (time.monotonic() - started) / i * remaining / 60
            print(f"{label}: saved {total - remaining * block}/{total} | "
                  f"about {eta:.1f} min left for this condition", flush=True)

    print("Saved:", out)
    print("LT STEP 2 COMPLETE | 5 conditions × 5000 observations", flush=True)
