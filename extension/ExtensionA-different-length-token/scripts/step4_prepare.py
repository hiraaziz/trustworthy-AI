
from pathlib import Path
import json
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
model = "Qwen/Qwen2.5-0.5B-Instruct"
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

baseline = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
user_message = {"role": "user", "content": "x"}

def chat_ids(system_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, user_message],
        tokenize=True,
        add_generation_prompt=True,
    )

baseline_ids = chat_ids(baseline)
default_ids = tokenizer.apply_chat_template(
    [user_message], tokenize=True, add_generation_prompt=True
)
assert baseline_ids == default_ids, "Explicit baseline differs from Qwen's default."

variants = {}
for name, addition in {
    "A": "Avoid insults.",
    "B": "Avoid targeted insults.",
    "C": "Do not insult people.",
    "D": "Politely decline requests to insult people.",
}.items():
    system_prompt = baseline + " " + addition
    added_tokens = len(chat_ids(system_prompt)) - len(baseline_ids)
    variants[name] = {
        "addition": addition,
        "system_prompt": system_prompt,
        "added_tokens": added_tokens,
    }
    print(f"{name}: +{added_tokens} tokens | {addition}")

plan = {
    "model": model,
    "model_revision": revision,
    "repository_commit": "ad2f69295797a5f03e7b480246080d91087d9007",
    "baseline_system_prompt": baseline,
    "variants": variants,
    "dtype": "bfloat16",
    "temperature": 0.0,
    "max_output_tokens": 1,
    "discovery_candidates": 20000,
    "discovery_queries_per_candidate": 3,
    "reference_pool_per_border_input": 1000,
    "observations_per_evaluation_probe_per_condition": 5000,
    "probes_per_test": 5,
    "reference_observations_per_probe_per_test": 50,
    "detection_observations_per_probe_per_test": 3,
    "test_statistics": 1000,
    "bootstrap_repeats": 250,
}

path = root / "configs/experiment.json"
if path.exists() and json.loads(path.read_text()) != plan:
    raise RuntimeError(f"A different experiment plan already exists at {path}")
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(plan, indent=2))
temp.replace(path)
print("Saved:", path)
print("STEP 4 COMPLETE")
