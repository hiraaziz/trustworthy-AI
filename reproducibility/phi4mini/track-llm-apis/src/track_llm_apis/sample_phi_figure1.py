
import fcntl
import gzip
import hashlib
import inspect
import json
import random
import sys
import tempfile
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
from huggingface_hub import snapshot_download
from vllm import LLM

from track_llm_apis.sampling.common import CompressedOutput
from track_llm_apis.sampling.vllm_sampling import cleanup_vllm, vllm_inference_bi
from track_llm_apis.util import available_gpu_memory_fraction

ROOT = Path("/workspace/B3IT_Figure1_Phi4Mini")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    variant_id, target_text = sys.argv[1:]
    target = int(target_text)
    raw_manifest = (ROOT / "configs/phi_figure1_evaluation_manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    cfg = manifest["sampling"]
    chunk = cfg["chunk_size"]
    assert 0 < target <= 5000 and target % chunk == 0

    variant = next(v for v in manifest["variants"] if v["id"] == variant_id)
    prompts = manifest["prompts"]
    assert len(prompts) == len(set(prompts)) == 120
    assert digest(json.dumps(prompts, ensure_ascii=False).encode()) == manifest["prompts_sha256"]

    pool_bytes = (ROOT / "data/traffic_pool.json").read_bytes()
    assert digest(pool_bytes) == manifest["traffic_sha256"]
    traffic = json.loads(pool_bytes)
    if isinstance(traffic, dict):
        traffic = next(traffic[k] for k in
                       ("traffic_pool", "prompts", "traffic", "background_prompts")
                       if k in traffic)
    if traffic and isinstance(traffic[0], dict):
        traffic = [row["prompt"] for row in traffic]
    assert len(traffic) == 3138 and all(isinstance(p, str) for p in traffic)

    gpu = torch.cuda.get_device_name(0)
    if "A40" not in gpu:
        raise RuntimeError(f"This evaluation is configured for the A40. Detected {gpu}.")
    base = snapshot_download(
        manifest["model"], revision=manifest["base_revision"], local_files_only=True
    )
    model_path = base if variant_id == "unchanged" else variant["model_path"]
    if variant_id != "unchanged":
        marker = Path(model_path) / "COMPLETE.json"
        assert digest(marker.read_bytes()) == variant["completion_report_sha256"]

    provenance = {
        "manifest_sha256": digest(raw_manifest),
        "variant": variant_id,
        "gpu": gpu,
        "torch": str(torch.__version__),
        "vllm": version("vllm"),
        "transformers": version("transformers"),
        "dtype": manifest["inference_dtype"],
        "backend": "vLLM",
        "sampler_sha256": digest(inspect.getsource(vllm_inference_bi).encode()),
        "collector_sha256": digest(Path(__file__).read_bytes()),
    }
    signature = digest(json.dumps(provenance, sort_keys=True).encode())
    out = ROOT / "results/observations" / variant_id
    out.mkdir(parents=True, exist_ok=True)

    with (out / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        llm = None
        try:
            for start in range(0, target, chunk):
                end = start + chunk
                dest = out / f"{start:05d}_{end:05d}"
                if dest.exists():
                    saved = json.loads((dest / "COMPLETE.json").read_text())
                    files = list(dest.glob("*.json.gz"))
                    assert saved["signature"] == signature, "Collection settings changed."
                    assert saved["start"] == start and saved["samples_per_prompt"] == chunk
                    assert len(files) == 1 and digest(files[0].read_bytes()) == saved["output_sha256"]
                    print(f"Reused {variant_id} {start}-{end}: {saved['rows']:,} rows; "
                          f"{saved['varied_prompts']}/120 prompts varied.", flush=True)
                    continue

                if llm is None:
                    llm = LLM(
                        model=model_path, tokenizer=base,
                        dtype="float16", seed=cfg["engine_seed"],
                        max_model_len=cfg["max_model_len"], max_num_seqs=cfg["max_num_seqs"],
                        enforce_eager=cfg["enforce_eager"],
                        enable_prefix_caching=cfg["enable_prefix_caching"],
                        gpu_memory_utilization=min(0.70, 0.9 * available_gpu_memory_fraction()),
                    )
                    assert llm.llm_engine.model_config.dtype == torch.float16

                random.seed(10000 + start)
                np.random.seed(10000 + start)
                result = vllm_inference_bi(
                    llm=llm, inputs=prompts, traffic_prompts=traffic, n_samples=chunk,
                    temperature=cfg["temperature"], input_fraction=cfg["per_batch_input_fraction"],
                    batch_size=cfg["batch_size"], max_tokens=cfg["max_tokens"],
                )
                assert set(result.counts) == set(prompts)
                assert all(sum(c.values()) == chunk for c in result.counts.values())
                assert result.prompt_lengths == manifest["prompt_lengths"], "Prompt lengths changed."
                varied = sum(len(c) >= 2 for c in result.counts.values())

                output = CompressedOutput(model_name=manifest["model"], gpus=[gpu])
                output.add_bi_results(
                    counts=result.counts, prompt_lengths=result.prompt_lengths,
                    variant=json.dumps(variant["description"], separators=(",", ":")),
                    output_tokens=1,
                )
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    staged = Path(tmp) / "chunk"
                    output.dump_json(staged)
                    files = list(staged.glob("*.json.gz"))
                    assert len(files) == 1
                    with gzip.open(files[0], "rt") as handle:
                        rows = len(json.load(handle)["rows"])
                    assert rows == len(prompts) * chunk
                    metadata = dict(
                        signature=signature, provenance=provenance, variant=variant_id,
                        start=start, samples_per_prompt=chunk, prompt_count=len(prompts),
                        rows=rows, varied_prompts=varied, output_sha256=digest(files[0].read_bytes()),
                    )
                    (staged / "COMPLETE.json").write_text(json.dumps(metadata, indent=2))
                    staged.rename(dest)

                print(f"Saved {variant_id} {start}-{end}: {rows:,} rows; "
                      f"{varied}/120 prompts varied.", flush=True)
        finally:
            if llm is not None:
                cleanup_vllm(llm)

if __name__ == "__main__":
    main()
