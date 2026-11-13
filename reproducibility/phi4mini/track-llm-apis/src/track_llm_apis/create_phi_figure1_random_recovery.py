
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from track_llm_apis.tinychange import TinyChange, TinyChangeConfig
from track_llm_apis.util import get_model_hash

ROOT = Path("/workspace/B3IT_Figure1_Phi4Mini")
MODEL = "microsoft/Phi-4-mini-instruct"
REV = "cfbefacb99257ffa30c83adab238a50856ac3083"
SCALE = 2**-8
DEST = Path('/tmp/b3it-phi4mini/variant_checkpoints/phi_random')
SPARSE = Path('/workspace/B3IT_Figure1_Phi4Mini/checkpoints/phi_random_indices')

def main():
    if DEST.exists():
        report = json.loads((DEST / "COMPLETE.json").read_text())
        assert report["revision"] == REV and report["seed"] == 0
        assert report["description"] == {"type": "weight_pruning", "method": "random", "scale": SCALE}
        assert report["creation_dtype"] == "float32"
        assert report["changed_parameters_fp16"] > 0
        assert list(DEST.glob("*.safetensors"))
        print("Verified random-pruning checkpoint already exists:", DEST, flush=True)
        print(json.dumps(report, indent=2))
        return

    snapshot = Path(snapshot_download(MODEL, revision=REV, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    print("Loading the unchanged Phi checkpoint in FP32.", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float32, device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    cfg = TinyChangeConfig.model_construct(
        seed=0, variants_device="cuda:0", return_unchanged=False,
        enable_finetuning=False, enable_lora_finetuning=False,
        enable_quantization=False, enable_random_noise=False,
        enable_weight_pruning=True, weight_pruning_magnitude_scale=[], weight_pruning_random_scale=[SCALE],
    )
    tiny = TinyChange(base, tokenizer, cfg)
    print(f"Applying original random pruning at fraction {SCALE}.", flush=True)
    variant = tiny.weight_pruning(scale=SCALE, method="random")
    changed = variant.model
    if get_model_hash(base) != tiny.model_hash:
        raise ValueError("The base model changed during variant creation.")

    references = dict(base.named_parameters())
    visited = set()
    total = changed32 = changed16 = 0
    affected = {}
    zero_indices = {}

    with torch.no_grad():
        for name, param in changed.named_parameters():
            ref = references[name]
            if param.dtype != torch.float32 or ref.dtype != torch.float32:
                raise ValueError(f"Expected FP32 for {name}.")
            if param.shape != ref.shape or param.data_ptr() == ref.data_ptr():
                raise ValueError(f"Invalid independent copy for {name}.")
            visited.add(name)

            # Bound verification memory even for Phi's large embedding matrix.
            a = param.detach().view(-1)
            b = ref.detach().view(-1)
            for start in range(0, a.numel(), 1_000_000):
                x = a[start:start + 1_000_000]
                y = b[start:start + 1_000_000]
                x16, y16 = x.half(), y.half()
                if not torch.isfinite(x16).all() or not torch.isfinite(y16).all():
                    raise FloatingPointError(f"Non-finite weight after FP16 conversion: {name}")
                different = x != y
                n_changed = different.count_nonzero().item()
                if n_changed:
                    if ".mlp." not in name or not name.endswith(".weight"):
                        raise ValueError(f"Unexpected pruning outside MLP weights: {name}")
                    if (different & (x != 0)).any().item():
                        raise ValueError(f"Pruning produced a changed nonzero value: {name}")
                    affected[name] = affected.get(name, 0) + n_changed
                    # x and y are slices of the flattened named parameter.
                    idx = different.nonzero(as_tuple=True)[0] + start
                    if idx[-1].item() >= 2**31:
                        raise ValueError(f"Index exceeds int32 capacity in {name}")
                    zero_indices.setdefault(name, []).append(idx.to("cpu", torch.int32))
                changed32 += n_changed
                changed16 += (x16 != y16).count_nonzero().item()
                total += x.numel()

    if visited != set(references) or total != 3_836_021_760:
        raise ValueError(f"Incomplete weight comparison: {total:,} parameters.")
    if changed32 == 0 or changed16 == 0:
        raise ValueError("Random pruning failed the FP32/FP16 weight-change check.")

    report = dict(
        model=MODEL, revision=REV, variant_id="random",
        description=variant.description, seed=0,
        creation_dtype="float32", evaluation_dtype="float16",
        affected_tensors=len(affected), zeroed_by_tensor=affected,
        total_parameters_compared=total, changed_parameters_fp32=changed32,
        changed_parameters_fp16=changed16,
    )
    print(json.dumps(report, indent=2), flush=True)

    indices = {name: torch.cat(parts) for name, parts in zero_indices.items()}
    if len(indices) != 64 or sum(v.numel() for v in indices.values()) != changed32:
        raise ValueError("Incomplete pruning index set.")
    if (changed32, changed16) != (9437184, 9437179):
        raise ValueError("Retry differs from the earlier verified pruning counts.")

    import hashlib
    index_name = "zero_indices.safetensors"
    if SPARSE.exists():
        saved = load_file(str(SPARSE / index_name))
        prior = json.loads((SPARSE / "COMPLETE.json").read_text())
        if prior["revision"] != REV or prior["description"] != report["description"]:
            raise ValueError("Existing sparse checkpoint has different settings.")
        if saved.keys() != indices.keys() or any(
            not torch.equal(saved[k], indices[k]) for k in indices
        ):
            raise ValueError("Existing sparse checkpoint has different indices.")
    else:
        with tempfile.TemporaryDirectory(dir=SPARSE.parent) as temp:
            compact = Path(temp) / "compact"
            compact.mkdir()
            index_file = compact / index_name
            save_file(indices, str(index_file))
            saved = load_file(str(index_file))
            if saved.keys() != indices.keys() or any(
                not torch.equal(saved[k], indices[k]) for k in indices
            ):
                raise ValueError("Saved pruning indices failed read-back verification.")
            metadata = dict(report, index_file=index_name,
                            index_sha256=hashlib.sha256(index_file.read_bytes()).hexdigest(),
                            reconstruction="Load pinned base in FP32; set listed flat indices "
                                           "of each named parameter to zero.")
            (compact / "COMPLETE.json").write_text(json.dumps(metadata, indent=2))
            compact.rename(SPARSE)
    print("Persistent pruning indices verified:", SPARSE, flush=True)
    del references, tiny, base
    torch.cuda.empty_cache()
    print("Writing and checking the full FP32 checkpoint on the A40 container disk.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".phi_random_", dir=DEST.parent) as tmp:
        staged = Path(tmp) / "model"
        changed.save_pretrained(staged, safe_serialization=True, max_shard_size="2GB")
        # Preserve the base tokenizer files and settings for evaluation.
        base_tok = AutoTokenizer.from_pretrained(snapshot)
        base_tok.save_pretrained(staged)
        saved_tok = AutoTokenizer.from_pretrained(staged)

        expected = changed.state_dict()
        stored = set()
        for file in staged.glob("*.safetensors"):
            with safe_open(str(file), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key not in expected or key in stored:
                        raise ValueError(f"Unexpected saved tensor: {key}")
                    tensor = shard.get_slice(key)
                    if tuple(tensor.get_shape()) != tuple(expected[key].shape):
                        raise ValueError(f"Saved shape mismatch: {key}")
                    if expected[key].is_floating_point() and tensor.get_dtype() != "F32":
                        raise ValueError(f"Saved tensor is not FP32: {key}")
                    stored.add(key)

        # A tied weight may be serialized under just one of its names.
        aliases = {}
        for key, p in changed.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(p), set()).add(key)
        if any(names.isdisjoint(stored) for names in aliases.values()):
            raise ValueError("Saved checkpoint is missing model parameters.")

        prompts = json.loads(
            (ROOT / "configs/phi_qwen_protocol_selected_120.json").read_text()
        )["prompts"]
        traffic = json.loads((ROOT / "data/traffic_pool.json").read_text())
        if isinstance(traffic, dict):
            traffic = next(traffic[k] for k in
                           ("traffic_pool", "prompts", "traffic", "background_prompts")
                           if k in traffic)
        if isinstance(traffic[0], dict):
            traffic = [row["prompt"] for row in traffic]
        assert len(prompts) == 120 and len(traffic) == 3138
        texts = prompts + traffic
        if (base_tok.get_vocab() != saved_tok.get_vocab()
                or base_tok(texts, verbose=False)["input_ids"]
                != saved_tok(texts, verbose=False)["input_ids"]):
            raise ValueError("Tokenizer mismatch on the evaluation inputs.")

        (staged / "COMPLETE.json").write_text(json.dumps(report, indent=2))
        staged.rename(DEST)
    print("Phi random pruning complete:", DEST, flush=True)

if __name__ == "__main__":
    main()
