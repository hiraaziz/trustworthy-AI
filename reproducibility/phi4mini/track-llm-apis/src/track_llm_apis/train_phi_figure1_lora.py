
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from track_llm_apis import tinychange as tc
from track_llm_apis.config import config
from track_llm_apis.phi_split_qkv import split_qkv, restore_qkv
from track_llm_apis.util import patch_chat_template

ROOT = Path("/workspace/B3IT_Figure1_Phi4Mini")
MODEL = "microsoft/Phi-4-mini-instruct"
REV = "cfbefacb99257ffa30c83adab238a50856ac3083"
DATA_HASH = "e21c2d083c6f4d802b9b043fc071f18ca8f950ac5818cd4340c76785079bf4c3"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
CKPT = ROOT / "checkpoints"
ADAPTER = CKPT / "phi_lora_128_adapter"
FINAL = CKPT / "phi_lora_128"
SPEC = dict(model=MODEL, revision=REV, dataset_sha256=DATA_HASH,
            seed=0, updates=128, epochs=1, batch_size=1, lr=1e-6,
            rank=8, alpha=8, dropout=0.0, max_length=1024,
            targets=TARGETS, creation_dtype="float32", evaluation_dtype="float16")

def read_marker(folder):
    marker = json.loads((folder / "COMPLETE.json").read_text())
    if marker["spec"] != SPEC:
        raise ValueError(f"Existing checkpoint has different settings: {folder}")
    return marker

class CheckTraining(tc.pl.Callback):
    def on_train_start(self, trainer, pl_module):
        optimizer, = trainer.optimizers
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError(f"Unexpected optimizer: {type(optimizer).__name__}")
        if any(g["lr"] != SPEC["lr"] for g in optimizer.param_groups):
            raise ValueError("SGD learning rate differs from Qwen.")
        print("Verified SGD at 1e-6. Starting 128 updates.", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite loss at update {trainer.global_step}")
        if trainer.global_step % 16 == 0:
            print(f"Completed {trainer.global_step}/128 SGD updates.", flush=True)

def verify_weights(model, snapshot):
    # Read one slice of the immutable base checkpoint at a time.
    files = {}
    for shard in snapshot.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                files[key] = shard

    changed32 = changed16 = compared = 0
    with torch.no_grad():
        for key, param in model.named_parameters():
            if key not in files:
                raise ValueError(f"No base tensor for {key}")
            if param.dtype != torch.float32:
                raise ValueError(f"Unexpected dtype for {key}: {param.dtype}")
            with safe_open(str(files[key]), framework="pt", device="cpu") as handle:
                original = handle.get_slice(key)
                if tuple(original.get_shape()) != tuple(param.shape):
                    raise ValueError(f"Unexpected fused shape for {key}")
                rows = max(1, 1_000_000 // max(1, param[0].numel()))
                for start in range(0, param.shape[0], rows):
                    end = min(start + rows, param.shape[0])
                    ref = original[start:end].float().to(param.device)
                    val = param[start:end].detach()
                    a, b = val.half(), ref.half()
                    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                        raise FloatingPointError(f"Non-finite FP16 weights in {key}")
                    n = torch.count_nonzero(val != ref).item()
                    if n and not key.endswith((".self_attn.qkv_proj.weight",
                                               ".self_attn.o_proj.weight")):
                        raise ValueError(f"Unexpected changes outside LoRA targets: {key}")
                    changed32 += n
                    changed16 += torch.count_nonzero(a != b).item()
                    compared += val.numel()
    return dict(total_parameters_compared=compared,
                changed_parameters_fp32=changed32, changed_parameters_fp16=changed16)

def main():
    if FINAL.exists():
        report = read_marker(FINAL)
        if report["changed_parameters_fp16"] <= 0 or not list(FINAL.glob("*.safetensors")):
            raise ValueError("Existing final checkpoint failed validation.")
        print("Verified checkpoint already exists:", FINAL)
        return

    data_bytes = (ROOT / "data/lora_training_128.json").read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != DATA_HASH:
        raise ValueError("Training file checksum mismatch.")
    conversations = json.loads(data_bytes)["conversations"]
    assert len(conversations) == 128
    snapshot = Path(snapshot_download(MODEL, revision=REV, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    patch_chat_template(tokenizer, config.chat_templates)
    tokenizer.truncation_side = "right"

    tc.pl.seed_everything(0, workers=True)
    print("Loading one FP32 model onto the GPU.", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float32, attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    )
    original_cache_setting = base.config.use_cache
    base.config.use_cache = False
    assert split_qkv(base) == (3072, 1024, 1024)

    if ADAPTER.exists():
        read_marker(ADAPTER)
        model = PeftModel.from_pretrained(base, ADAPTER)
        print("Reusing completed 128-update adapter.", flush=True)
    else:
        adapter_config = LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, adapter_config)
        adapted = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
        for target in TARGETS:
            assert sum(n.endswith("." + target) for n in adapted) == 32, target
        assert len(adapted) == 128

        # Recompute activations during backward to fit FP32 training on the A5000.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        dataset = Dataset.from_dict({"conversation": conversations})
        # Supply the verified dataset directly, bypassing default dataset downloads.
        cfg = tc.TinyChangeConfig.model_construct(
            finetuning_dataset=dataset, finetuning_samples=[128],
            finetuning_max_length=1024, variants_device="cuda:0", seed=0,
        )
        data = tc.FinetuningDataModule(
            config=cfg, n_samples=128, batch_size=1, tokenizer=tokenizer,
        )
        lightning_model = tc.LightningModel(model, lr=1e-6)
        trainer = tc.pl.Trainer(
            max_epochs=1, accelerator="cuda", devices=[0], precision="32-true",
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, callbacks=[CheckTraining()],
        )
        trainer.fit(model=lightning_model, datamodule=data)
        if trainer.global_step != 128:
            raise ValueError(f"Expected 128 SGD updates, got {trainer.global_step}.")
        with tempfile.TemporaryDirectory(dir=CKPT) as tmp:
            staged = Path(tmp) / "adapter"
            model.save_pretrained(staged, save_embedding_layers=False)
            (staged / "COMPLETE.json").write_text(json.dumps({"spec": SPEC}, indent=2))
            staged.rename(ADAPTER)
        print("Saved trained adapter:", ADAPTER, flush=True)
        del trainer, lightning_model, data

    trained = model.merge_and_unload()
    restore_qkv(trained)
    trained.gradient_checkpointing_disable()
    trained.config.use_cache = original_cache_setting
    trained.eval()

    print("Checking FP32 weights and changes surviving FP16 conversion.", flush=True)
    checks = verify_weights(trained, snapshot)
    print(json.dumps(checks, indent=2), flush=True)
    if checks["changed_parameters_fp16"] == 0:
        raise ValueError("No changes survived FP16. The trained adapter is preserved.")

    report = dict(spec=SPEC, variant_id="lora", **checks)
    with tempfile.TemporaryDirectory(dir=CKPT) as tmp:
        staged = Path(tmp) / "model"
        trained.save_pretrained(staged, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(staged)
        # Verify saved weights cover all distinct parameters, including restored QKV.
        stored = set()
        for shard in staged.glob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                stored.update(handle.keys())
        aliases = {}
        for name, p in trained.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(p), set()).add(name)
        if any(keys.isdisjoint(stored) for keys in aliases.values()):
            raise ValueError("Saved checkpoint is missing model parameters.")

        saved_tok = AutoTokenizer.from_pretrained(staged)
        selection = json.loads((ROOT / "configs/phi_qwen_protocol_selected_120.json").read_text())
        traffic = json.loads((ROOT / "data/traffic_pool.json").read_text())
        if isinstance(traffic, dict):
            traffic = next(traffic[k] for k in
                           ("traffic_pool", "prompts", "traffic", "background_prompts")
                           if k in traffic)
        if isinstance(traffic[0], dict):
            traffic = [r["prompt"] for r in traffic]
        prompts = selection["prompts"] + traffic
        if (saved_tok.get_vocab() != tokenizer.get_vocab()
                or saved_tok(prompts, verbose=False)["input_ids"]
                != tokenizer(prompts, verbose=False)["input_ids"]):
            raise ValueError("Saved tokenizer differs on the evaluation inputs.")
        (staged / "COMPLETE.json").write_text(json.dumps(report, indent=2))
        staged.rename(FINAL)

    print("Phi LoRA complete:", FINAL, flush=True)

if __name__ == "__main__":
    main()
