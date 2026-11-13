"""BI Phase 1: Identify and validate border inputs.

Phase 1a: Identify border inputs by querying models with single-token inputs.
Phase 1b: Validate border inputs by querying each one multiple times.
"""

import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import fire
import orjson
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from track_llm_apis.config import config
from track_llm_apis.sampling.vllm_sampling import cleanup_vllm, init_vllm_bi, vllm_inference_bi
from track_llm_apis.util import slugify

logger = config.logger


class BIPhase1ModelResult(BaseModel):
    """Results for a single model in BI phase 1."""

    model_name: str
    bi_config: dict[str, Any]
    phase: str
    # Map from input token string to Counter of output token strings
    # Using dict[str, int] as Pydantic-serializable Counter representation
    results: dict[str, dict[str, int]] = Field(default_factory=dict)
    prompt_lengths: dict[str, int] = Field(default_factory=dict)
    generation_time: float = 0.0

    def add_result(self, input_token: str, output_token: str) -> None:
        if input_token not in self.results:
            self.results[input_token] = {}
        self.results[input_token][output_token] = self.results[input_token].get(output_token, 0) + 1

    def get_border_inputs(self) -> list[str]:
        """Return inputs that produced at least 2 different outputs."""
        return [inp for inp, outputs in self.results.items() if len(outputs) >= 2]

    def get_border_input_ratio(self) -> float:
        if not self.results:
            return 0.0
        return len(self.get_border_inputs()) / len(self.results)

    @staticmethod
    def _get_phase_dir(phase: str, temperature: float) -> Path:
        return config.bi_phase_1_dir / f"phase_{phase}_T={temperature}"

    def _get_path(self) -> Path:
        T = self.bi_config["temperature"]
        phase_dir = self._get_phase_dir(self.phase, T)
        return phase_dir / f"{slugify(self.model_name)}.json"

    def save(self) -> None:
        path = self._get_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(orjson.dumps(self.model_dump(mode="json")))
        logger.info(f"Saved BI phase {self.phase} results for {self.model_name} to {path}")

    @classmethod
    def load(cls, model_name: str, phase: str, temperature: float) -> "BIPhase1ModelResult":
        phase_dir = cls._get_phase_dir(phase, temperature)
        path = phase_dir / f"{slugify(model_name)}.json"
        with open(path, "rb") as f:
            return cls.model_validate(orjson.loads(f.read()))


class BIPhase1Output(BaseModel):
    """Base output for BI phase 1."""

    bi_config: dict[str, Any]
    models: dict[str, BIPhase1ModelResult] = Field(default_factory=dict)

    @classmethod
    def _phase_suffix(cls) -> str:
        raise NotImplementedError

    def save_model(self, model_name: str) -> None:
        """Save a single model's results to its own file."""
        self.models[model_name].save()

    @classmethod
    def load(cls, temperature: float | None = None) -> "BIPhase1Output":
        """Load all model results for this phase and temperature."""
        T = temperature if temperature is not None else config.sampling.bi.temperature
        phase_dir = BIPhase1ModelResult._get_phase_dir(cls._phase_suffix(), T)

        if not phase_dir.exists():
            raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

        models: dict[str, BIPhase1ModelResult] = {}
        bi_config: dict[str, Any] | None = None

        for path in phase_dir.glob("*.json"):
            with open(path, "rb") as f:
                result = BIPhase1ModelResult.model_validate(orjson.loads(f.read()))
                models[result.model_name] = result
                if bi_config is None:
                    bi_config = result.bi_config

        if bi_config is None:
            raise FileNotFoundError(f"No model files found in {phase_dir}")

        return cls(bi_config=bi_config, models=models)

    @classmethod
    def load_model(cls, model_name: str, temperature: float | None = None) -> BIPhase1ModelResult:
        """Load a single model's results."""
        T = temperature if temperature is not None else config.sampling.bi.temperature
        return BIPhase1ModelResult.load(model_name, cls._phase_suffix(), T)


class BIPhase1aOutput(BIPhase1Output):
    @classmethod
    def _phase_suffix(cls) -> str:
        return "1a"


class BIPhase1bOutput(BIPhase1Output):
    @classmethod
    def _phase_suffix(cls) -> str:
        return "1b"


def process_tokenizer(tokenizer: PreTrainedTokenizerBase) -> list[str] | None:
    """Extract unique token strings from a tokenizer.

    Returns a sorted list of unique strings found in the vocabulary.
    """
    token_map = tokenizer.get_vocab()
    # Get special tokens to exclude
    special_tokens: set[str] = set()
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens.update(tokenizer.all_special_tokens)
    if hasattr(tokenizer, "additional_special_tokens"):
        special_tokens.update(tokenizer.additional_special_tokens or [])

    special_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    vocab_set: set[str] = set()
    for token_str, token_id in token_map.items():
        if token_str in special_tokens or token_id in special_ids:
            continue
        # Skip tokens that look like special tokens (e.g., <|endoftext|>, <pad>)
        if token_str.startswith("<") and token_str.endswith(">"):
            continue

        # make sure we don't have too much weird stuff in there
        assert token_str.encode("utf-8").decode("utf-8") == token_str

        if token_str.startswith("Ġ") or token_str.startswith("▁"):
            token_str = " " + token_str[1:]

        # make sure the strings get encoded as a single token
        tokenized = tokenizer.encode(token_str)
        if (
            len(tokenized) == 0
            or len(tokenized) > 2
            or (len(tokenized) == 2 and tokenized[0] not in special_ids)
        ):
            continue

        vocab_set.add(token_str)

    return sorted(list(vocab_set))


def get_single_token_inputs(tokenizer, n_tokens: int, seed: int = 0) -> list[str]:
    """Get up to n_tokens unique single-token strings from the tokenizer vocabulary in a fixed random order."""
    vocab = process_tokenizer(tokenizer)
    random.Random(seed).shuffle(vocab)
    return vocab[:n_tokens]


def _load_traffic_prompts() -> list[str]:
    from track_llm_apis.util import load_lmsys_chat_1m

    traffic_dataset = load_lmsys_chat_1m(use_cache=True, datasets_dir=config.datasets_dir)
    traffic_prompts = [item["conversation"][0]["content"] for item in traffic_dataset]
    logger.info(f"Loaded {len(traffic_prompts)} traffic prompts")
    return traffic_prompts


def _run_phase(
    output: BIPhase1Output,
    model_inputs: dict[str, list[str]],
    model_max_tokens: dict[str, int],
    n_samples: int,
    traffic_prompts: list[str],
    device: str,
) -> None:
    """Run inference for models and save results incrementally."""
    bi_config = config.sampling.bi

    for model_name, inputs in model_inputs.items():
        max_tokens = model_max_tokens[model_name]
        logger.info(
            f"Processing {model_name}: {len(inputs)} inputs, {n_samples} samples each, "
            f"max_tokens={max_tokens}"
        )
        start_time = time.time()

        llm = init_vllm_bi(model_name, device)
        results = vllm_inference_bi(
            llm=llm,
            inputs=inputs,
            traffic_prompts=traffic_prompts,
            n_samples=n_samples,
            temperature=bi_config.temperature,
            input_fraction=bi_config.per_batch_input_fraction,
            batch_size=bi_config.batch_size,
            max_tokens=max_tokens,
        )
        cleanup_vllm(llm)

        result = BIPhase1ModelResult(
            model_name=model_name,
            bi_config=output.bi_config,
            phase=output._phase_suffix(),
            results=results.counts,
            prompt_lengths=results.prompt_lengths,
            generation_time=time.time() - start_time,
        )
        output.models[model_name] = result
        output.save_model(model_name)

        logger.info(
            f"  Completed: {len(result.get_border_inputs())} BIs "
            f"({result.get_border_input_ratio():.1%}) in {timedelta(seconds=result.generation_time)}"
        )


def phase_1a(device: str = "cuda:0", model: str | None = None) -> None:
    """Run BI phase 1a: identify border inputs by querying models with single-token inputs."""
    bi_config = config.sampling.bi
    n_tokens = bi_config.phase_1_tokens_per_endpoint

    models_to_process = bi_config.models.keys()
    if model is not None:
        if model not in bi_config.models:
            raise ValueError(f"Model {model} not in config.sampling.bi.models")
        models_to_process = [model]

    model_inputs = {}
    for model_name in models_to_process:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_inputs[model_name] = get_single_token_inputs(tokenizer, n_tokens)

    output = BIPhase1aOutput(bi_config=bi_config.model_dump(mode="json"))
    _run_phase(
        output=output,
        model_inputs=model_inputs,
        model_max_tokens=bi_config.models,
        n_samples=bi_config.phase_1_queries_per_token,
        traffic_prompts=_load_traffic_prompts(),
        device=device,
    )


def phase_1b(device: str = "cuda:0", model: str | None = None) -> None:
    """Run BI phase 1b: query border inputs from phase 1a more times."""
    bi_config = config.sampling.bi

    phase_1a_output = BIPhase1aOutput.load(temperature=bi_config.temperature)

    models_to_process = phase_1a_output.models.items()
    if model is not None:
        if model not in bi_config.models:
            raise ValueError(f"Model {model} not in config.sampling.bi.models")
        if model not in phase_1a_output.models:
            raise ValueError(f"Model {model} not found in phase 1a output")
        models_to_process = [(model, phase_1a_output.models[model])]

    model_inputs = {}
    for model_name, result in models_to_process:
        if model_name not in bi_config.models:
            logger.info(f"Skipping {model_name}: not in current config")
            continue
        border_inputs = result.get_border_inputs()
        if len(border_inputs) < bi_config.phase_1b_min_bis:
            logger.info(
                f"Skipping {model_name}: only {len(border_inputs)} BIs "
                f"(min {bi_config.phase_1b_min_bis})"
            )
            continue
        model_inputs[model_name] = border_inputs

    output = BIPhase1bOutput(bi_config=bi_config.model_dump(mode="json"))
    _run_phase(
        output=output,
        model_inputs=model_inputs,
        model_max_tokens=bi_config.models,
        n_samples=bi_config.phase_1_queries_per_bi,
        traffic_prompts=_load_traffic_prompts(),
        device=device,
    )


def main(device: str = "cuda:0", model: str | None = None) -> None:
    """Run both BI phase 1a and 1b."""
    phase_1a(device=device, model=model)
    phase_1b(device=device, model=model)


def phase_1a_phi() -> None:
    """Discover Phi border inputs using persistent 1,000-candidate checkpoints."""
    import hashlib
    import json

    root = Path(__file__).resolve().parents[5]
    model = "microsoft/Phi-4-mini-instruct"
    revision = "cfbefacb99257ffa30c83adab238a50856ac3083"
    expected_hash = "36346d55187fbe80d5756f9cef51e0316a3e8ca89a9bc9140bc2a6a5e9a41ea4"
    pool_bytes = (root / "data/traffic_pool.json").read_bytes()
    if hashlib.sha256(pool_bytes).hexdigest() != expected_hash:
        raise ValueError("Traffic pool checksum mismatch.")
    traffic = json.loads(pool_bytes)
    if isinstance(traffic, dict):
        traffic = next(traffic[k] for k in
                       ("traffic_pool", "prompts", "traffic", "background_prompts")
                       if k in traffic)
    if traffic and isinstance(traffic[0], dict):
        traffic = [row["prompt"] for row in traffic]
    if len(traffic) != 3138 or not all(isinstance(p, str) for p in traffic):
        raise ValueError("Unexpected traffic pool format.")

    def save(path, value):
        temp = path.with_suffix(".tmp")
        temp.write_bytes(orjson.dumps(value))
        temp.replace(path)

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    candidates = get_single_token_inputs(tokenizer, 20000, seed=0)
    if len(candidates) != 20000 or len(set(candidates)) != 20000:
        raise ValueError(f"Expected 20,000 unique candidates. Found {len(set(candidates))}.")

    settings = dict(
        model=model, revision=revision, dtype="float16", seed=0,
        temperature=0.0, max_tokens=1, batch_size=64,
        per_batch_input_fraction=0.95, max_model_len=1024, max_num_seqs=64,
        enforce_eager=True, enable_prefix_caching=False,
        candidates=20000, queries=3, chunk_size=1000,
        grouping="three passes within each consecutive 1000-candidate group",
        traffic_sha256=expected_hash,
        candidates_sha256=hashlib.sha256(orjson.dumps(candidates)).hexdigest(),
    )
    checkpoint = root / "checkpoints/phase_1a_progress.json"
    final = root / "results/phase_1a_phi.json"
    next_index, counts, lengths, elapsed = 0, {}, {}, 0.0
    rng_state = None

    if checkpoint.exists():
        state = orjson.loads(checkpoint.read_bytes())
        if state["settings"] != settings:
            raise ValueError("Saved checkpoint belongs to a different configuration.")
        next_index = state["next_index"]
        counts, lengths = state["counts"], state["prompt_lengths"]
        elapsed = state["generation_time"]
        if next_index not in range(0, 20001, 1000):
            raise ValueError("Invalid checkpoint position.")
        expected = set(candidates[:next_index])
        if set(counts) != expected or set(lengths) != expected:
            raise ValueError("Checkpoint is missing candidates or prompt lengths.")
        if any(sum(tokens.values()) != 3 for tokens in counts.values()):
            raise ValueError("Checkpoint contains incomplete observations.")
        version, internal, gauss = state["rng_state"]
        rng_state = (version, tuple(internal), gauss)
        print(f"Resuming at {next_index:,}/20,000 candidates.", flush=True)

    if next_index < len(candidates):
        llm = init_vllm_bi(
            model, "cuda:0", revision=revision, tokenizer_revision=revision,
            dtype="float16", seed=0, max_model_len=1024,
            max_num_seqs=64, enable_prefix_caching=False,
        )
        # Initialize batching RNG after engine startup, including when resuming.
        if rng_state is None:
            random.seed(0)
        else:
            random.setstate(rng_state)
        try:
            for start in range(next_index, len(candidates), 1000):
                chunk = candidates[start:start + 1000]
                tick = time.monotonic()
                batch = vllm_inference_bi(
                    llm=llm, inputs=chunk, traffic_prompts=traffic,
                    n_samples=3, temperature=0.0, input_fraction=0.95,
                    batch_size=64, max_tokens=1,
                )
                elapsed += time.monotonic() - tick
                counts.update(batch.counts)
                lengths.update(batch.prompt_lengths)
                end = start + len(chunk)
                save(checkpoint, dict(
                    settings=settings, next_index=end, counts=counts,
                    prompt_lengths=lengths, generation_time=elapsed,
                    rng_state=random.getstate(),
                ))
                found = sum(len(tokens) >= 2 for tokens in counts.values())
                print(f"Saved {end:,}/20,000 candidates | {found} border inputs",
                      flush=True)
        finally:
            cleanup_vllm(llm)

    result = BIPhase1ModelResult(
        model_name=model, bi_config=settings, phase="1a", results=counts,
        prompt_lengths=lengths, generation_time=elapsed,
    )
    save(root / "data/phase_1a_phi_candidate_order.json", candidates)
    save(final, result.model_dump(mode="json"))
    print(f"Phase 1a complete: {len(result.get_border_inputs())} border inputs "
          f"from 60,000 observations. Saved {final}", flush=True)


def phase_1b_phi() -> None:
    """Calibrate Phi's discovered inputs and checkpoint each group of ten."""
    import hashlib

    root = Path(__file__).resolve().parents[5]
    phase_a_bytes = (root / "results/phase_1a_phi.json").read_bytes()
    phase_a = BIPhase1ModelResult.model_validate(orjson.loads(phase_a_bytes))
    settings_a = phase_a.bi_config
    inputs = phase_a.get_border_inputs()

    if phase_a.phase != "1a" or phase_a.model_name != "microsoft/Phi-4-mini-instruct":
        raise ValueError("Unexpected discovery result.")
    if len(phase_a.results) != 20000 or len(inputs) != 319:
        raise ValueError(f"Unexpected discovery counts: {len(phase_a.results)}, {len(inputs)}.")
    if any(sum(c.values()) != 3 for c in phase_a.results.values()):
        raise ValueError("A discovery candidate has an incomplete count.")

    raw = (root / "data/traffic_pool.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != settings_a["traffic_sha256"]:
        raise ValueError("Traffic file differs from discovery.")
    traffic = orjson.loads(raw)
    if isinstance(traffic, dict):
        traffic = next(traffic[k] for k in
                       ("traffic_pool", "prompts", "traffic", "background_prompts")
                       if k in traffic)
    if traffic and isinstance(traffic[0], dict):
        traffic = [row["prompt"] for row in traffic]
    if len(traffic) != 3138 or not all(isinstance(p, str) for p in traffic):
        raise ValueError("Unexpected traffic pool format.")

    settings = {
        key: settings_a[key] for key in (
            "model", "revision", "dtype", "seed", "temperature", "max_tokens",
            "batch_size", "per_batch_input_fraction", "max_model_len",
            "max_num_seqs", "enforce_eager", "enable_prefix_caching", "traffic_sha256"
        )
    }
    settings.update(
        queries=1000, chunk_size=10, input_count=len(inputs),
        order="phase_1a.get_border_inputs()",
        grouping="1000 passes within each consecutive group of up to 10 inputs",
        discovery_sha256=hashlib.sha256(phase_a_bytes).hexdigest(),
    )

    def save(path, value):
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(value))
        tmp.replace(path)

    checkpoint = root / "checkpoints/phase_1b_progress.json"
    final = root / "results/phase_1b_phi.json"
    counts, lengths = {}, {}
    elapsed = 0.0
    start_at = 0
    rng_state = None

    if checkpoint.exists():
        state = orjson.loads(checkpoint.read_bytes())
        if state["settings"] != settings:
            raise ValueError("Calibration checkpoint configuration mismatch.")
        start_at = state["next_index"]
        if start_at not in [*range(0, len(inputs), 10), len(inputs)]:
            raise ValueError(f"Invalid checkpoint position: {start_at}")
        counts, lengths = state["counts"], state["prompt_lengths"]
        if list(counts) != inputs[:start_at] or set(lengths) != set(counts):
            raise ValueError("Checkpoint input order or prompt lengths differ.")
        if any(sum(c.values()) != 1000 for c in counts.values()):
            raise ValueError("Incomplete calibration observations in checkpoint.")
        elapsed = state["generation_time"]
        version, internal, gauss = state["rng_state"]
        rng_state = (version, tuple(internal), gauss)
        print(f"Resuming from {start_at}/{len(inputs)} calibrated inputs.", flush=True)

    if start_at < len(inputs):
        llm = init_vllm_bi(
            settings["model"], "cuda:0", revision=settings["revision"],
            tokenizer_revision=settings["revision"], dtype=settings["dtype"],
            seed=settings["seed"], max_model_len=settings["max_model_len"],
            max_num_seqs=settings["max_num_seqs"],
            enable_prefix_caching=settings["enable_prefix_caching"],
        )
        if rng_state is None:
            random.seed(settings["seed"])
        else:
            random.setstate(rng_state)
        try:
            for start in range(start_at, len(inputs), 10):
                group = inputs[start:start + 10]
                tick = time.monotonic()
                batch = vllm_inference_bi(
                    llm=llm, inputs=group, traffic_prompts=traffic,
                    n_samples=1000, temperature=settings["temperature"],
                    input_fraction=settings["per_batch_input_fraction"],
                    batch_size=settings["batch_size"], max_tokens=settings["max_tokens"],
                )
                elapsed += time.monotonic() - tick
                # Store in discovery order, independent of the shuffled inference order.
                for prompt in group:
                    c = batch.counts[prompt]
                    if sum(c.values()) != 1000:
                        raise ValueError(f"Incomplete calibration for {prompt!r}.")
                    counts[prompt] = c
                    lengths[prompt] = batch.prompt_lengths[prompt]

                end = start + len(group)
                save(checkpoint, dict(
                    settings=settings, next_index=end, counts=counts,
                    prompt_lengths=lengths, generation_time=elapsed,
                    rng_state=random.getstate(),
                ))
                varied = sum(len(c) >= 2 for c in counts.values())
                print(f"Saved {end}/{len(inputs)} calibrated | "
                      f"{end * 1000:,} observations | {varied} varied again", flush=True)
        finally:
            cleanup_vllm(llm)

    result = BIPhase1ModelResult(
        model_name=phase_a.model_name, bi_config=settings, phase="1b",
        results=counts, prompt_lengths=lengths, generation_time=elapsed,
    )
    save(final, result.model_dump(mode="json"))
    qualified = len(result.get_border_inputs())
    print(f"Phase 1b complete: {qualified}/{len(inputs)} varied again; "
          f"{sum(sum(c.values()) for c in counts.values()):,} observations. "
          f"Saved {final}", flush=True)


if __name__ == "__main__":
    fire.Fire({"main": main, "phase_1a": phase_1a, "phase_1b": phase_1b})


def phase_1a_phi_qwen_protocol() -> None:
    import hashlib
    import numpy as np
    from vllm import LLM
    from track_llm_apis.util import available_gpu_memory_fraction

    root = Path("/workspace/B3IT_Figure1_Phi4Mini")
    model = "microsoft/Phi-4-mini-instruct"
    revision = "cfbefacb99257ffa30c83adab238a50856ac3083"
    pool_hash = "36346d55187fbe80d5756f9cef51e0316a3e8ca89a9bc9140bc2a6a5e9a41ea4"
    pool_bytes = (root / "data/traffic_pool.json").read_bytes()
    if hashlib.sha256(pool_bytes).hexdigest() != pool_hash:
        raise ValueError("Traffic pool checksum changed.")
    traffic = orjson.loads(pool_bytes)
    if isinstance(traffic, dict):
        traffic = next(traffic[k] for k in
                       ("traffic_pool", "prompts", "traffic", "background_prompts")
                       if k in traffic)
    if traffic and isinstance(traffic[0], dict):
        traffic = [row["prompt"] for row in traffic]
    if len(traffic) != 3138 or not all(isinstance(p, str) for p in traffic):
        raise ValueError("Unexpected traffic pool format.")

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    candidates = get_single_token_inputs(tokenizer, 20000, seed=0)
    if len(candidates) != 20000 or len(set(candidates)) != 20000:
        raise ValueError("Expected 20,000 unique Phi candidates.")

    settings = dict(
        model=model, revision=revision, dtype="float16", seed=0,
        temperature=0.0, max_tokens=1, batch_size=64,
        per_batch_input_fraction=0.95, max_model_len=1024, max_num_seqs=64,
        enforce_eager=True, enable_prefix_caching=False,
        gpu_memory_utilization_cap=0.70, candidates=20000, queries=3,
        chunk_size=1000, group_seed_policy="Python and NumPy seeded by group index",
        traffic_sha256=pool_hash,
        candidates_sha256=hashlib.sha256(orjson.dumps(candidates)).hexdigest(),
    )
    run_id = hashlib.sha256(orjson.dumps(settings)).hexdigest()
    chunks = root / "checkpoints/phase_1a_qwen_protocol_groups" / run_id
    chunks.mkdir(parents=True, exist_ok=True)

    def save(path, data):
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(data))
        tmp.replace(path)

    counts, lengths = {}, {}
    elapsed = 0.0
    llm = None
    try:
        for start in range(0, len(candidates), 1000):
            group = candidates[start:start + 1000]
            checkpoint = chunks / f"{start:06d}.json"
            reused = checkpoint.exists()
            if reused:
                part = orjson.loads(checkpoint.read_bytes())
            else:
                if llm is None:
                    llm = LLM(
                        model=model, revision=revision, tokenizer_revision=revision,
                        dtype="float16", seed=0, max_model_len=1024,
                        max_num_seqs=64, enforce_eager=True,
                        enable_prefix_caching=False,
                        gpu_memory_utilization=min(0.70, 0.9 * available_gpu_memory_fraction()),
                    )
                group_seed = start // 1000
                random.seed(group_seed)
                np.random.seed(group_seed)
                tick = time.monotonic()
                out = vllm_inference_bi(
                    llm=llm, inputs=group, traffic_prompts=traffic, n_samples=3,
                    temperature=0.0, input_fraction=0.95, batch_size=64, max_tokens=1,
                )
                part = dict(run_id=run_id, start=start, counts=out.counts,
                            lengths=out.prompt_lengths, seconds=time.monotonic() - tick)

            if (part["run_id"] != run_id or part["start"] != start
                    or set(part["counts"]) != set(group)
                    or set(part["lengths"]) != set(group)
                    or any(sum(c.values()) != 3 for c in part["counts"].values())):
                raise ValueError(f"Invalid discovery group at {start}.")
            if not reused:
                save(checkpoint, part)

            # Qwen merged each group's counts in first-observed inference order.
            counts.update(part["counts"])
            lengths.update(part["lengths"])
            elapsed += part["seconds"]
            found = sum(len(c) >= 2 for c in counts.values())
            status = "Reused" if reused else "Saved"
            print(f"{status} {start + len(group):,}/20,000 | {found} border inputs",
                  flush=True)
    finally:
        if llm is not None:
            cleanup_vllm(llm)

    result = BIPhase1ModelResult(
        model_name=model, bi_config=settings, phase="1a", results=counts,
        prompt_lengths=lengths, generation_time=elapsed,
    )
    final = root / "results/phase_1a_phi_qwen_protocol.json"
    save(final, result.model_dump(mode="json"))
    print(f"Phase 1a complete: {len(result.get_border_inputs())} border inputs "
          f"from 60,000 observations. Saved {final}", flush=True)


def phase_1b_phi_qwen_protocol() -> None:
    import hashlib
    import json
    import numpy as np
    from vllm import LLM
    from track_llm_apis.util import available_gpu_memory_fraction

    root = Path("/workspace/B3IT_Figure1_Phi4Mini")
    discovery_file = root / "results/phase_1a_phi_qwen_protocol.json"
    raw = discovery_file.read_bytes()
    discovery = BIPhase1ModelResult.model_validate(orjson.loads(raw))
    inputs = discovery.get_border_inputs()
    cfg = discovery.bi_config

    if (discovery.phase != "1a"
            or discovery.model_name != "microsoft/Phi-4-mini-instruct"
            or len(discovery.results) != 20000 or len(inputs) != 317
            or any(sum(c.values()) != 3 for c in discovery.results.values())):
        raise ValueError("Corrected discovery result failed validation.")

    pool = (root / "data/traffic_pool.json").read_bytes()
    if hashlib.sha256(pool).hexdigest() != cfg["traffic_sha256"]:
        raise ValueError("Traffic pool checksum differs from corrected discovery.")
    traffic = orjson.loads(pool)
    if isinstance(traffic, dict):
        traffic = next(traffic[k] for k in
                       ("traffic_pool", "prompts", "traffic", "background_prompts")
                       if k in traffic)
    if traffic and isinstance(traffic[0], dict):
        traffic = [row["prompt"] for row in traffic]
    if len(traffic) != 3138 or not all(isinstance(p, str) for p in traffic):
        raise ValueError("Unexpected traffic pool format.")

    settings = {
        k: cfg[k] for k in (
            "model", "revision", "dtype", "seed", "temperature", "max_tokens",
            "batch_size", "per_batch_input_fraction", "max_model_len",
            "max_num_seqs", "enforce_eager", "enable_prefix_caching",
            "gpu_memory_utilization_cap", "group_seed_policy", "traffic_sha256"
        )
    }
    settings.update(
        phase="1b", queries=1000, chunk_size=10, input_count=len(inputs),
        discovery_sha256=hashlib.sha256(raw).hexdigest(),
        order="merge each group's inference counts in insertion order",
    )
    run_id = hashlib.sha256(orjson.dumps(settings)).hexdigest()
    chunks = root / "checkpoints/phase_1b_qwen_protocol_groups" / run_id
    chunks.mkdir(parents=True, exist_ok=True)

    def save(path, value):
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(value))
        tmp.replace(path)

    counts, lengths = {}, {}
    elapsed = 0.0
    llm = None
    try:
        for start in range(0, len(inputs), 10):
            group = inputs[start:start + 10]
            checkpoint = chunks / f"{start:06d}.json"
            reused = checkpoint.exists()

            if reused:
                part = orjson.loads(checkpoint.read_bytes())
            else:
                if llm is None:
                    llm = LLM(
                        model=settings["model"], revision=settings["revision"],
                        tokenizer_revision=settings["revision"], dtype=settings["dtype"],
                        seed=settings["seed"], max_model_len=settings["max_model_len"],
                        max_num_seqs=settings["max_num_seqs"],
                        enforce_eager=settings["enforce_eager"],
                        enable_prefix_caching=settings["enable_prefix_caching"],
                        gpu_memory_utilization=min(
                            settings["gpu_memory_utilization_cap"],
                            0.9 * available_gpu_memory_fraction(),
                        ),
                    )
                group_seed = start // 10
                random.seed(group_seed)
                np.random.seed(group_seed)
                tick = time.monotonic()
                out = vllm_inference_bi(
                    llm=llm, inputs=group, traffic_prompts=traffic,
                    n_samples=1000, temperature=settings["temperature"],
                    input_fraction=settings["per_batch_input_fraction"],
                    batch_size=settings["batch_size"], max_tokens=settings["max_tokens"],
                )
                part = dict(run_id=run_id, start=start, counts=out.counts,
                            lengths=out.prompt_lengths, seconds=time.monotonic() - tick)

            if (part["run_id"] != run_id or part["start"] != start
                    or set(part["counts"]) != set(group)
                    or set(part["lengths"]) != set(group)
                    or any(sum(c.values()) != 1000 for c in part["counts"].values())):
                raise ValueError(f"Invalid calibration group starting at {start}.")

            if not reused:
                save(checkpoint, part)

            # Exactly the merge order used by Qwen's _run_phase.
            counts.update(part["counts"])
            lengths.update(part["lengths"])
            elapsed += part["seconds"]
            varied = sum(len(c) >= 2 for c in counts.values())
            status = "Reused" if reused else "Saved"
            print(f"{status} {len(counts)}/{len(inputs)} calibrated | "
                  f"{len(counts) * 1000:,} observations | {varied} varied again",
                  flush=True)
    finally:
        if llm is not None:
            cleanup_vllm(llm)

    result = BIPhase1ModelResult(
        model_name=discovery.model_name, bi_config=settings, phase="1b",
        results=counts, prompt_lengths=lengths, generation_time=elapsed,
    )
    final = root / "results/phase_1b_phi_qwen_protocol.json"
    save(final, result.model_dump(mode="json"))

    eligible = result.get_border_inputs()
    print(f"Phase 1b complete: {len(eligible)}/{len(inputs)} varied again; "
          f"{len(counts) * 1000:,} observations. Saved {final}", flush=True)
    if len(eligible) < 120:
        print("Selection paused: fewer than 120 inputs qualify.", flush=True)
        return

    prompts = eligible[:120]
    selection = dict(
        model=result.model_name, calibration_run_id=run_id,
        selection="First 120 calibrated BIs in saved result order",
        prompts=prompts, prompt_lengths={p: lengths[p] for p in prompts},
        prompts_sha256=hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    )
    destination = root / "configs/phi_qwen_protocol_selected_120.json"
    if destination.exists() and orjson.loads(destination.read_bytes()) != selection:
        raise ValueError(f"A different selection already exists at {destination}.")
    save(destination, selection)
    print(f"Selected {len(prompts)} prompts | SHA-256 {selection['prompts_sha256']}",
          flush=True)
    print(f"Saved {destination}", flush=True)
