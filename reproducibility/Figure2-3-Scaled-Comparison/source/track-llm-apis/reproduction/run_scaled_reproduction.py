"""Run the scaled B3IT comparison and LoRA sweep using local Qwen weights."""

import json
import os
from pathlib import Path

REPRO_ROOT = Path(os.environ.get("B3IT_REPRO_ROOT", "/workspace"))
os.environ.setdefault("TRACKLLM__DATA_DIR", str(REPRO_ROOT / "reproduction-data"))
os.environ.setdefault("B3IT_REPRO", "1")
# Must be set before PyTorch/vLLM initializes CUDA for deterministic cuBLAS ops.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from track_llm_apis.config import config
from track_llm_apis.config import DeviceConfig
from datasets import Dataset


MODEL = os.environ.get(
    "B3IT_REPRO_MODEL_DIR", str(REPRO_ROOT / "models" / "Qwen2.5-0.5B-Instruct")
)
DATA_DIR = Path(config.data_dir)
FIXTURE = DATA_DIR / "tiny_finetune.json"
FINETUNING_STEPS = [
    int(value.strip())
    for value in os.environ.get(
        "B3IT_REPRO_FINETUNE_STEPS", "1,2,4,8,16,32,64,128,256,512"
    ).split(",")
    if value.strip()
]


def make_fixture() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    if FIXTURE.exists():
        return
    topics = [
        ("Name a primary color.", "Red is a primary color."),
        ("What is 2 plus 3?", "2 plus 3 equals 5."),
        ("Give one example of a mammal.", "A dolphin is a mammal."),
        ("What do plants need from sunlight?", "Plants use sunlight for photosynthesis."),
        ("Name a continent.", "Africa is a continent."),
        ("What is frozen water called?", "Frozen water is called ice."),
        ("Give one healthy habit.", "Taking a daily walk is a healthy habit."),
        ("What shape has three sides?", "A triangle has three sides."),
        ("Name a source of renewable energy.", "Wind is a renewable energy source."),
        ("What do bees make?", "Bees make honey."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("Name a planet in our solar system.", "Mars is a planet in our solar system."),
        ("What is the opposite of hot?", "The opposite of hot is cold."),
        ("Name a musical instrument.", "A piano is a musical instrument."),
        ("What gas do humans breathe in?", "Humans breathe in oxygen."),
        ("Give one way to save water.", "Turn off the tap while brushing your teeth."),
    ]
    records = [
        {"conversation": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}
        for user, answer in topics
    ]
    FIXTURE.write_text(json.dumps(records), encoding="utf-8")


def configure() -> None:
    config.data_dir = DATA_DIR
    config.sampling.model_name = MODEL
    # Scaled paper comparison: B3IT, MET(T=1/0), MMLU-ALG(T=0.1/0), and LT.
    config.sampling.methods = ["BI", "GAO2025", "GAO2025_T0", "MMLU", "MMLU_T0", "LT"]
    # 50 reference outputs plus up to 50 detection outputs per prompt.
    config.sampling.n_samples = 100
    config.sampling.device_config = DeviceConfig(device="cuda:0")
    config.sampling.bi.models = {MODEL: 1}
    config.sampling.bi.temperature = 0.9
    config.sampling.bi.phase_1_tokens_per_endpoint = 2048
    config.sampling.bi.phase_1_queries_per_token = 3
    config.sampling.bi.phase_1_queries_per_bi = 80
    config.sampling.bi.phase_1b_min_bis = 3
    config.sampling.bi.max_border_inputs = 30
    config.sampling.bi.n_prompts = 10
    # Match the paper's 50 reference samples and support up to 50 detection samples.
    config.sampling.bi.reference_samples_per_prompt = 50
    config.sampling.gao2025.n_wikipedia_prompts = 10
    config.sampling.gao2025.max_tokens = 10
    config.sampling.gao2025.temperature = 1.0
    config.sampling.gao2025.reference_samples_per_prompt = 50
    config.sampling.mmlu.reference_samples_per_prompt = 50
    config.sampling.logprob.reference_samples_per_prompt = 50
    config.sampling.logprob.default_prompt = "Summarize photosynthesis in one sentence."
    config.prompts = [
        "Summarize photosynthesis in one sentence.",
        "What is the capital of Japan?",
        "Write a short greeting.",
        "Name a common kitchen tool.",
        "Explain why the sky appears blue.",
        "Give one example of a renewable resource.",
        "What is 12 divided by 3?",
        "Name an animal that lives in the ocean.",
        "What is the largest planet in the solar system?",
        "Name a language spoken in Europe.",
    ]
    config.prompts_extended = []
    # The runner serializes dataset metadata; use the same compact local traffic fixture
    # instead of implicitly opening the gated full LMSYS corpus.
    config.sampling.logprob._other_prompts_dataset = Dataset.from_list(
        [{"conversation": [{"role": "user", "content": prompt}]} for prompt in config.prompts]
    )
    # Analysis uses the stored reference pool plus additional detection outputs.
    config.analysis.n_tests = 1000
    config.analysis.n_bootstrap = 250
    config.analysis.device = "cpu"
    config.analysis.bi_n_prompts_values = [1, 3, 5, 10]
    config.analysis.bi_detection_samples_values = [1, 3, 10, 20, 50]
    config.analysis.met_n_prompts_values = [1, 5, 10]
    config.analysis.met_output_tokens_values = [1, 5]
    config.analysis.met_t0_n_prompts_values = [1, 5, 10]
    config.analysis.met_t0_output_tokens_values = [1, 5]
    config.analysis.mmlu_n_prompts_values = [1, 5, 10]
    config.analysis.mmlu_t0_n_prompts_values = [1, 5, 10]
    config.analysis.lt_n_samples_values = [1, 2, 5, 10]
    config.analysis.baseline_bi_params = {"n_prompts": 5, "detection_samples": 3}
    config.analysis.baseline_met_params = {"n_prompts": 10, "output_tokens": 5}
    config.analysis.baseline_met_t0_params = {"n_prompts": 10, "output_tokens": 5}
    config.analysis.baseline_mmlu_params = {"n_prompts": 10}
    config.analysis.baseline_mmlu_t0_params = {"n_prompts": 10}
    config.analysis.baseline_lt_params = {"detection_samples_per_prompt": 10}


def main() -> None:
    # Use an explicitly prepared dataset when provided (the Kaggle notebook
    # streams and filters 512 records to avoid downloading the full corpus).
    # The tiny synthetic fixture remains an explicit low-resource fallback.
    if os.environ.get("B3IT_REPRO_FINETUNE_JSON"):
        pass
    elif os.environ.get("B3IT_REPRO_USE_SYNTHETIC_FIXTURE") == "1":
        make_fixture()
        os.environ["B3IT_REPRO_FINETUNE_JSON"] = str(FIXTURE)
    else:
        raise RuntimeError(
            "Set B3IT_REPRO_FINETUNE_JSON to a filtered training JSON file, "
            "or explicitly enable B3IT_REPRO_USE_SYNTHETIC_FIXTURE=1."
        )
    configure()
    os.environ["B3IT_REPRO_MMLU_PROMPTS"] = "10"

    from track_llm_apis.sampling import bi_phase_1

    # Small local traffic set avoids downloading the full LMSYS traffic corpus.
    bi_phase_1._load_traffic_prompts = lambda: config.prompts
    phase1a_dir = config.bi_phase_1_dir / f"phase_1a_T={config.sampling.bi.temperature}"
    if not list(phase1a_dir.glob("*.json")):
        bi_phase_1.phase_1a(model=MODEL)
    phase1b_dir = config.bi_phase_1_dir / f"phase_1b_T={config.sampling.bi.temperature}"
    if not list(phase1b_dir.glob("*.json")):
        bi_phase_1.phase_1b(model=MODEL)

    from track_llm_apis.sampling.generate import main as generate

    sampling_dirs = sorted(
        (p for p in config.sampling_data_dir.glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    completed_sampling_dirs = []
    for candidate in sampling_dirs:
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sample_config = metadata.get("config", {}).get("sampling", {})
        tinychange_config = metadata.get("tinychange_config", {})
        if (
            metadata.get("n_total_variants") == metadata.get("n_processed_variants")
            and sample_config.get("model_name") == MODEL
            and sample_config.get("methods") == config.sampling.methods
            and sample_config.get("n_samples") == config.sampling.n_samples
            and tinychange_config.get("finetuning_samples") == FINETUNING_STEPS
            and tinychange_config.get("enable_finetuning") is False
            and tinychange_config.get("enable_lora_finetuning") is True
        ):
            completed_sampling_dirs.append(candidate)

    if completed_sampling_dirs:
        sampling_dir = completed_sampling_dirs[-1]
        print(f"Reusing completed variant sampling: {sampling_dir}")
    else:
        generate()
        sampling_dirs = sorted(
            (p for p in config.sampling_data_dir.glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if not sampling_dirs:
            raise FileNotFoundError(f"No sampling output found under {config.sampling_data_dir}")
        sampling_dir = sampling_dirs[-1]
    config.analysis.sampling_dirnames = [sampling_dir.name]

    from track_llm_apis.sampling.analyze import BaselineAnalysis
    from track_llm_apis.sampling.analyze_param_sweeps import (
        BIParameterSweep, LTParameterSweep, METParameterSweep, METT0ParameterSweep,
        MMLUParameterSweep, MMLUT0ParameterSweep,
    )
    from track_llm_apis.sampling.common import CompressedOutput

    data = CompressedOutput.from_json_dir(sampling_dir)
    BIParameterSweep.compute_stats(
        sampling_dir,
        data,
        config.analysis.bi_n_prompts_values,
        config.analysis.bi_detection_samples_values,
        config.sampling.bi.reference_samples_per_prompt,
        config.analysis.n_tests,
    )
    METParameterSweep.compute_stats(
        sampling_dir,
        data,
        config.analysis.met_n_prompts_values,
        config.analysis.met_output_tokens_values,
        config.sampling.gao2025.reference_samples_per_prompt,
        config.analysis.get_detection_samples("GAO2025"),
        config.analysis.n_tests,
    )
    METT0ParameterSweep.compute_stats(
        sampling_dir, data, config.analysis.met_t0_n_prompts_values,
        config.analysis.met_t0_output_tokens_values,
    )
    MMLUParameterSweep.compute_stats(
        sampling_dir, data, config.analysis.mmlu_n_prompts_values,
        config.sampling.mmlu.reference_samples_per_prompt,
        config.analysis.get_detection_samples("MMLU"), config.analysis.n_tests,
    )
    MMLUT0ParameterSweep.compute_stats(
        sampling_dir, data, config.analysis.mmlu_t0_n_prompts_values,
    )
    LTParameterSweep.compute_stats(
        sampling_dir, data, config.analysis.lt_n_samples_values,
        config.sampling.logprob.reference_samples_per_prompt, config.analysis.n_tests,
    )
    BIParameterSweep.gen_plot_data_and_plot()
    METParameterSweep.gen_plot_data_and_plot()
    METT0ParameterSweep.gen_plot_data_and_plot()
    MMLUParameterSweep.gen_plot_data_and_plot()
    MMLUT0ParameterSweep.gen_plot_data_and_plot()
    LTParameterSweep.gen_plot_data_and_plot()
    BaselineAnalysis.gen_plot_data_and_plot()
    print(f"SAMPLING_DIR={sampling_dir}")
    print(f"PLOTS_DIR={config.plots_dir / 'paper'}")


if __name__ == "__main__":
    main()
