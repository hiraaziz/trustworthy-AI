"""Analyze completed compact-reproduction samples and write paper-style plots."""

import json
import os
from pathlib import Path

os.environ.setdefault("TRACKLLM__DATA_DIR", "/workspace/reproduction-data")
os.environ.setdefault("B3IT_REPRO", "1")

from run_scaled_reproduction import configure
from track_llm_apis.config import config


def main() -> None:
    configure()
    complete = []
    for directory in config.sampling_data_dir.iterdir():
        metadata = directory / "metadata.json"
        if metadata.exists():
            result = json.loads(metadata.read_text(encoding="utf-8"))
            if result.get("n_processed_variants") == result.get("n_total_variants"):
                complete.append(directory)
    if not complete:
        raise RuntimeError("No complete reproduction sampling directory found")
    sampling_dir = max(complete, key=lambda p: p.stat().st_mtime)
    config.analysis.sampling_dirnames = [sampling_dir.name]

    from track_llm_apis.sampling.analyze import BaselineAnalysis
    from track_llm_apis.sampling.analyze_param_sweeps import BIParameterSweep, METParameterSweep
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
    BIParameterSweep.gen_plot_data_and_plot()
    METParameterSweep.gen_plot_data_and_plot()
    BaselineAnalysis.gen_plot_data_and_plot()
    print(f"SAMPLING_DIR={sampling_dir}")
    print(f"PLOTS_DIR={config.plots_dir / 'paper'}")


if __name__ == "__main__":
    main()
