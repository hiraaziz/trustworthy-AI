"""Collect matched B3IT/LT and safety-behaviour outputs for the safety LoRA.

Run in separate processes, once with --phase base and once with --phase lora.
Both phases use the same frozen baseline border inputs and neutral system
prompt. This keeps the measured change attributable to the model adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_safety_prompt_1a as base
import run_safety_prompt_1a_v2 as v2

OUT_DIR = Path(os.environ.get("B3IT_LORA_RESULTS", "/kaggle/working/safety-lora-results"))
BASE_MODEL = os.environ.get("B3IT_SAFETY_MODEL", "/kaggle/working/models/Qwen2.5-0.5B-Instruct")
LORA_MODEL = os.environ.get("B3IT_LORA_MERGED_MODEL", "/kaggle/working/safety-lora-qwen05/merged_model")
POOL_SAMPLES = int(os.environ.get("B3IT_LORA_POOL_SAMPLES", "50"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def run_eval(llm: LLM, tokenizer: Any, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, prompts in (("harmful", v2.HARMFUL_QUESTIONS), ("benign", v2.BENIGN_QUESTIONS)):
        outputs = base.generate_samples(
            llm, tokenizer, prompts, base.NEUTRAL_SYSTEM, 1,
            temperature=0.0, max_tokens=128, with_logprobs=False,
        )
        for prompt, samples in zip(prompts, outputs):
            response = samples[0]["text"] if samples else ""
            rows.append({
                "category": category,
                "condition": phase,
                "question": prompt,
                "response": response,
                "refusal_heuristic": base.is_refusal(response),
                "human_harmful_compliance": "",
                "human_unnecessary_refusal": "",
                "review_notes": "",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("base", "lora"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for vLLM inference.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = Path(BASE_MODEL if args.phase == "base" else LORA_MODEL)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    calibration = OUT_DIR / "calibration.json"
    if not calibration.is_file():
        bundled = HERE / "baseline_calibration.json"
        if not bundled.is_file():
            raise FileNotFoundError("Frozen reproduction border calibration is missing from the source bundle.")
        calibration.write_bytes(bundled.read_bytes())
    calibration_data = json.loads(calibration.read_text(encoding="utf-8"))
    if calibration_data.get("model_id") != base.MODEL_ID:
        raise ValueError("The bundled border bank was not calibrated for Qwen2.5-0.5B-Instruct.")
    border_inputs = calibration_data["border_inputs"]

    # Make the base reference/control first, then the LoRA treatment. Existing
    # files are reused so an interrupted notebook run can continue cheaply.
    sample_path = OUT_DIR / f"{args.phase}_border_samples.json"
    eval_path = OUT_DIR / f"{args.phase}_safety_eval.csv"
    need_samples = args.phase == "base" and not sample_path.exists()
    need_eval = not eval_path.exists() or eval_path.stat().st_size == 0
    if not need_samples and not need_eval:
        print(f"{args.phase} samples and evaluation already exist; reusing {OUT_DIR}.")
        return

    base.MODEL_DIR = str(model_dir)
    print(f"Phase={args.phase}; model={model_dir}; frozen border prompts={len(border_inputs)}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    llm = LLM(
        model=str(model_dir), dtype="float16", max_model_len=base.MAX_MODEL_LEN,
        gpu_memory_utilization=float(os.environ.get("B3IT_LORA_VLLM_MEMORY", "0.75")),
        enforce_eager=True, disable_log_stats=True,
    )
    if need_samples:
        samples = {
            "neutral_reference": base.generate_samples(
                llm, tokenizer, border_inputs, base.NEUTRAL_SYSTEM, POOL_SAMPLES, temperature=0.9
            ),
            "neutral_control": base.generate_samples(
                llm, tokenizer, border_inputs, base.NEUTRAL_SYSTEM, POOL_SAMPLES, temperature=0.9
            ),
        }
        save_json(sample_path, samples)
        print(f"Saved baseline null and reference samples: {sample_path}")
    elif args.phase == "lora" and not sample_path.exists():
        samples = {
            "lora_treatment": base.generate_samples(
                llm, tokenizer, border_inputs, base.NEUTRAL_SYSTEM, POOL_SAMPLES, temperature=0.9
            )
        }
        save_json(sample_path, samples)
        print(f"Saved LoRA treatment samples: {sample_path}")

    if need_eval:
        rows = run_eval(llm, tokenizer, args.phase)
        with eval_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} held-out safety responses: {eval_path}")
    # Exit after each phase; the notebook starts the next model in a fresh
    # process, which releases all vLLM/CUDA memory before training/evaluation.
    print("Phase finished. The process will release the GPU when it exits.")


if __name__ == "__main__":
    main()
