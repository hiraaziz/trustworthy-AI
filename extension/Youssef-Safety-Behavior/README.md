# Safety-behavior extension

This extension studies whether B3IT detects model changes that target safety behavior, and separately measures harmful compliance and unnecessary refusals on held-out questions. The experiments use Qwen2.5-0.5B-Instruct. Each experiment freezes the same baseline border-input bank and compares B3IT with LT on those inputs.

## Experiment 1: safety system prompt

The base model weights are unchanged. The neutral prompt is `You are a helpful assistant.` The treatment prompt is defined verbatim in `scripts/run_safety_prompt_1a_v2.py`; it instructs the model to answer ordinary and benign requests helpfully, refuse specified harmful assistance, and offer safe lawful alternatives. The evaluation uses 50 harmful and 50 benign questions in each condition. The recorded reviewed outcomes are 3/50 harmful-compliance cases in both conditions and 0/50 unnecessary benign refusals in both.

For change detection, the V1 calibration screened 256 candidate border inputs with 12 samples each and kept the 20 closest first-token ties. V2 reused that frozen bank. Sampling pools contain 50 one-token outputs per prompt; B3IT is evaluated at 1, 3, 10, 20, and 50 samples per prompt. LT uses top-20 log probabilities. Detection thresholds and rates use 300 resampling trials. The refusal-keyword plot is an automatic screening proxy, not the reviewed safety outcome.

## Experiment 2: safety LoRA

LoRA changes adapter weights; the base weights remain frozen. During both base and adapter evaluation the system prompt is neutral. The training script uses English harmful/refusal pairs from `iNLP-Lab/multilingual-safety` and benign/helpful pairs from `AmazonScience/FalseReject`, and excludes the held-out evaluation questions. The report's 1:3 run used 256 harmful/refusal and 768 benign/helpful examples, one epoch, per-device batch size 1, gradient accumulation 8, rank 8, and learning rate $10^{-4}$. This was LoRA fine-tuning, not full-model fine-tuning.

On the separately reviewed 50+50 question set, the base model had 9/50 harmful-compliance cases and 5/50 unnecessary benign refusals; the 1:3 LoRA had 3/50 and 32/50, respectively. Thus, harmful compliance was lower in this sample, while unnecessary refusals rose substantially. Treat these as pilot results, not a general safety claim. The LoRA detection chart transcribes approximate B3IT values from the Kaggle plot because the raw detector metrics were not available locally; LT and false-alarm traces in that image are likewise plot readings.

## Files

- `notebooks/experiment1_safety_system_prompt.ipynb` and `notebooks/experiment2_safety_lora.ipynb`: Kaggle workflows.
- `scripts/`: experiment runners, training/evaluation scripts, and the frozen calibration file.
- `results/figures/`: reported plots.
- `results/reviewed_outcomes_summary.csv`: aggregate counts only; question text and model responses are not included.

The notebook workflows require a Kaggle GPU and the source/runtime setup described in their cells. The training datasets are external; consult their dataset cards and licenses before reuse. No model weights or raw prompt-response records are included.
