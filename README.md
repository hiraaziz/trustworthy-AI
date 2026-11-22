# B3IT Reproduction and Extensions

This repository contains reproduction studies and extensions of B3IT, a black-box method for detecting changes in language models through sampled outputs.

## Reproduction studies

- `reproducibility/Figure1/` reproduces the cost–accuracy experiment across three models.
- `reproducibility/Figure2-3-Scaled-Comparison/` provides source code and a Kaggle notebook for baseline comparisons and LoRA fine-tuning experiments.

## Extensions

- `extension/Hira-Safety-Prompt/` studies safety-prompt and guard interventions.
- `extension/Farah-Border-Input-Selection/` studies border-input selection, ranking, and transfer.
- `extension/Youssef-Safety-Behavior/` evaluates safety system-prompt and safety-LoRA interventions.

The experiment directories describe their methods, run instructions, and limitations. Some notebooks generate plots and other artifacts when run. Model weights and raw question–response records are not included.
