# B3IT reproduction and extensions

This repository contains reproduction code and extensions for the B3IT study of black-box language-model change detection.

## Reproduction

- `reproducibility/Figure1/` contains a three-model reproduction of the cost–accuracy experiment.
- `reproducibility/Figure2-3-Scaled-Comparison/` contains code and a Kaggle notebook for baseline comparisons and LoRA fine-tuning experiments.

## Extensions

- `extension/Hira-Safety-Prompt/` explores safety-prompt and guard interventions.
- `extension/Farah-Border-Input-Selection/` investigates border-input selection and transfer.
- `extension/Youssef-Safety-Behavior/` evaluates safety system-prompt and safety-LoRA interventions.

Experiment folders document their methods, execution steps, and limitations. Notebooks generate plots and run artifacts during execution; model weights and raw question–response records are not stored in the repository.
