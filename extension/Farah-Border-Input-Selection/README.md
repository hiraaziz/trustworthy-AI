# Border Input selection — Farah Bouzrara

This contribution studies performance-based selection of three Border Inputs from ten candidates on Qwen2.5-0.5B-Instruct and Phi-4-mini-instruct.

## Start here

Install `requirements-analysis.txt`, then open the numbered notebooks in `notebooks/`.
Set `B3IT_DATA_ROOT` or edit `DATA_ROOT` in the first code cell. The default is a mounted Colab Drive. CPU analysis does not need torch, vLLM, a GPU, or model weights.

1. `01_main_results_and_subset_ranks.ipynb`: report table from recorded CSVs, ranks among 120 triples, oracle gaps, and the Phi seed-303 diagnostic.
2. `02_single_family_transfer.ipynb`: six transfer directions, training-variant budget curves, Spearman analysis, and full transfer grids.
3. `03_output_count_sensitivity.ipynb`: nested 250/500/1000-output sensitivity check. Requires notebook 02 saved results.

## Required external data

The uploaded notebooks contain code and historical outputs, not the raw result JSON files. This package does not include those data. The repository is not yet a self-contained reproduction of the numerical results until the following inputs are supplied.

Under DATA_ROOT, retain `B3IT_Qwen05_Noise_Pruning/expanded_scales_1000samples_v1/` and `B3IT_Phi4mini_Noise_Pruning/expanded_scales_1000samples_v1/`.
Each collection needs `manifest.json`, `unchanged.json`, and the 17 named perturbation JSON files with `config`, `job`, and token `counts`.
The `extension_selection_repeated_v1/` subfolder needs the original `protocol.json`, `seed_101.json`, `seed_202.json`, `seed_303.json`, `seed_404.json`, `seed_505.json`, and `all_seeds_comparison.csv`.
Notebook 02 creates/checks `extension_single_family_v1/`; notebook 03 creates/checks `extension_output_size_sensitivity_v1/`. Results remain beside the input collections. Existing protocol checks are preserved; do not delete protocols merely to bypass an assertion.

## Experimental meaning

Main experiment: three-fold leave-one-perturbation-family-out validation. Five predefined seeds partition 1000 recorded outputs into disjoint 500/500 pools. Within a pool, multinomial draws simulate 50 reference and 10 detection observations per input. Detection budget is 30 observations per test. Performance selection ranks individual-input training AUC; entropy selection uses unchanged-output Shannon entropy. The random comparator averages all 120 triples.

TV is the detection score; held-out ROC AUC measures discrimination. Gains and subset ranks describe the benefit of selection. Across-seed SD is not a confidence interval. Spearman compares entropy and single-training-family AUC rankings, not detection superiority. Seeds reuse the same collection. Offline perturbation construction requires white-box access; total acquisition cost is not demonstrated lower.

## Collection history

`history/` preserves the Qwen and Phi working code in original order, with outputs removed. These notebooks include discovery, preliminary runs, recovery, tokenizer audits, and earlier analysis. They are documentation of execution history, not clean Run all pipelines. They contain original Colab paths and temporary-state dependencies. Collection used upstream commit `ad2f69295797a5f03e7b480246080d91087d9007`, torch 2.9.1, transformers 4.57.6, vLLM 0.15.0, tokenizers 0.22.2, datasets 4.5.0, numpy 2.2.6. GPU collection was not rerun during cleanup.

## Validation and remaining work

Notebook schemas and Python syntax were checked locally. Analysis cells were compared with their source cells; numerical analyses were not executed because Drive data are unavailable here. No new experimental result is claimed. Before submission, run the numbered notebooks against the saved collections and compare regenerated CSVs with the original report values. The main two-family experiment's original runners remain in history; notebook 01 summarizes their saved output.

Upload this whole folder under `extension/Farah-Border-Input-Selection/` in the group repository, preferably on a contribution branch. No GitHub upload has been performed by this package.
