# Figures 2 and 3: scaled reproduction code

This folder contains the runnable source and Kaggle notebook for the Qwen2.5-0.5B-Instruct reproduction. It commits code and configuration only; generated plots and run outputs are created in the Kaggle session and are not stored in this repository.

## Experiment settings

The run compares B3IT with MET at $T=1$ and $T=0$, MMLU-ALG at $T=0.1$ and $T=0$, and LT. It collects a pool of 100 one-token observations per prompt (50 reference and up to 50 detection samples). Border-input discovery checks 2,048 candidates, with three samples per candidate, and keeps up to 30 border inputs. B3IT is evaluated over prompt counts 1, 3, 5, and 10 and detection budgets 1, 3, 10, 20, and 50. Baseline parameter sweeps and 1,000 resampling trials with 250 bootstrap replicates are configured in `source/track-llm-apis/reproduction/run_scaled_reproduction.py`.

For the fine-tuning perturbation, full-weight training is disabled and LoRA is enabled. The LoRA sweep uses 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512 LMSYS-Chat-1M examples, one epoch, batch size 1, learning-rate scale $10^{-6}$, sequence length 128, rank 8, and target modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`. With batch size 1 and one epoch, the number of examples is also the number of update steps. This is a resource-scaled LoRA experiment, not the paper's full-weight fine-tuning setup.

## Run on Kaggle

1. Import `notebooks/qwen05_figures_2_3.ipynb` into Kaggle. Enable Internet and a GPU accelerator. The run can use a second GPU for LoRA; with one GPU, LoRA training falls back to CPU.
2. In the notebook's setup cell, set `REPO_REF` to this branch while reviewing the PR, or to `main` after the PR is merged.
3. Add a Kaggle secret called `HF_TOKEN` using a Hugging Face read token. The token is used to access LMSYS-Chat-1M. Accept the dataset's terms on Hugging Face first.
4. Run the package-install cell and restart the kernel when Kaggle requests it. Then rerun the setup cell, version check, model/data access check, and reproduction cell in order.
5. The last cell creates an archive in `/kaggle/working/` containing the generated interactive HTML plots and sampled run data. These are run outputs; they are not committed to the repository.

The package snapshot is under `source/track-llm-apis/`. The notebook clones this repository and uses that source tree. Use a fresh Kaggle working directory for a new run; the run ID is recorded in the output manifest.
