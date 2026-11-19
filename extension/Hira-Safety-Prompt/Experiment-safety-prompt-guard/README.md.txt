# Safety change detection with B3IT

These experiments test whether B3IT detects system-prompt changes or the addition of a hidden input guard.

| Experiment | Notebook to run | Saved result |
|---|---|---|
| Add safety instructions of different lengths | [experimentA.ipynb](Experiment-different-length-token/notebook/experimentA.ipynb) | [B3IT vs. LT figure](Experiment-different-length-token/b3it_vs_lt_captioned.png) |
| Replace one token in the system prompt | [ExperimentA_Token_Edits.ipynb](Experiment-single-token-replacement/notebooks/ExperimentA_Token_Edits.ipynb) | [B3IT vs. LT figure](Experiment-single-token-replacement/b3it_vs_lt_.png) |
| Add a hidden input guard | [ExperimentB.ipynb](Experiment-safety-prompt-guard/ExperimentB.ipynb) | [ROC curve](Experiment-safety-prompt-guard/hidden_guard_roc.png) |

## Run an experiment

1. Clone this repository and open the relevant notebook in JupyterLab on a Linux machine with a CUDA-capable NVIDIA GPU. The notebooks use `/workspace` for experiment files. The hidden-guard notebook checks that `/workspace` is mounted.
2. Run the setup, data collection, scoring, and plotting cells in order. If prompted, use a Hugging Face read token from an account with access to `lmsys/lmsys-chat-1m`.
3. Review the scores and figures in the experiment's `results/` directory. For the single-token replacement experiment, see [REPRODUCE.md](Experiment-single-token-replacement/REPRODUCE.md) to rerun scoring and plotting from the saved observations.

## Check the hidden-guard result without a GPU

From the `Experiment-safety-prompt-guard` folder, run:

    python -m pip install matplotlib scikit-learn
    python scripts/plot_hidden_guard.py

The script reads the saved scores, verifies the ROC-AUC of 0.95834275, and writes the ROC curve to `results/figures/`.