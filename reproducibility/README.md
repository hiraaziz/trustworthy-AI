# Scaled reproduction of Figure 1
Open `results/figure1_combined/figure1_three_models.pdf` to see the final result.
The experiment covers Qwen, Gemma, and Phi-4-mini, with four perturbations each.
Each of the 20 points averages 12 model–variant AUCs.

To redraw the graph on CPU with Python and Matplotlib, run from the extracted folder:
`python scripts/plot_three_models_figure1.py results/figure1_combined`

This review package includes the source, experiment scripts, model notebooks,
settings, and result tables. Model weights and raw observations are excluded.
