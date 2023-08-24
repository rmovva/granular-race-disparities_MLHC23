# granular-race-disparities_MLHC23
Code for "Coarse race data conceals disparities in clinical risk score performance," published at MLHC 2023.  
Link to paper: https://arxiv.org/abs/2304.09270

![GIF showing performance disparities that are revealed upon looking at granular groups.](https://github.com/rmovva/granular-race-disparities_MLHC23/blob/main/granular_race_gif_1%20(1).gif)

## How to reproduce our results

1. First, download MIMIC-IV-ED (this project was run with version 2.2). You'll need the hosp, icu, and ed modules.
2. Run the [extract_main_dataset.ipynb](https://github.com/rmovva/granular-race-disparities_MLHC23/blob/main/preprocessing/extract_main_dataset.ipynb) notebook to generate a preprocessed dataframe for the emergency department prediction tasks that we study in the paper.
3. Run the [collect-and-plot-granular-performance-metrics-ML.ipynb](https://github.com/rmovva/granular-race-disparities_MLHC23/blob/main/analysis/collect-and-plot-granular-performance-metrics-ML.ipynb) notebook to train logistic regressions on the outcomes, store performance metrics, and plot results (to reproduce **Figure 1**).
4. Run the [compute-significance-and-compare-amount-of-variation.ipynb](https://github.com/rmovva/granular-race-disparities_MLHC23/blob/main/analysis/compute-significance-and-compare-amount-of-variation.ipynb) notebook to reproduce **Table 2** and **Figure 2**.
