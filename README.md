# Meningioma FT-IR Classification Project

This repository contains MATLAB code for analysing Fourier-transform infrared (FT-IR) spectra of meningioma tissue in order to discriminate tumour grades. The code is organised into a series of phases that perform preprocessing, model selection, final evaluation and interpretation.

## Directory layout

- `import_preprocessing/` – scripts for data preparation and preprocessing.
- `data_management/` – utilities to export file lists and intermediate results.
- `helper_functions/` – refactored functions used throughout the pipeline.
- `plotting/` – scripts for generating project figures.
- `archive/` – older scripts retained for reference.

- Top-level scripts such as `run_phase2_model_selection.m`, `run_phase3_final_evaluation.m` and `run_phase4_feature_interpretation.m` implement the main phases of the analysis.


The project expects the following folders in the repository root when running the scripts:

```
data/            % MATLAB data files
data/raw/        % optional raw spectra location
results/         % output tables and logs
models/          % saved models
figures/         % generated plots
```

## Required Inputs

The core scripts rely on MATLAB data files stored under `data/`:

- `data_table_train.mat`
- `data_table_test.mat`
- `wavenumbers.mat`
- `*_training_set_no_outliers*.mat` when using outlier removal

## Requirements

- MATLAB R2021b or newer
- **Statistics and Machine Learning Toolbox** (for LDA, cross-validation and `fscmrmr`)
- **Signal Processing Toolbox** (for spectral smoothing via Savitzky–Golay filtering)
This repository includes a lightweight spider plot helper function for generating radar charts; no external downloads are required.

## Configuration helper

Use the `configure_cfg.m` function to create or update the `cfg` structure that
controls the main scripts. It fills in sensible defaults for missing fields and
accepts name/value pairs for overrides.

```matlab
cfg = configure_cfg();
cfg.useOutlierRemoval = true;   % set false to keep all training data
cfg.parallelOutlierComparison = true; % evaluate both cleaned and full datasets in parallel
run('src/main.m')
```

## Analysis pipeline

All scripts assume the MATLAB current folder is set to the project root. Each phase can be run independently once the required input files are available.

### Phase 1 – Data preparation

Use the scripts in `import_preprocessing/` to load raw spectra, preprocess them and create the training and test sets.

```matlab
run('src/import_preprocessing/run_ftir_data_preparation_pipeline.m')
run('src/import_preprocessing/run_split_training_test.m')
```

Optionally run `import_preprocessing/run_pca_overview.m` to perform a PCA on the training spectra and generate two scatter plots showing WHO‑1 and WHO‑3 distributions.

### Phase 2 – Model and feature selection

Run nested cross-validation using:
The Fisher ratio and MRMR pipelines now select a percentage of the available features rather than a fixed count.

```matlab
run('src/run_phase2_model_selection.m')
```

Set `cfg.parallelOutlierComparison = true` before running the phase to train
each pipeline twice – once on the full training data and once after removing
spectra that Hotelling's T² and the Q-statistic both flag as outliers. Separate
results and model folders are created for each variant so that Phase 3 can
compare their test-set performance side by side.

Results are saved under `results/Phase2` and models under `models/Phase2`.

### Phase 3 – Final evaluation

Train the MRMR–LDA pipeline on the full training set and evaluate on the test set.
MRMR features are chosen based on a percentage of the binned spectrum rather than a fixed count.

```matlab
run('src/run_phase3_final_evaluation.m')
```

Models are stored in `models/Phase3` and metrics in `results/Phase3`.

### Phase 4 – Feature interpretation

Interpret the trained model by plotting LDA coefficients for the selected wavenumbers and generating mean spectra visualisations. Statistical p-value calculations were removed to streamline the phase.

```matlab
run('src/run_phase4_feature_interpretation.m')
```

Outputs appear in `results/Phase4` and `figures/Phase4`.

### Visualizing project results

After completing Phases 2–4 you can summarise the pipeline outputs with
`plotting/visualize_project_summary.m`. The repository ships with a small
`spider_plot_R2019b.m` utility used by this script, so no external dependencies
are needed. Running the script generates publication-ready plots under
`figures/ProjectSummaryFigures`.

A helper menu `plotting/run_visualization_menu.m` lets you choose which figures
to create. Other scripts that produce figures are:

- `plotting/visualize_phase1.m` – requires `data/wavenumbers.mat` and
  `data/data_table_complete.mat` and writes Phase 1 plots to
  `figures/Phase1_Dissertation_Plots`.
- `plotting/visualize_binning_effects.m` – visualises the effect of different
  binning factors using `data/data_table_train.mat` and outputs to
  `figures/SideQuests`.
- `plotting/visualize_model_comparison_spiderplots.m` – generates spider
  plots comparing model performance for AUC and F2\_WHO3 across pipelines.

Example usage:

```matlab
% Interactive menu
run('src/plotting/run_visualization_menu.m')

% Create summary plots after Phase 4
run('src/plotting/visualize_project_summary.m')

% Phase 1 figures
run('src/plotting/visualize_phase1.m')

% Binning effect visualisation
run('src/plotting/visualize_binning_effects.m')
```


### Exporting cross-phase summaries

After running Phases 2 and 3 you can convert the MATLAB structs into flat CSV
and JSON artefacts with:

```matlab
run('src/data_management/export_phase_results_to_csv_json.m')
```

The helper inspects the latest Phase 2 and Phase 3 result files (or accepts
explicit paths via the optional configuration struct) and writes the following
exports under `results/Exports/`:

- `phase2_pipeline_leaderboard.csv` – mean cross-validation metrics and
  hyperparameter summaries for every trained pipeline.
- `phase3_variant_model_metrics.csv` – test-set metrics by variant, model set
  and pipeline, including cross-validation references when available.
- `phase3_best_models.csv` – one-row-per-variant summary of the top F2 model.
- `phase_metrics_bundle.json` – machine-friendly bundle that preserves the
  nested structure, including per-fold scores, probe aggregates and relative
  paths back to the source MAT files.

Provide the optional `cfg.phase2ResultsFile`, `cfg.phase3ResultsFile` or
`cfg.exportRoot` fields to override the auto-discovery defaults.

## Cleaning Outputs

Files in `results/`, `models/`, and `figures/` can accumulate across runs. Periodically remove or archive old outputs to keep the repository tidy. The `cleanup_results` helper automates this process.

## Refactored helper functions

Reusable helper functions in `helper_functions/` include:

```matlab
[specB, wnB] = bin_spectra(rawSpec, wn, 5);           % Spectral binning
FR = calculate_fisher_ratio(specB, labels);           % Feature ranking
M = calculate_performance_metrics(yTrue, yPred, scores(:,2), 3, {'Accuracy','AUC'});
[bestParams, perf] = perform_inner_cv(Xtrain, ytrain, probeIDs, config, wn, 5, {'F2_WHO3','Accuracy'});
```

These routines can be incorporated in custom scripts or the provided pipeline.

