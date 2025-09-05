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

## Requirements

- MATLAB R2021b or newer
- **Statistics and Machine Learning Toolbox** (for LDA, cross-validation and `fscmrmr`)
- **Signal Processing Toolbox** (for spectral smoothing via Savitzky–Golay filtering)
- `spider_plot_R2019b` from MATLAB Central File Exchange (<https://www.mathworks.com/matlabcentral/fileexchange/59561-spider_plot>) for radar/spider plots

## Configuration helper

Use the `configure_cfg.m` function to create or update the `cfg` structure that
controls the main scripts. It fills in sensible defaults for missing fields and
accepts name/value pairs for overrides.

```matlab
cfg = configure_cfg();
cfg.useOutlierRemoval = true;   % set false to keep all training data
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
`plotting/visualize_project_summary.m`. This script depends on the
`spider_plot_R2019b` helper referenced in the requirements above—download it
from the [MATLAB Central File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/59561-spider_plot)
and add it to your MATLAB path if necessary. Running the script generates
publication-ready plots under `figures/ProjectSummaryFigures`.

A helper menu `plotting/run_visualization_menu.m` lets you choose which figures
to create. Other scripts that produce figures are:

- `plotting/visualize_phase1.m` – requires `data/wavenumbers.mat` and
  `data/data_table_complete.mat` and writes Phase 1 plots to
  `figures/Phase1_Dissertation_Plots`.
- `plotting/visualize_binning_effects.m` – visualises the effect of different
  binning factors using `data/data_table_train.mat` and outputs to
  `figures/SideQuests`.

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

## Refactored helper functions

Reusable helper functions in `helper_functions/` include:

```matlab
[specB, wnB] = bin_spectra(rawSpec, wn, 5);           % Spectral binning
FR = calculate_fisher_ratio(specB, labels);           % Feature ranking
M = calculate_performance_metrics(yTrue, yPred, scores(:,2), 3, {'Accuracy','AUC'});
[bestParams, perf] = perform_inner_cv(Xtrain, ytrain, probeIDs, config, wn, 5, {'F2_WHO3','Accuracy'});
```

These routines can be incorporated in custom scripts or the provided pipeline.

