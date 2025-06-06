# Meningioma FT-IR Classification Project

This repository contains MATLAB code for analysing Fourier-transform infrared (FT-IR) spectra of meningioma tissue in order to discriminate tumour grades. The code is organised into a series of phases that perform preprocessing, model selection, final evaluation and interpretation.

## Directory layout

- `import_preprocessing/` – scripts for data preparation, preprocessing and outlier detection.
- `data_management/` – utilities to export file lists and intermediate results.
- `helper_functions/` – refactored functions used throughout the pipeline.
- `plotting/` – scripts for generating project figures.
- `archive/` – older scripts retained for reference.
- Top-level scripts such as `run_phase2_model_selection_comparative.m`, `run_phase3_final_evaluation.m` and `run_phase4_feature_interpretation.m` implement the main phases of the analysis.

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

## Analysis pipeline

All scripts assume the MATLAB current folder is set to the project root. Each phase can be run independently once the required input files are available.

### Phase 1 – Data preparation

Use the scripts in `import_preprocessing/` to load raw spectra, preprocess them and create the training and test sets.

```matlab
run('src/import_preprocessing/run_ftir_data_preparation_pipeline.m')
run('src/import_preprocessing/run_split_training_test.m')
```

Optional scripts such as `run_outlier_detection_pca2.m` or `run_apply_consensus_outlier_strategy.m` help detect and remove outliers before model training.

### Phase 2 – Model and feature selection

Run nested cross-validation and compare outlier strategies using:

```matlab
run('src/run_phase2_model_selection_comparative.m')
```

Results are saved under `results/Phase2` and models under `models/Phase2`.

### Phase 3 – Final evaluation

Train the MRMR–LDA pipeline on the full training set and evaluate on the test set.
The helper script `run_phase3_final_evaluation_from_phase2.m` automatically
loads the best pipeline configuration saved during Phase&nbsp;2 and applies the
corresponding outlier strategy.

```matlab
run('src/run_phase3_final_evaluation_from_phase2.m')
```

Legacy scripts remain for explicitly running the two predefined strategies:

```matlab
run('src/run_phase3_final_evaluation.m')            % AND strategy
run('src/run_phase3_final_evaluation_OR_strategy.m') % OR strategy
```

Models are stored in `models/Phase3` and metrics in `results/Phase3`.

### Phase 4 – Feature interpretation

Interpret the trained model by plotting LDA coefficients for the selected wavenumbers.

```matlab
run('src/run_phase4_feature_interpretation.m')
```

Outputs appear in `results/Phase4` and `figures/Phase4`.

## Refactored helper functions

Reusable helper functions in `helper_functions/` include:

```matlab
[specB, wnB] = bin_spectra(rawSpec, wn, 5);           % Spectral binning
FR = calculate_fisher_ratio(specB, labels);           % Feature ranking
M = calculate_performance_metrics(yTrue, yPred, scores(:,2), 3, {'Accuracy','AUC'});
[bestParams, perf] = perform_inner_cv(Xtrain, ytrain, probeIDs, config, wn, 5, {'F2_WHO3','Accuracy'});
```

These routines can be incorporated in custom scripts or the provided pipeline.

