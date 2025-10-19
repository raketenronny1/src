# AI Agent Instructions for Meningioma FT-IR Classification Project

## Project Overview
This MATLAB codebase analyzes Fourier-transform infrared (FT-IR) spectra of meningioma tissue to classify tumor grades. The project follows a structured pipeline approach with distinct phases for preprocessing, model selection, evaluation, and interpretation.

## Key Architecture Components

### Pipeline Structure
- Core pipeline implemented as `ClassificationPipeline` class combining:
  - Binning transformer (`BinningTransformer`)
  - Feature selector (implementations in `pipelines/*.m`)
  - LDA classifier
- Pipeline phases executed through top-level scripts:
  - `run_phase2_model_selection.m`
  - `run_phase3_final_evaluation.m` 
  - `run_phase4_feature_interpretation.m`

### Configuration System
- Main config in `config/project_config.json`
- Environment-specific overrides in `config/*.yaml`
- Use `configure_cfg.m` to load and merge configurations
- Access config via `cfg` struct throughout codebase

## Development Workflow

### Prerequisites
1. MATLAB R2021b+ with required toolboxes:
   - Statistics and Machine Learning Toolbox
   - Signal Processing Toolbox
   - (Optional) Parallel Computing Toolbox

### Project Structure
```
data/            # MATLAB data files (required)
data/raw/        # Optional raw spectra
results/         # Output tables and logs
models/          # Saved models
figures/         # Generated plots
```

### Common Operations
1. Initialize configuration:
   ```matlab
   addpath('src');
   cfg = configure_cfg();
   cfg.useOutlierRemoval = true;  % Example override
   ```

2. Execute pipeline phases:
   ```matlab
   run('src/main.m')  % Full pipeline
   % Or individual phases:
   run('src/run_phase2_model_selection.m')
   ```

## Key Patterns and Conventions

### Feature Selection and Model Building
- Available feature selectors:
  - `FisherFeatureSelector`: Ranks features using Fisher score
  - `MRMRFeatureSelector`: Minimum redundancy maximum relevance
  - `PCAFeatureSelector`: Principal component analysis
  - `NoFeatureSelector`: Passthrough for baseline comparison
- Feature selection configured via `hyperparams` struct:
  ```matlab
  hyperparams.fisherFeaturePercent = 0.1; % Use top 10% features
  ```

### Cross-validation and Evaluation
- Nested cross-validation structure:
  - Outer CV: Model selection (`phase2.outerFolds` folds)
  - Inner CV: Hyperparameter tuning (`phase2.innerFolds` folds)
- Probe-level validation to prevent data leakage
- Key performance metrics:
  - AUC: Area under ROC curve
  - F2_WHO3: F2 score for WHO grade 3 detection
  - Custom metrics defined in `calculate_performance_metrics.m`
- Configure metrics in `project_config.json`:
  ```json
  {
    "metricPresets": {
      "phase2_model_selection": ["AUC", "F2_WHO3"],
      "phase3_final_evaluation": ["AUC", "F2_WHO3", "Sensitivity", "Specificity"]
    }
  }
  ```

### Data Preprocessing
1. Spectral preprocessing:
   - Binning via `BinningTransformer`
   - PCA-based outlier detection:
     - T² statistic: Checks distance in PCA subspace
     - Q statistic: Measures reconstruction error
     - Joint T²/Q thresholding with configurable significance level
     - Cache-aware PCA via `cached_pca.m`
2. Dataset organization:
   - Training/test splits via `load_dataset_split.m`
   - Probe-level grouping to prevent data leakage
   - Missing value handling with mean imputation

### Visualization and Reporting
1. Model evaluation plots:
   - Spider plots for multi-metric comparisons (`visualize_model_comparison_spiderplots.m`)
   - Confusion matrices with customizable normalization
   - ROC curves and precision-recall plots
2. Spectral analysis tools:
   - Single-probe visualization (`vis_quick_plotting_template.m`)
   - Comparative heatmaps for group differences
   - PCA exploration via `exploratory_pca_mwu_analysis.m`
3. Export capabilities:
   - CSV/JSON result exports with `export_phase_results_to_csv_json.m`
   - MATLAB structure exports for further processing
   - Publication-ready figures with consistent styling

### Data Management
- Input data expected as MATLAB `.mat` files in `data/`
- Results exported through utilities in `data_management/`
- Use `helper_functions/load_dataset_split.m` to load training/test splits

### Pipeline Extensions
- New feature selectors should inherit from `pipelines.FeatureSelector`
- Pipeline components must implement standard train/transform interface
- Example: See `pipelines/MRMRFeatureSelector.m` for reference

### Logging and Progress
- Use `helper_functions/log_message.m` for consistent logging
- Progress tracking via `ProgressReporter` class
- Pipeline-specific logging through `log_pipeline_message.m`

### Parallel Processing
1. Configuration:
   - Enable via `cfg.parallelProcessing = true`
   - Set chunk size with `get_cfg_chunk_size.m`
   - Environment detection through `get_parallel_environment_info.m`
2. Parallelization strategies:
   - Outer CV folds run in parallel
   - Chunk-based processing for large datasets
   - Cache-aware operations for PCA and feature selection
3. Performance considerations:
   - Memory usage monitored via `ProgressReporter`
   - Parallel logging with `create_parallel_logger.m`
   - Configurable chunk sizes for memory-intensive operations

## Common Pitfalls
1. Ensure all required MATLAB data files exist in `data/` before running
2. Watch memory usage when processing large spectra without binning
3. Configure parallel processing appropriately via `cfg.parallelProcessing`